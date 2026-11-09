"""Part D: policy chunking, retrieval (RAG) and the refund agent.

Policies are read from data/policies/*.txt (written by generate_data.py). Retrieval is TF-IDF + cosine
similarity so everything runs offline. `PolicyRAG.build_prompt()` is the grounded prompt an LLM would receive.
"""
import csv
import itertools
import re
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import guardrails as gr
import triage
from config import MODELS_DIR, OUTPUT_DIR, POLICY_DIR, RAW_DIR

MIN_SIMILARITY = 0.10   # below this no policy is considered relevant -> escalate to a human


def load_policies(policy_dir=POLICY_DIR):
    return {p.name: p.read_text(encoding="utf-8").strip() for p in sorted(policy_dir.glob("*.txt"))}


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_by_sentence(policies, size=2, overlap=1):
    """Strategy 1: groups of `size` sentences sliding by size-overlap (keeps rules intact). Returns (source, chunk)."""
    step = max(1, size - overlap)
    out = []
    for name, text in policies.items():
        sents = _sentences(text)
        for i in range(0, max(1, len(sents) - overlap), step):
            out.append((name, " ".join(sents[i:i + size])))
    return out


def chunk_by_window(policies, size=300, overlap=60):
    """Strategy 2: fixed-size character windows with overlap. Returns (source, chunk)."""
    step = size - overlap
    return [(name, text[i:i + size].strip()) for name, text in policies.items()
            for i in range(0, len(text), step) if text[i:i + size].strip()]


class VectorStore:
    """Minimal local vector database: chunk texts + their embedding matrix + cosine-similarity search.

    Embedding = hybrid of two normalised vectors per chunk: sparse TF-IDF (exact wording) and dense LSA / TruncatedSVD
    (semantic overlap). It runs offline; swap `embed()` for a sentence-embedding model to use a neural embedder.
    """
    DENSE_WEIGHT = 0.3

    def __init__(self, chunks):
        self.chunks = chunks                                    # list of (source, text)
        texts = [c for _, c in chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        sparse = self.vectorizer.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=min(24, sparse.shape[0] - 1, sparse.shape[1] - 1), random_state=42).fit(sparse)
        self.sparse = sparse
        self.dense = normalize(self.svd.transform(sparse))

    def embed(self, text):
        sparse = self.vectorizer.transform([text])
        return sparse, normalize(self.svd.transform(sparse))

    def search(self, query, k=3):
        """Top-k (index, similarity) by hybrid cosine similarity."""
        q_sparse, q_dense = self.embed(query)
        sims = (1 - self.DENSE_WEIGHT) * (self.sparse @ q_sparse.T).toarray().ravel() \
            + self.DENSE_WEIGHT * (self.dense @ q_dense.T).ravel()
        order = sims.argsort()[::-1][:k]
        return [(int(i), float(sims[i])) for i in order]

    def save(self, path):
        MODELS_DIR.mkdir(exist_ok=True, parents=True)
        joblib.dump(self, path)


def default_llm(prompt):
    """The LLM step. Offline default: a grounded extractive 'LLM' that answers from the retrieved context only.
    To use a hosted model, replace this with a function that sends `prompt` to it (the prompt is already PII-free)."""
    context = prompt.split("Context:\n", 1)[1].split("\n\nQuestion:", 1)[0]
    first = context.splitlines()[0]
    return re.sub(r"^\[[^\]]+\]\s*", "", first)


class PolicyRAG:
    """Policy documents -> chunks -> embeddings -> vector store -> retrieve -> prompt -> LLM answer (or escalate)."""

    def __init__(self, policies=None, chunker=chunk_by_sentence, llm=default_llm):
        self.policies = policies or load_policies()
        self.chunks = chunker(self.policies)
        self.store = VectorStore(self.chunks)
        self.llm = llm

    def retrieve(self, query, k=3):
        """Top-k (source, chunk, similarity) with similarity above MIN_SIMILARITY."""
        return [(self.chunks[i][0], self.chunks[i][1], sim) for i, sim in self.store.search(query, k) if sim >= MIN_SIMILARITY]

    @staticmethod
    def build_prompt(question, hits):
        context = "\n".join(f"[{src}] {chunk}" for src, chunk, _ in hits)
        return f"Answer using ONLY this policy context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

    def answer(self, question):
        guard = gr.validate_ticket(question)          # same guardrails as Part C before anything is retrieved
        if not guard.allowed:
            return {"answer": "Cannot process this request.", "reason": guard.reason, "sources": []}
        hits = self.retrieve(guard.clean_text)
        if not hits:
            return {"answer": "No policy covers that; handing over to a human agent.", "reason": "no_context", "sources": []}
        prompt = self.build_prompt(guard.clean_text, hits)
        if gr.contains_pii(prompt):
            raise ValueError("PII leaked into RAG prompt")
        return {"answer": self.llm(prompt), "reason": "ok", "sources": sorted({s for s, _, _ in hits}), "prompt": prompt}


class EscalationQueue:
    """Human hand-off. Every escalated case gets an ID and a short REDACTED summary (escalation_policy.txt),
    appended to outputs/escalations.csv for the human support team."""
    FIELDS = ["case_id", "created_at", "ticket_id", "category", "priority", "reason", "summary"]

    def __init__(self, path=OUTPUT_DIR / "escalations.csv", reset=False):
        self.path, self._ids = path, itertools.count(1)
        path.parent.mkdir(exist_ok=True, parents=True)
        if reset or not path.exists():
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.FIELDS)
        else:
            self._ids = itertools.count(sum(1 for _ in open(path, encoding="utf-8")))

    def add(self, redacted_text, reason, category=None, priority=None, ticket_id=None):
        case_id = f"ESC-{next(self._ids):04d}"
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([case_id, datetime.now().isoformat(timespec="seconds"), ticket_id or "", category or "",
                                    priority or "", reason, redacted_text[:200]])
        return case_id


def evaluate_retrieval(rag, eval_csv=RAW_DIR / "policy_eval.csv"):
    """Check retrieval on policy_eval.csv: right document, expected fact present, or 'NONE' for off-topic questions."""
    rows = []
    for _, r in pd.read_csv(eval_csv).iterrows():
        hits = rag.retrieve(r["question"])
        if r["expected_doc"] == "NONE":
            ok_doc = ok_fact = not hits
        else:
            ok_doc = bool(hits) and hits[0][0] == r["expected_doc"]
            ok_fact = any(str(r["expected_fact"]).lower() in c.lower() for _, c, _ in hits)
        rows.append({"question": r["question"], "doc_ok": ok_doc, "fact_ok": ok_fact})
    return pd.DataFrame(rows)


class RefundAgent:
    """Applies refund_policy.txt to the orders table.

    Rules: only COMPLETED orders can be refunded; delivery > 45 min -> full refund of the order amount;
    refund above INR 1000 needs human approval; fraud -> security team; damaged/poor-quality items and billing
    disputes -> human verification; no order found -> human. Every human hand-off is logged in the EscalationQueue.
    Decisions: AUTO_APPROVE, HUMAN_APPROVAL, NO_REFUND, HUMAN_REVIEW, REJECT_INPUT.
    """
    LATE_MINUTES = 45
    AUTO_APPROVE_LIMIT = 1000
    ESCALATE = {"HUMAN_REVIEW", "HUMAN_APPROVAL"}

    def __init__(self, orders, rag, queue=None):
        self.orders = orders.set_index("order_id")
        self.rag = rag
        self.queue = queue

    def decide(self, ticket_text, category=None, order_id=None, ticket_id=None):
        guard = gr.validate_ticket(ticket_text)
        if not guard.allowed:
            return {"decision": "REJECT_INPUT", "reason": guard.reason, "category": None, "refund_amount": 0.0, "case_id": None}
        route = triage.triage_ticket(guard.clean_text)
        category = category or route["category"]
        priority = triage.ROUTING.get(category, triage.ROUTING[triage.UNKNOWN_LABEL])[0]
        order_id = gr.extract_order_id(ticket_text) or order_id     # looked up from raw text; never put in a prompt

        if category == "Fraud Report":
            result = self._result("HUMAN_REVIEW", "Fraud: marked high priority and sent to the security team.", category)
        elif category == "App Crash":
            result = self._result("NO_REFUND", "Technical issue, not a refund request; routed to app support.", category)
        elif category == "Product Quality":
            result = self._result("HUMAN_REVIEW", "Damaged/poor-quality item: a human agent must verify the damage with the "
                                  "customer before any refund (the policy has no automatic rule for it).", category)
        elif order_id is None or order_id not in self.orders.index:
            result = self._result("HUMAN_REVIEW", "No matching order found; a human agent must verify the order.", category)
        elif category == "Payment Issue":
            result = self._result("HUMAN_REVIEW", "Billing dispute: verify charges against the order before refunding.", category)
        else:                                                         # Delivery Delay / Refund Request / Other with a valid order
            result = self._late_delivery(order_id, category)
        result["case_id"] = None
        if result["decision"] in self.ESCALATE and self.queue is not None:
            result["case_id"] = self.queue.add(guard.clean_text, result["reason"], category, priority, ticket_id)
        return result

    def _late_delivery(self, order_id, category):
        order = self.orders.loc[order_id]
        if order["status"] != "completed":
            return self._result("NO_REFUND", f"Order status is '{order['status']}'; only completed orders are refundable.", category)
        minutes, amount = float(order["delivery_minutes"]), float(order["order_amount"])
        if minutes <= self.LATE_MINUTES:
            return self._result("NO_REFUND", f"Delivered in {minutes:.0f} min (<= {self.LATE_MINUTES}); no late-delivery refund.", category)
        if amount > self.AUTO_APPROVE_LIMIT:
            return self._result("HUMAN_APPROVAL", f"Delivered in {minutes:.0f} min; refund INR {amount:.2f} is above "
                                f"INR {self.AUTO_APPROVE_LIMIT}, needs a human support lead.", category, amount)
        return self._result("AUTO_APPROVE", f"Delivered in {minutes:.0f} min (> {self.LATE_MINUTES}): full refund of INR {amount:.2f}.",
                            category, amount)

    @staticmethod
    def _result(decision, reason, category, amount=0.0):
        return {"decision": decision, "reason": reason, "category": category, "refund_amount": amount}


SMALLTALK_HELP = "Ask me about refunds, delivery or payments, or describe a problem with an order."
# Conversational phrases (not support requests). A message is small talk ONLY if it is made up entirely of these phrases,
# so "hi, my order is late" or "thanks, but where is my refund?" still go through the normal support flow.
_SMALLTALK_PHRASES = [
    ("bye", r"good\s*bye|bye+(?:\s*bye+)?|see\s+you(?:\s+(?:later|soon))?|take\s+care|talk\s+to\s+you\s+later"),
    ("thanks", r"thank\s*(?:you|u)|thanks|thx|tysm|ty|many\s+thanks|much\s+appreciated|cheers"),
    ("how_are_you", r"how\s+are\s+(?:you|u)(?:\s+doing)?|how\s+r\s+u|how'?s\s+it\s+going|what'?s\s+up"),
    ("time_greeting", r"good\s+(?:morning|afternoon|evening|night|day)"),
    ("greeting", r"h+i+|he+y+|hello+|hlo|helo|hallo|hiya|howdy|namaste|yo|sup"),
]
_SMALLTALK_RE = [(kind, re.compile(rf"^(?:{pat})\b")) for kind, pat in _SMALLTALK_PHRASES]
_SMALLTALK_FILLER = re.compile(r"^(?:ok|okay|okk|alright|great|awesome|there|team|support|instamart|everyone|all|again|so\s+much|a\s+lot|very\s+much|a\s+ton|"
                               r"bot|assistant|buddy|friend|and|dear|sir|madam|mam)\b")
_SMALLTALK_PRIORITY = ["bye", "thanks", "how_are_you", "time_greeting", "greeting"]     # winner for a combined message


def detect_smalltalk(text):
    """'greeting' | 'time_greeting' | 'how_are_you' | 'thanks' | 'bye' if the WHOLE message is small talk, else None."""
    rest = re.sub(r"[^a-z0-9'\s]", " ", (text or "").lower())
    rest = re.sub(r"\s+", " ", rest).strip()
    found = set()
    while rest:
        for kind, rx in _SMALLTALK_RE:
            m = rx.match(rest)
            if m:
                found.add(kind)
                break
        else:
            m = _SMALLTALK_FILLER.match(rest)
            if not m:
                return None                      # contains something that is not small talk -> normal support flow
        rest = rest[m.end():].strip()
    return next((k for k in _SMALLTALK_PRIORITY if k in found), None)


def smalltalk_reply(kind, text=""):
    if kind == "time_greeting":
        part = re.search(r"good\s+(morning|afternoon|evening|night)", text.lower())
        if part and part.group(1) == "night":
            return "Good night! Thanks for contacting Instamart Support."
        return f"Good {part.group(1) if part else 'day'}! {SMALLTALK_HELP}"
    return {"greeting": f"Hello! {SMALLTALK_HELP}",
            "how_are_you": f"I'm doing well, thank you for asking! {SMALLTALK_HELP}",
            "thanks": "You're welcome! Let me know if there is anything else I can help you with.",
            "bye": "Goodbye! Thanks for contacting Instamart Support. Have a great day!"}[kind]


QUESTION_RE = re.compile(r"\?|^\s*(what|how|who|when|why|where|which|can|is|are|do|does|will|should)\b", re.IGNORECASE)
NEEDS_ORDER = ("Refund Request", "Delivery Delay", "Payment Issue")


class CoPilot:
    """The chat assistant: guardrails -> triage -> small talk | (refund agent | policy RAG) -> reply, escalating to a human when needed."""

    def __init__(self, orders, policy_rag=None, queue=None):
        self.rag = policy_rag or PolicyRAG()
        self.queue = queue if queue is not None else EscalationQueue()
        self.agent = RefundAgent(orders, self.rag, self.queue)

    def respond(self, message, ticket_id=None):
        guard = gr.validate_ticket(message)
        if not guard.allowed:
            msg = {"empty_or_whitespace": "Please type your question first.",
                   "too_long": "Your message is too long. Please shorten it to the key details."}.get(guard.reason, "I cannot process that message.")
            return {"reply": msg, "blocked": True, "reason": guard.reason, "escalated": False}
        route = triage.triage_ticket(guard.clean_text)
        has_order = gr.extract_order_id(message) is not None
        base = {"blocked": False, "category": route["category"], "priority": route["priority"], "action": route["action"],
                "sources": [], "escalated": False, "case_id": None, "redacted_text": guard.clean_text}
        smalltalk = detect_smalltalk(message)
        if smalltalk:                                  # greeting / thanks / bye: no retrieval, no refund logic, no escalation
            return {**base, "reply": smalltalk_reply(smalltalk, message), "decision": "GREETING", "intent": smalltalk,
                    "category": None, "priority": None, "action": None}
        if route["category"] in NEEDS_ORDER and not has_order and not QUESTION_RE.search(message):
            return {**base, "reply": "I can help with that. Please share your order ID (for example ORD-100816) and tell me what went wrong.",
                    "decision": "NEED_ORDER_ID"}
        if route["category"] in ("Fraud Report", "Product Quality") or has_order:
            d = self.agent.decide(message, category=route["category"], ticket_id=ticket_id)
            hits = self.rag.retrieve(guard.clean_text, k=2)
            sources = sorted({h[0] for h in hits})
            reply = {
                "AUTO_APPROVE": f"Good news - {d['reason']} The refund will be credited to your original payment method within 5 working days.",
                "NO_REFUND": f"I checked this against our policy: {d['reason']}",
                "REJECT_INPUT": d["reason"],
            }.get(d["decision"], f"{d['reason']} I have passed your case to a human agent (case {d['case_id']}); they will contact you.")
            sorry = {"Product Quality": "I'm sorry that your item had a problem. ", "Fraud Report": "I'm sorry to hear that - your security matters. ",
                     "Delivery Delay": "I'm sorry your order was late. "}.get(route["category"], "")
            return {"reply": sorry + reply, "blocked": False, "category": route["category"], "priority": route["priority"],
                    "action": route["action"], "decision": d["decision"], "refund_amount": d["refund_amount"],
                    "sources": sources, "escalated": d["case_id"] is not None, "case_id": d["case_id"],
                    "redacted_text": guard.clean_text}
        ans = self.rag.answer(message)
        escalated = ans["reason"] == "no_context"
        case_id = self.queue.add(guard.clean_text, "No policy covers this question", route["category"], route["priority"], ticket_id) if escalated else None
        reply = ans["answer"] if not escalated else f"{ans['answer']} Your case reference is {case_id}."
        return {"reply": reply, "blocked": False, "category": route["category"], "priority": route["priority"],
                "action": route["action"], "decision": "POLICY_ANSWER" if not escalated else "HUMAN_REVIEW",
                "sources": ans["sources"], "escalated": escalated, "case_id": case_id, "redacted_text": guard.clean_text}
