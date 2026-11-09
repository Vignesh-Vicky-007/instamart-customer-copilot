"""Part D: tool-using refund agent with limits, checkpoints, crash-and-resume, and an audit ledger.

Everything is offline and deterministic: the "LLM" is a rule-based planner (`plan`) that returns the next action as JSON,
tool latency is simulated by a fake clock, and no network call or API key is used anywhere.

Agent loop (one iteration = one LLM call, then at most one tool execution):
    build prompt -> leak check -> count tokens -> planner picks {"tool", "args"} or {"final", "answer"} -> run tool -> checkpoint
Run statuses:
    REFUND_ISSUED | ESCALATED | NO_REFUND                       (normal outcomes)
    FALLBACK_MAX_ITERATIONS | FALLBACK_TOKEN_LIMIT | FALLBACK_TIMEOUT   (a limit was hit -> case handed to a human)
"""
import json
from pathlib import Path

import guardrails as gr
import triage
from config import OUTPUT_DIR, ROOT

STATE_DIR = ROOT / "state"
TOKEN_LIMIT = 6000          # default budget of prompt + completion tokens per run
MAX_ITERATIONS = 8          # default cap on LLM calls per run
TIMEOUT_SECONDS = 30.0      # default wall-clock budget per run (simulated clock)
LATE_MINUTES = 45           # refund_policy: delivered later than this -> full refund
AUTO_APPROVE_LIMIT = 1000   # refund_policy: above this INR a human must approve
TOOL_LATENCY = {"lookup_order": 0.2, "check_policy": 0.5, "compute_refund": 0.1, "issue_refund": 0.4, "escalate_to_human": 0.3}

SYSTEM_PROMPT = ("You are a refund agent for a quick-commerce app. Use ONLY these tools: lookup_order, check_policy, compute_refund, "
                 "issue_refund, escalate_to_human. Refund only completed orders delivered late; escalate fraud, unknown orders and "
                 "refunds above the approval limit. Reply with one JSON action per turn.")


class CrashError(RuntimeError):
    """Injected failure used to demonstrate checkpoint + resume."""


class Ledger:
    """Append-only JSONL audit log of money actions: ISSUED refunds and ESCALATED cases."""

    def __init__(self, path=OUTPUT_DIR / "refund_ledger.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def reset(self):
        self.path.write_text("")

    def add(self, **entry):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


class ToolBox:
    """The five tools. `executions` records every real tool run so tests can count them."""

    def __init__(self, orders, policy_rag, ledger):
        self.orders = orders.set_index("order_id")
        self.rag, self.ledger = policy_rag, ledger
        self.executions = []
        self.scenario_id = None

    def run(self, name, args):
        self.executions.append(name)
        return getattr(self, name)(**args)

    def lookup_order(self, order_id):
        if order_id not in self.orders.index:
            return {"found": False}
        o = self.orders.loc[order_id]
        return {"found": True, "status": o["status"], "delivery_minutes": None if str(o["delivery_minutes"]) in ("<NA>", "nan") else float(o["delivery_minutes"]),
                "order_amount": float(o["order_amount"])}

    def check_policy(self, query):
        hits = self.rag.retrieve(query, k=2)
        return {"sources": sorted({s for s, _, _ in hits}), "late_minutes": LATE_MINUTES, "auto_limit": AUTO_APPROVE_LIMIT}

    def compute_refund(self, status, delivery_minutes, order_amount):
        if status != "completed":
            return {"eligible": False, "amount": 0.0, "needs_approval": False, "reason": f"order status is '{status}'; only completed orders are refundable"}
        if delivery_minutes is None or delivery_minutes <= LATE_MINUTES:
            return {"eligible": False, "amount": 0.0, "needs_approval": False, "reason": f"delivered within {LATE_MINUTES} minutes; no late-delivery refund"}
        approval = order_amount > AUTO_APPROVE_LIMIT
        return {"eligible": True, "amount": round(order_amount, 2), "needs_approval": approval,
                "reason": f"delivered in {delivery_minutes:.0f} min (> {LATE_MINUTES})" + ("; above INR 1000, needs human approval" if approval else "")}

    def issue_refund(self, order_id, amount):
        self.ledger.add(scenario_id=self.scenario_id, action="ISSUED", order_id=order_id, amount=amount)
        return {"issued": True, "amount": amount}

    def escalate_to_human(self, reason, order_id=None):
        self.ledger.add(scenario_id=self.scenario_id, action="ESCALATED", order_id=order_id, reason=reason)
        return {"escalated": True}


def plan(ctx):
    """Offline stand-in for the LLM: next action from the request context. Deterministic."""
    hist, cands, cat = ctx["history"], ctx["order_ids"], ctx["category"]
    last = hist[-1] if hist else None
    if last and last["tool"] == "escalate_to_human":
        return {"final": "ESCALATED", "answer": "I have passed your case to a human agent who will contact you."}
    if last and last["tool"] == "issue_refund":
        return {"final": "REFUND_ISSUED", "answer": f"Your refund of INR {last['obs']['amount']:.2f} has been issued to the original payment method."}
    if cat == "Fraud Report":
        return {"tool": "escalate_to_human", "args": {"reason": "Fraud report: security team review (high priority)"}}
    if not cands:
        return {"tool": "escalate_to_human", "args": {"reason": "No order ID given; a human must verify the order"}}
    lookups = [h for h in hist if h["tool"] == "lookup_order"]
    if not lookups or (not last["obs"].get("found", True) and last["tool"] == "lookup_order"):
        tried = {h["args"]["order_id"] for h in lookups}
        remaining = [c for c in cands if c not in tried]
        if not lookups or remaining:
            if len(cands) == 1 and lookups:
                return {"tool": "escalate_to_human", "args": {"reason": "Order not found; a human must verify the order", "order_id": cands[0]}}
            return {"tool": "lookup_order", "args": {"order_id": remaining[0]}}
        return {"tool": "escalate_to_human", "args": {"reason": "None of the order IDs could be found"}}
    order_id, order = lookups[-1]["args"]["order_id"], lookups[-1]["obs"]
    if last["tool"] == "lookup_order":
        return {"tool": "check_policy", "args": {"query": "refund eligibility late delivery approval limit"}}
    if last["tool"] == "check_policy":
        return {"tool": "compute_refund", "args": {"status": order["status"], "delivery_minutes": order["delivery_minutes"], "order_amount": order["order_amount"]}}
    if last["tool"] == "compute_refund":
        r = last["obs"]
        if not r["eligible"]:
            return {"final": "NO_REFUND", "answer": f"No refund: {r['reason']}."}
        if r["needs_approval"]:
            return {"tool": "escalate_to_human", "args": {"reason": f"Refund INR {r['amount']:.2f} {r['reason']}", "order_id": order_id}}
        return {"tool": "issue_refund", "args": {"order_id": order_id, "amount": r["amount"]}}
    return {"tool": "escalate_to_human", "args": {"reason": "Unexpected state"}}


def _prompt(redacted_request, history):
    steps = "\n".join(gr.redact_pii(json.dumps({"tool": h["tool"], "obs": h["obs"]})) for h in history)
    return f"{SYSTEM_PROMPT}\n\nCustomer: {redacted_request}\n{steps}\nNext action:"


class RefundAgent:
    def __init__(self, toolbox, ledger, token_limit=TOKEN_LIMIT, max_iterations=MAX_ITERATIONS, timeout_s=TIMEOUT_SECONDS,
                 latency=None, state_dir=STATE_DIR):
        self.tools, self.ledger = toolbox, ledger
        self.token_limit, self.max_iterations, self.timeout_s = token_limit, max_iterations, timeout_s
        self.latency = {**TOOL_LATENCY, **(latency or {})}
        self.state_dir = Path(state_dir)

    def _ckpt(self, run_id):
        return self.state_dir / f"{run_id}.json"

    def _save(self, run_id, st):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._ckpt(run_id).write_text(json.dumps(st, indent=1))

    def run(self, scenario_id, message, resume=False, crash_after=None, run_id=None):
        """Run the loop. resume=True continues from state/<run_id>.json without re-running finished tools."""
        run_id = run_id or scenario_id
        self.tools.scenario_id = scenario_id
        guard = gr.validate_ticket(message)
        if not guard.allowed:
            return self._finish(scenario_id, "NO_REFUND", f"Request rejected by guardrails ({guard.reason}).", 0, 0, [], run_id)
        ids = []
        for m in gr.ORDER_RE.finditer(message):
            if m.group(0).upper() not in ids:
                ids.append(m.group(0).upper())
        route = triage.triage_ticket(guard.clean_text)
        ctx = {"category": route["category"], "order_ids": ids, "history": []}
        st = {"scenario_id": scenario_id, "history": [], "steps": [], "llm_calls": 0, "tokens_used": 0, "clock": 0.0, "complete": False}
        if resume and self._ckpt(run_id).exists():
            st = json.loads(self._ckpt(run_id).read_text())
        ctx["history"] = st["history"]
        executed_now = 0
        while True:
            if st["clock"] > self.timeout_s:
                return self._fallback("FALLBACK_TIMEOUT", f"timeout after {st['clock']:.1f}s (> {self.timeout_s}s)", st, run_id)
            if st["llm_calls"] >= self.max_iterations:
                return self._fallback("FALLBACK_MAX_ITERATIONS", f"stopped after {self.max_iterations} LLM calls without an answer", st, run_id)
            prompt = _prompt(guard.clean_text, st["history"])
            if gr.contains_pii(prompt):
                raise ValueError("PII leaked into agent prompt")
            action = plan(ctx)
            used = gr.approx_tokens(prompt) + gr.approx_tokens(json.dumps(action))
            st["llm_calls"] += 1
            st["tokens_used"] += used
            if st["tokens_used"] > self.token_limit:
                return self._fallback("FALLBACK_TOKEN_LIMIT", f"token budget exceeded ({st['tokens_used']} > {self.token_limit})", st, run_id)
            if "final" in action:
                st["steps"].append({"step": len(st["steps"]) + 1, "type": "final", "status": action["final"], "answer": action["answer"], "tokens": used})
                return self._finish(scenario_id, action["final"], action["answer"], st["llm_calls"], st["tokens_used"], st["steps"], run_id, st)
            obs = self.tools.run(action["tool"], action["args"])
            executed_now += 1
            st["clock"] += self.latency[action["tool"]]
            st["history"].append({"tool": action["tool"], "args": action["args"], "obs": obs})
            st["steps"].append({"step": len(st["steps"]) + 1, "type": "tool", "tool": action["tool"], "args": action["args"], "observation": obs, "tokens": used})
            self._save(run_id, st)                               # checkpoint after every tool execution
            if crash_after is not None and executed_now >= crash_after:
                raise CrashError(f"injected crash after {executed_now} tool executions")

    def _fallback(self, status, why, st, run_id):
        self.ledger.add(scenario_id=st["scenario_id"], action="ESCALATED", order_id=None, reason=f"{status}: {why}")
        answer = f"I could not finish this automatically ({why}); your case has been handed to a human agent."
        st["steps"].append({"step": len(st["steps"]) + 1, "type": "fallback", "status": status, "answer": answer})
        return self._finish(st["scenario_id"], status, answer, st["llm_calls"], st["tokens_used"], st["steps"], run_id, st)

    def _finish(self, scenario_id, status, answer, llm_calls, tokens, steps, run_id, st=None):
        refund = next((s["observation"]["amount"] for s in steps if s.get("tool") == "issue_refund"), 0.0)
        if st is not None:
            st["complete"] = True
            self._save(run_id, st)
        return {"scenario_id": scenario_id, "status": status, "final_answer": answer, "refund_amount": refund,
                "llm_calls": llm_calls, "tokens_used": tokens, "steps": steps}


def build_scenarios(orders):
    """Seven scenarios chosen deterministically from the cleaned orders. expected = (status, refund_amount)."""
    done = orders[orders["status"] == "completed"]
    late = done[done["delivery_minutes"].astype(float) > LATE_MINUTES]
    s1 = late[late["order_amount"] <= AUTO_APPROVE_LIMIT].iloc[0]
    s2 = late[late["order_amount"] > AUTO_APPROVE_LIMIT].iloc[0]
    s3 = done[done["delivery_minutes"].astype(float) <= LATE_MINUTES].iloc[0]
    s4 = orders[orders["status"] == "cancelled"].iloc[0]
    fake = [f"ORD-9{n:05d}" for n in range(12)]                  # IDs that do not exist
    return [
        {"id": "S1", "title": "Late delivery, amount <= INR 1000 -> auto refund",
         "message": f"{s1['order_id']} was delivered very late, I want a refund.", "expected": ("REFUND_ISSUED", round(float(s1["order_amount"]), 2))},
        {"id": "S2", "title": "Late delivery, amount > INR 1000 -> human approval",
         "message": f"{s2['order_id']} arrived very late, please refund me.", "expected": ("ESCALATED", 0.0)},
        {"id": "S3", "title": "On-time delivery -> no refund",
         "message": f"{s3['order_id']} was delivered late, I want a refund.", "expected": ("NO_REFUND", 0.0)},
        {"id": "S4", "title": "Cancelled order -> no refund",
         "message": f"Please refund {s4['order_id']}, it was delivered late.", "expected": ("NO_REFUND", 0.0)},
        {"id": "S5", "title": "Fraud report -> security team",
         "message": f"There is an unauthorized transaction on my card for {s1['order_id']}. I did not place it.", "expected": ("ESCALATED", 0.0)},
        {"id": "S6", "title": "Unknown order ID -> human verifies",
         "message": "ORD-999999 was delivered late, I want a refund.", "expected": ("ESCALATED", 0.0)},
        {"id": "S7", "title": "Customer lists 12 invalid order IDs -> agent loops until the iteration cap",
         "message": "Refund one of these late orders: " + ", ".join(fake) + ".", "expected": ("FALLBACK_MAX_ITERATIONS", 0.0)},
    ]


def write_runs(results, path=OUTPUT_DIR / "agent_runs.jsonl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({k: r[k] for k in ("scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "steps")} |
                               {"refund_amount": r["refund_amount"]}) + "\n")
