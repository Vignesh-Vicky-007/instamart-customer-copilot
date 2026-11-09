"""Part C: input guardrails + zero-shot / few-shot prompt builders.

VALIDATION RULES (documented explicitly):
  R1  Empty or whitespace-only ticket text           -> SKIP, no model call.
  R2  Ticket longer than MAX_TOKENS (approx.)         -> SKIP, no model call.
  R3  PII (email, phone, order ID) is redacted BEFORE any prompt is built.
  R4  After redaction a final leak check runs; if PII remains the prompt is blocked.
"""
import re
from dataclasses import dataclass

MAX_TOKENS = 200          # approximate token budget for a ticket
CHARS_PER_TOKEN = 4       # rough heuristic (no tokenizer dependency)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\d)")
ORDER_RE = re.compile(r"\bORD[-_]?\d{4,}\b", re.IGNORECASE)

LABELS = ["Refund Request", "Payment Issue", "Delivery Delay", "Product Quality", "Fraud Report", "App Crash"]

FEW_SHOT_EXAMPLES = [
    ("I want my money back for the items I returned.", "Refund Request"),
    ("I was charged twice for my order, please return the extra money.", "Payment Issue"),
    ("My order is very late and the rider is not answering.", "Delivery Delay"),
    ("The milk packet I received was leaking and the eggs were broken.", "Product Quality"),
    ("Someone used my wallet without my permission, this is fraud.", "Fraud Report"),
    ("The app freezes on the payment page and I cannot log in.", "App Crash"),
]

_PROMPT_HEADER = "You are a support triage assistant for a quick-commerce app.\n"


@dataclass
class GuardResult:
    allowed: bool
    reason: str
    clean_text: str = ""


def approx_tokens(text):
    return max(1, len(text) // CHARS_PER_TOKEN) if text else 0


def redact_pii(text):
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = ORDER_RE.sub("[ORDER_ID]", text)
    return PHONE_RE.sub("[PHONE]", text)


def contains_pii(text):
    return any(rx.search(text) for rx in (EMAIL_RE, PHONE_RE, ORDER_RE))


def extract_order_id(text):
    """First order ID in the RAW text (upper-cased), or None. Never pass this into a prompt."""
    match = ORDER_RE.search(text or "")
    return match.group(0).upper() if match else None


def validate_ticket(text, max_tokens=MAX_TOKENS):
    """Apply R1-R4. Only an allowed result carries redacted text that is safe to prompt with."""
    if text is None or not str(text).strip():
        return GuardResult(False, "empty_or_whitespace")
    text = str(text).strip()
    if approx_tokens(text) > max_tokens:
        return GuardResult(False, "too_long")
    clean = redact_pii(text)
    if contains_pii(clean):
        return GuardResult(False, "pii_leak_blocked")
    return GuardResult(True, "ok", clean)


def zero_shot_prompt(clean_text):
    return (
        f"{_PROMPT_HEADER}"
        f"Classify the ticket into exactly one of: {', '.join(LABELS)}.\n"
        "Reply with the label only.\n\n"
        f"Ticket: {clean_text}\nLabel:"
    )


def few_shot_prompt(clean_text, examples=FEW_SHOT_EXAMPLES):
    shots = "\n".join(f"Ticket: {text}\nLabel: {label}" for text, label in examples)
    return (
        f"{_PROMPT_HEADER}"
        f"Classify each ticket into one of: {', '.join(LABELS)}. Reply with the label only.\n\n"
        f"{shots}\n\nTicket: {clean_text}\nLabel:"
    )
