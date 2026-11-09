"""Part C: ticket triage (category + priority + action) and the guarded batch pipeline.

`mock_llm` is an offline stand-in for the LLM so the project runs without API keys. To use a real model pass any
function `llm(prompt) -> label` to `run_guarded_pipeline` / `triage_ticket`; the guardrails, prompts and the JSON
output are identical.
"""
import json

import guardrails as gr

UNKNOWN_LABEL = "Other"   # no keyword matched -> a human decides

# Checked in this order: most specific / most urgent first.
KEYWORDS = {
    "Fraud Report": ["fraud", "unauthorized", "hacked", "did not place", "never made", "never placed"],
    "App Crash": ["crash", "freez", "login", "otp", "error", "not loading"],
    "Product Quality": ["damaged", "broken", "spoilt", "spoiled", "expired", "stale", "rotten", "leaking",
                        "wrong item", "missing item", "bad quality"],
    "Payment Issue": ["charged", "invoice", "bill", "deducted", "payment", "double charge"],
    "Refund Request": ["refund", "money back", "return my money"],
    "Delivery Delay": ["late", "delay", "rider", "not arrived", "delivered", "delivery", "not received", "never received"],
}

# category -> (priority, action); the brief's example: Delivery issue -> Medium -> "Check delivery status"
ROUTING = {
    "Fraud Report": ("High", "Escalate to the security team within 2 hours"),
    "App Crash": ("Low", "Route to app support"),
    "Product Quality": ("Medium", "Ask for order details and verify the damage before any refund"),
    "Payment Issue": ("Medium", "Verify the charge against the order"),
    "Refund Request": ("Medium", "Check refund eligibility against the refund policy"),
    "Delivery Delay": ("Medium", "Check delivery status"),
    UNKNOWN_LABEL: ("Low", "Route to a human agent"),
}

# the raw ticket labels in tickets.csv -> the brief's category names (used only to measure accuracy)
TRUE_LABEL_MAP = {"billing": "Payment Issue", "delivery": "Delivery Delay", "technical": "App Crash",
                  "fraud": "Fraud Report", "none": None}


def mock_llm(prompt):
    """Deterministic keyword classifier; reads only the final ticket in the prompt."""
    ticket = prompt.rsplit("Ticket:", 1)[-1].lower()
    for label, words in KEYWORDS.items():
        if any(word in ticket for word in words):
            return label
    return UNKNOWN_LABEL


def triage_ticket(clean_text, llm=mock_llm, prompt_builder=gr.few_shot_prompt):
    """Classify an ALREADY-VALIDATED ticket -> {"category", "priority", "action"}."""
    label = llm(prompt_builder(clean_text))
    label = label if label in ROUTING else UNKNOWN_LABEL
    priority, action = ROUTING[label]
    return {"category": label, "priority": priority, "action": action}


def classify(clean_text, llm=mock_llm, prompt_builder=gr.few_shot_prompt):
    return triage_ticket(clean_text, llm, prompt_builder)["category"]


def run_guarded_pipeline(tickets, llm=mock_llm, mode="few_shot"):
    """Run guardrails on every ticket; only allowed ones reach the model.
    Returns (rows, model_calls, pii_leaks_into_prompts)."""
    build = gr.few_shot_prompt if mode == "few_shot" else gr.zero_shot_prompt
    rows, calls, leaks = [], 0, 0
    for ticket_id, text in zip(tickets["ticket_id"], tickets["text"]):
        guard = gr.validate_ticket(text)
        row = {"ticket_id": ticket_id, "mode": mode, "allowed": guard.allowed, "reason": guard.reason,
               "tokens_approx": gr.approx_tokens(str(text).strip()) if text else 0,
               "redacted_text": guard.clean_text, "category": None, "priority": None, "action": None}
        if guard.allowed:
            prompt = build(guard.clean_text)
            leaks += gr.contains_pii(prompt)
            row.update(triage_ticket(guard.clean_text, llm, build))
            calls += 1
        row["predicted"] = row["category"]
        rows.append(row)
    return rows, calls, int(leaks)


def prompt_examples(clean_text="My order is delayed. I placed it 2 hours ago, but it has not arrived."):
    """The brief's example ticket through both prompt styles, with the expected JSON."""
    out = triage_ticket(clean_text)
    return (f"EXAMPLE TICKET: {clean_text}\n\n--- ZERO-SHOT PROMPT ---\n{gr.zero_shot_prompt(clean_text)}\n\n"
            f"--- FEW-SHOT PROMPT ---\n{gr.few_shot_prompt(clean_text)}\n\n--- CLASSIFICATION (JSON) ---\n"
            f"{json.dumps(out, indent=2)}\n")
