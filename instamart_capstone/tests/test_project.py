"""QA test suite: TC001-TC012 from the assessment brief + negative and boundary tests.

Run alone:   python -m unittest discover tests -v      (after `python run_project.py` has produced the outputs)
Also runs as the last step of `python run_project.py` (results saved to outputs/test_results.txt).
"""
import sys
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data_pipeline as dp  # noqa: E402
import guardrails as gr  # noqa: E402
import rag  # noqa: E402
import segmentation as seg  # noqa: E402
import triage  # noqa: E402
from config import MODELS_DIR, OUTPUT_DIR, PROCESSED_DIR, RAW_DIR  # noqa: E402


class CountingLLM:
    """Stub model that records every call, to prove blocked tickets never reach it."""

    def __init__(self):
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        return triage.mock_llm(prompt)


def tickets_df(*texts):
    return pd.DataFrame({"ticket_id": [f"T{i}" for i in range(len(texts))], "text": list(texts)})


# ------------------------------------------------------------------ Part A
class PartA(unittest.TestCase):
    def test_tc001_import_customer_csv(self):
        """TC001 Import customer CSV -> loads successfully"""
        customers, orders, tickets, products = dp.load_raw(RAW_DIR)
        self.assertGreaterEqual(len(customers), 500)
        self.assertEqual(list(customers.columns), ["customer_id", "city", "signup_date", "age", "gender"])
        self.assertTrue(len(orders) > 0 and len(tickets) > 0 and len(products) > 0)

    def test_tc002_duplicate_order_records_identified(self):
        """TC002 Duplicate order records -> identified (and removed)"""
        _, orders, _, products = dp.load_raw(RAW_DIR)
        self.assertGreater(dp.audit_orders(orders)["exact_duplicate_rows"], 0)
        cleaned = dp.clean_orders(orders, products)
        self.assertEqual(int(cleaned["order_id"].duplicated().sum()), 0)

    def test_tc003_invalid_gender_standardised_to_unknown(self):
        """TC003 Invalid gender value -> Unknown"""
        df = pd.DataFrame({"customer_id": ["C1", "C2", "C3", "C4", "C5"], "city": ["pune"] * 5, "age": [30] * 5,
                           "gender": ["", "xyz", "FEMALE", "m", "  Male "]})
        self.assertEqual(dp.clean_customers(df)["gender"].tolist(), ["Unknown", "Unknown", "Female", "Male", "Male"])

    def test_tc004_missing_customer_id_handled(self):
        """TC004 Missing customer ID -> row dropped by the documented cleaning rule"""
        df = pd.DataFrame({"customer_id": ["C1", "", "C3"], "city": ["Pune"] * 3, "age": [30, 31, 32], "gender": ["M", "F", "M"]})
        self.assertEqual(dp.clean_customers(df)["customer_id"].tolist(), ["C1", "C3"])
        _, _, _, products = dp.load_raw(RAW_DIR)
        orders = pd.DataFrame({"order_id": ["O1", "O2"], "customer_id": ["", "C1"], "product_id": ["P001", "P001"], "quantity": [1, 1],
                               "order_ts": ["2025-09-01 10:00:00"] * 2, "status": ["completed"] * 2, "discount_pct": [0, 0],
                               "order_amount": [129, 129], "delivery_minutes": [20, 20]})
        self.assertEqual(dp.clean_orders(orders, products)["order_id"].tolist(), ["O2"])

    def test_city_and_age_cleaning(self):
        """Cities map to 6 canonical names; impossible ages (-4, 0, 150, 999) are replaced"""
        df = pd.DataFrame({"customer_id": list("ABCDE"), "city": [" CHENNAI", "bengaluru ", "Delhi", "mumbai", "PUNE"],
                           "age": [-4, 0, 150, 999, 30], "gender": ["M"] * 5})
        out = dp.clean_customers(df)
        self.assertEqual(sorted(out["city"]), ["Bengaluru", "Chennai", "Delhi", "Mumbai", "Pune"])
        self.assertTrue(out["age"].between(10, 100).all())

    def test_pandas_matches_sql(self):
        """SQLite validation: every Pandas-vs-SQL check passes"""
        c, o, t = (pd.read_csv(PROCESSED_DIR / f"{n}_clean.csv", keep_default_na=True) for n in ("customers", "orders", "tickets"))
        o["order_ts"] = pd.to_datetime(o["order_ts"])
        t["text"] = t["text"].fillna("")
        _, _, passed = dp.validate(c, o, t)
        self.assertTrue(passed)


# ------------------------------------------------------------------ Part B
class PartB(unittest.TestCase):
    def test_rfm_calculation(self):
        """RFM: recency/frequency/monetary come only from COMPLETED orders"""
        orders = pd.DataFrame({"order_id": ["1", "2", "3", "4"], "customer_id": ["A", "A", "A", "B"],
                               "order_ts": pd.to_datetime(["2025-09-30", "2025-09-20", "2025-09-25", "2025-09-29"]),
                               "status": ["completed", "completed", "cancelled", "completed"], "order_amount": [100.0, 50.0, 999.0, 70.0]})
        rfm = seg.build_rfm(orders, snapshot=pd.Timestamp("2025-10-01")).set_index("customer_id")
        self.assertEqual(rfm.loc["A", "frequency"], 2)
        self.assertEqual(rfm.loc["A", "monetary"], 150.0)
        self.assertEqual(rfm.loc["A", "recency"], 1)
        self.assertEqual(rfm.loc["B", "recency"], 2)

    def test_kmeans_k2_to_k8_recorded(self):
        """K-Means K=2..8: inertia + silhouette recorded, inertia falls as K grows, best K = max silhouette"""
        scores = pd.read_csv(OUTPUT_DIR / "kmeans_scores.csv")
        self.assertEqual(scores["k"].tolist(), list(range(2, 9)))
        self.assertTrue(scores["inertia"].is_monotonic_decreasing)
        seg_df = pd.read_csv(OUTPUT_DIR / "customer_segments.csv")
        self.assertEqual(seg_df["segment_id"].nunique(), int(scores.loc[scores["silhouette"].idxmax(), "k"]))

    def test_tc005_customer_with_no_recent_orders_is_at_risk(self):
        """TC005 Customer with no recent orders -> assigned to an at-risk segment"""
        import joblib
        model = joblib.load(MODELS_DIR / "kmeans_segments.joblib")
        segs = pd.read_csv(OUTPUT_DIR / "customer_segments.csv")
        cluster = model["kmeans"].predict(model["scaler"].transform(pd.DataFrame({"recency": [170], "frequency": [1], "monetary": [150.0]})))[0]
        name = segs.loc[segs["segment_id"] == cluster, "segment"].iloc[0]
        self.assertIn("At-Risk", name)

    def test_retention_list_outputs(self):
        """Retention list + model files exist and are ranked by lapse probability"""
        r = pd.read_csv(OUTPUT_DIR / "retention_list.csv")
        self.assertTrue(r["lapse_probability"].is_monotonic_decreasing)
        for f in ("kmeans_segments.joblib", "laps_model.joblib"):
            self.assertTrue((MODELS_DIR / f).exists(), f)


# ------------------------------------------------------------------ Part C
class PartC(unittest.TestCase):
    def test_tc006_empty_ticket_skips_llm(self):
        """TC006 Empty support ticket -> LLM call skipped"""
        llm = CountingLLM()
        rows, calls, _ = triage.run_guarded_pipeline(tickets_df(""), llm=llm)
        self.assertEqual((llm.calls, calls, rows[0]["reason"]), (0, 0, "empty_or_whitespace"))

    def test_tc007_whitespace_ticket_skips_llm(self):
        """TC007 Whitespace-only ticket -> LLM call skipped"""
        llm = CountingLLM()
        triage.run_guarded_pipeline(tickets_df("   ", "\n\t  "), llm=llm)
        self.assertEqual(llm.calls, 0)

    def test_tc008_oversized_ticket_blocked(self):
        """TC008 Oversized ticket -> LLM call blocked"""
        llm = CountingLLM()
        rows, _, _ = triage.run_guarded_pipeline(tickets_df("My delivery was late. " * 200), llm=llm)
        self.assertEqual((llm.calls, rows[0]["reason"]), (0, "too_long"))

    def test_token_limit_boundary(self):
        """Boundary: exactly MAX_TOKENS is allowed, one token over is blocked"""
        at_limit = "a" * (gr.MAX_TOKENS * gr.CHARS_PER_TOKEN)
        over = "a" * ((gr.MAX_TOKENS + 1) * gr.CHARS_PER_TOKEN)
        self.assertTrue(gr.validate_ticket(at_limit).allowed)
        self.assertEqual(gr.validate_ticket(over).reason, "too_long")

    def test_tc009_phone_number_masked(self):
        """TC009 Ticket containing phone number -> masked"""
        for phone in ("9562272031", "+91 9562272031", "+91-9562272031"):
            out = gr.redact_pii(f"Call me on {phone} please")
            self.assertIn("[PHONE]", out)
            self.assertFalse(any(ch.isdigit() for ch in out), out)

    def test_email_and_order_id_masked(self):
        """PII masking: email and order ID are masked, in any case"""
        out = gr.redact_pii("Order ord-100816 paid, mail user61@example.com or ORD_200000")
        self.assertEqual(out, "Order [ORDER_ID] paid, mail [EMAIL] or [ORDER_ID]")

    def test_no_pii_reaches_any_prompt(self):
        """No sensitive information leaks into zero-shot or few-shot prompts"""
        text = "ORD-100816 is late. Call 9562272031 or mail a.b@example.com"
        guard = gr.validate_ticket(text)
        for build in (gr.zero_shot_prompt, gr.few_shot_prompt):
            self.assertFalse(gr.contains_pii(build(guard.clean_text)))

    def test_none_ticket_is_blocked(self):
        """Negative: None as ticket text is treated as empty"""
        self.assertFalse(gr.validate_ticket(None).allowed)

    def test_triage_json_for_brief_example(self):
        """Triage returns the brief's JSON: Delivery Delay / Medium / Check delivery status"""
        out = triage.triage_ticket("My order is delayed. I placed it 2 hours ago, but it has not arrived.")
        self.assertEqual(out, {"category": "Delivery Delay", "priority": "Medium", "action": "Check delivery status"})

    def test_all_six_categories_reachable(self):
        """Every category of the brief can be produced"""
        self.assertEqual(triage.triage_ticket("My order is delayed")["category"], "Delivery Delay")
        samples = {"Refund Request": "I want my money back", "Payment Issue": "I was charged twice",
                   "Delivery Delay": "my order is very late", "Product Quality": "the eggs were broken",
                   "Fraud Report": "this is fraud", "App Crash": "the app keeps crashing"}
        for expected, text in samples.items():
            self.assertEqual(triage.triage_ticket(text)["category"], expected)

    def test_dataset_guardrails_and_accuracy(self):
        """Full dataset: blocked tickets never reach the model, zero PII leaks, triage accuracy >= 95%"""
        tk = pd.read_csv(PROCESSED_DIR / "tickets_clean.csv", keep_default_na=False)
        llm = CountingLLM()
        rows, calls, leaks = triage.run_guarded_pipeline(tk, llm=llm)
        df = pd.DataFrame(rows).merge(tk[["ticket_id", "true_category"]], on="ticket_id")
        self.assertEqual((leaks, llm.calls), (0, int(df["allowed"].sum())))
        ok = df[df["allowed"]]
        self.assertGreaterEqual((ok["category"] == ok["true_category"].map(triage.TRUE_LABEL_MAP)).mean(), 0.95)


# ------------------------------------------------------------------ Part D
class PartD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orders = pd.read_csv(PROCESSED_DIR / "orders_clean.csv")
        cls.rag = rag.PolicyRAG()
        cls.queue = rag.EscalationQueue(path=OUTPUT_DIR / "_test_escalations.csv", reset=True)
        cls.copilot = rag.CoPilot(cls.orders, cls.rag, cls.queue)

    @classmethod
    def tearDownClass(cls):
        (OUTPUT_DIR / "_test_escalations.csv").unlink(missing_ok=True)

    def test_chunking_strategies(self):
        """Two chunking strategies produce non-empty chunks for every policy file"""
        policies = rag.load_policies()
        for chunker in (rag.chunk_by_sentence, rag.chunk_by_window):
            chunks = chunker(policies)
            self.assertEqual({s for s, _ in chunks}, set(policies))
            self.assertTrue(all(c.strip() for _, c in chunks))

    def test_tc010_refund_policy_question_retrieves_refund_policy(self):
        """TC010 Refund policy question -> relevant policy retrieved"""
        hits = self.rag.retrieve("Who must approve a refund above INR 1000?")
        self.assertEqual(hits[0][0], "refund_policy.txt")
        self.assertIn("human support lead", " ".join(c for _, c, _ in hits))

    def test_retrieval_eval_set(self):
        """policy_eval.csv: right document retrieved for every question"""
        ev = rag.evaluate_retrieval(self.rag)
        self.assertEqual(int(ev["doc_ok"].sum()), len(ev))

    def test_tc011_unsupported_refund_requests(self):
        """TC011 Unsupported refund request -> rejected or escalated, never auto-approved"""
        cancelled = self.orders[self.orders["status"] == "cancelled"]["order_id"].iloc[0]
        out = self.copilot.respond(f"I want a refund for {cancelled}")
        self.assertEqual(out["decision"], "NO_REFUND")
        unknown = self.copilot.respond("I want a refund for ORD-999999")
        self.assertEqual((unknown["decision"], unknown["escalated"]), ("HUMAN_REVIEW", True))
        offtopic = self.copilot.respond("Do you sell gift cards for international travel?")
        self.assertEqual((offtopic["decision"], offtopic["escalated"]), ("HUMAN_REVIEW", True))

    def test_refund_rules(self):
        """Late completed order -> auto refund (<= INR 1000) or human approval (> INR 1000); on-time -> no refund"""
        done = self.orders[self.orders["status"] == "completed"]
        late_small = done[(done["delivery_minutes"] > 45) & (done["order_amount"] <= 1000)]["order_id"].iloc[0]
        late_big = done[(done["delivery_minutes"] > 45) & (done["order_amount"] > 1000)]["order_id"].iloc[0]
        on_time = done[done["delivery_minutes"] <= 45]["order_id"].iloc[0]
        agent = rag.RefundAgent(self.orders, self.rag, self.queue)
        self.assertEqual(agent.decide(f"{late_small} was very late, refund please")["decision"], "AUTO_APPROVE")
        self.assertEqual(agent.decide(f"{late_big} was very late, refund please")["decision"], "HUMAN_APPROVAL")
        self.assertEqual(agent.decide(f"{on_time} was very late, refund please")["decision"], "NO_REFUND")

    def test_damaged_product_goes_to_human_verification(self):
        """Brief's example: damaged product -> empathetic reply, human verification, case logged"""
        out = self.copilot.respond("I received a damaged product. Can I get a refund?")
        self.assertEqual((out["category"], out["decision"], out["escalated"]), ("Product Quality", "HUMAN_REVIEW", True))
        self.assertTrue(out["case_id"].startswith("ESC-"))

    def test_greeting_is_not_escalated(self):
        """Chat: 'hi' gets a greeting, no human case is opened"""
        out = self.copilot.respond("hi")
        self.assertEqual((out["decision"], out["escalated"]), ("GREETING", False))

    SMALLTALK = ["Hi", "Hello", "Hey", "HEY", "Good morning", "Good afternoon", "Good evening", "How are you?", "Thanks",
                 "Thank you", "Bye", "hello!", "  hey  ", "Hi there", "heyyy", "Hey!", "thanks a lot", "Thank you so much",
                 "Goodbye", "hi, how are you?", "Good morning!"]
    FRONTEND_FIELDS = {"reply", "blocked", "category", "priority", "action", "decision", "sources", "escalated", "case_id",
                       "redacted_text"}

    def _escalation_rows(self):
        return len(pd.read_csv(self.queue.path))

    def test_smalltalk_gets_friendly_reply_and_no_escalation(self):
        """Hi/Hello/Hey/Good morning/afternoon/evening/How are you?/Thanks/Thank you/Bye -> friendly reply, no human case"""
        for msg in self.SMALLTALK:
            before = self._escalation_rows()
            out = self.copilot.respond(msg)
            self.assertEqual((out["decision"], out["escalated"], out["case_id"], out["blocked"]), ("GREETING", False, None, False), msg)
            self.assertEqual(out["sources"], [], msg)
            self.assertTrue(out["reply"].strip(), msg)
            self.assertNotIn("No policy covers", out["reply"], msg)
            self.assertNotIn("human agent", out["reply"], msg)
            self.assertTrue(self.FRONTEND_FIELDS <= set(out), msg)        # same fields the frontend already reads
            self.assertEqual(self._escalation_rows(), before, msg)

    def test_smalltalk_common_variants(self):
        """Informal variants (hlo, good day, sup, ok thanks, thank u) are small talk too; no escalation"""
        for msg in ("hlo", "Good day", "sup", "ok thanks", "Okay, thank you!", "thank u", "Hi 👋"):
            before = self._escalation_rows()
            out = self.copilot.respond(msg)
            self.assertEqual((out["decision"], out["escalated"], out["case_id"]), ("GREETING", False, None), msg)
            self.assertEqual(self._escalation_rows(), before, msg)
        for msg in ("ok refund please", "great, my order is late", "sup with my payment?"):
            self.assertIsNone(rag.detect_smalltalk(msg), msg)

    def test_smalltalk_never_touches_policy_retrieval(self):
        """Greetings and thanks skip policy retrieval and the RAG answer entirely"""
        boom = AssertionError("policy retrieval must not run for small talk")
        with mock.patch.object(self.rag, "retrieve", side_effect=boom), mock.patch.object(self.rag, "answer", side_effect=boom):
            for msg in self.SMALLTALK:
                self.assertEqual(self.copilot.respond(msg)["decision"], "GREETING", msg)

    def test_smalltalk_reply_matches_the_intent(self):
        """Each kind of small talk gets an appropriate reply"""
        reply = lambda m: self.copilot.respond(m)["reply"]
        self.assertTrue(reply("Hello").startswith("Hello"))
        self.assertTrue(reply("Good morning").startswith("Good morning"))
        self.assertTrue(reply("Good afternoon").startswith("Good afternoon"))
        self.assertTrue(reply("Good evening").startswith("Good evening"))
        self.assertIn("doing well", reply("How are you?"))
        self.assertIn("welcome", reply("Thank you").lower())
        self.assertIn("goodbye", reply("Bye").lower())

    def test_greeting_plus_real_request_is_not_small_talk(self):
        """Small talk is only detected when the WHOLE message is small talk; a request after a greeting is still handled"""
        for msg in ("hi my order is late", "Hello, I want a refund", "thanks but I was charged twice", "hey where is my refund",
                    "Bye the way, my payment failed", "How are you handling refunds?", "Thank you for the refund on ORD-100816"):
            self.assertIsNone(rag.detect_smalltalk(msg), msg)
        out = self.copilot.respond("hi my order is late")
        self.assertEqual(out["decision"], "NEED_ORDER_ID")
        out = self.copilot.respond("Hello, what delivery time makes a completed order eligible for a full refund?")
        self.assertEqual(out["decision"], "POLICY_ANSWER")
        self.assertIn("45 minutes", out["reply"])

    def test_unrelated_question_still_escalates(self):
        """Off-topic questions are NOT small talk: still 'no policy' -> human case, exactly as before"""
        for msg in ("Do you sell gift cards for international travel?", "What is the capital of France?"):
            before = self._escalation_rows()
            out = self.copilot.respond(msg)
            self.assertEqual((out["decision"], out["escalated"]), ("HUMAN_REVIEW", True), msg)
            self.assertTrue(out["case_id"].startswith("ESC-"), msg)
            self.assertEqual(self._escalation_rows(), before + 1, msg)

    def test_real_support_requests_unchanged_after_smalltalk_check(self):
        """Refund / fraud / damaged requests behave exactly as before the small-talk check was added"""
        done = self.orders[self.orders["status"] == "completed"]
        late_small = done[(done["delivery_minutes"] > 45) & (done["order_amount"] <= 1000)]["order_id"].iloc[0]
        on_time = done[done["delivery_minutes"] <= 45]["order_id"].iloc[0]
        self.assertEqual(self.copilot.respond(f"hi, {late_small} was very late, refund please")["decision"], "AUTO_APPROVE")
        self.assertEqual(self.copilot.respond(f"{on_time} was very late, refund please")["decision"], "NO_REFUND")
        fraud = self.copilot.respond("hello, there is an unauthorized transaction on my card")
        self.assertEqual((fraud["priority"], fraud["decision"], fraud["escalated"]), ("High", "HUMAN_REVIEW", True))
        damaged = self.copilot.respond("I received a damaged product. Can I get a refund?")
        self.assertEqual((damaged["decision"], damaged["escalated"]), ("HUMAN_REVIEW", True))

    def test_vague_request_asks_for_order_id(self):
        """Chat: vague complaint without an order ID -> asks for it instead of dumping policy text"""
        for msg in ("I want to request a refund", "My order is delayed", "I have a payment issue"):
            out = self.copilot.respond(msg)
            self.assertEqual((out["decision"], out["escalated"]), ("NEED_ORDER_ID", False), msg)
            self.assertIn("order ID", out["reply"])

    def test_policy_question_still_answered_from_policy(self):
        """Chat: a real question without an order ID is still answered from the policy"""
        out = self.copilot.respond("What delivery time makes a completed order eligible for a full refund?")
        self.assertEqual(out["decision"], "POLICY_ANSWER")
        self.assertIn("45 minutes", out["reply"])

    def test_fraud_is_escalated_high_priority(self):
        """Fraud ticket -> High priority, human security review"""
        out = self.copilot.respond("There is an unauthorized transaction on my card")
        self.assertEqual((out["priority"], out["decision"], out["escalated"]), ("High", "HUMAN_REVIEW", True))

    def test_copilot_blocks_bad_input(self):
        """Empty / oversized chat messages are blocked before any retrieval or model call"""
        self.assertTrue(self.copilot.respond("   ")["blocked"])
        self.assertTrue(self.copilot.respond("late " * 1000)["blocked"])

    def test_escalation_summary_is_redacted(self):
        """Human hand-off summaries contain no PII"""
        self.copilot.respond("Fraud on ORD-100816! Call 9562272031 or mail me@example.com")
        log = pd.read_csv(OUTPUT_DIR / "_test_escalations.csv")
        self.assertFalse(log["summary"].map(gr.contains_pii).any())

    def test_rag_prompt_contains_no_pii(self):
        """RAG prompt built from a question with PII is clean"""
        ans = self.rag.answer("Refund for ORD-100816? Call 9562272031")
        self.assertFalse(gr.contains_pii(ans["prompt"]))


# ------------------------------------------------------------------ Integration
class Integration(unittest.TestCase):
    def test_tc012_complete_pipeline_outputs_exist(self):
        """TC012 Run complete pipeline -> all expected outputs generated"""
        expected = [
            PROCESSED_DIR / "customers_clean.csv", PROCESSED_DIR / "orders_clean.csv", PROCESSED_DIR / "tickets_clean.csv",
            PROCESSED_DIR / "instamart.db", OUTPUT_DIR / "eda_profile.txt", OUTPUT_DIR / "data_cleaning_report.txt",
            OUTPUT_DIR / "validation_report.txt", OUTPUT_DIR / "sql_analysis.txt", OUTPUT_DIR / "eda" / "eda_1_quality_before_after.png",
            OUTPUT_DIR / "customer_segments.csv", OUTPUT_DIR / "retention_list.csv", OUTPUT_DIR / "kmeans_scores.csv",
            OUTPUT_DIR / "cluster_visualisation.png", OUTPUT_DIR / "segment_profile.csv", MODELS_DIR / "kmeans_segments.joblib",
            MODELS_DIR / "laps_model.joblib", OUTPUT_DIR / "validated_tickets.csv", OUTPUT_DIR / "ticket_classification.csv",
            OUTPUT_DIR / "prompt_examples.txt", OUTPUT_DIR / "guardrail_results.csv", OUTPUT_DIR / "policy_eval_results.csv",
            OUTPUT_DIR / "refund_decisions.csv", OUTPUT_DIR / "escalations.csv", MODELS_DIR / "policy_vector_store.joblib",
            OUTPUT_DIR / "chat_demo.txt",
        ]
        missing = [str(p.relative_to(ROOT)) for p in expected if not p.exists()]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
