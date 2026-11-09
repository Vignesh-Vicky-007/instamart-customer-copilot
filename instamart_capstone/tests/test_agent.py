"""Part D agent tests: seven scenarios, limits, crash-and-resume, no network / API key."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

import agent  # noqa: E402
import rag  # noqa: E402
from config import PROCESSED_DIR  # noqa: E402


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orders = pd.read_csv(PROCESSED_DIR / "orders_clean.csv")
        cls.rag = rag.PolicyRAG()
        cls.tmp = Path(tempfile.mkdtemp())
        cls.ledger = agent.Ledger(cls.tmp / "ledger.jsonl")
        cls.ledger.reset()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def make(self, **kw):
        box = agent.ToolBox(self.orders, self.rag, self.ledger)
        return box, agent.RefundAgent(box, self.ledger, state_dir=self.tmp / "state", **kw)

    def test_seven_scenarios_match_expected_outcome_and_amount(self):
        for sc in agent.build_scenarios(self.orders):
            _, ag = self.make()
            res = ag.run(sc["id"], sc["message"])
            self.assertEqual((res["status"], res["refund_amount"]), sc["expected"], sc["id"])

    def test_s7_hits_iteration_cap_under_default_token_limit(self):
        s7 = agent.build_scenarios(self.orders)[6]
        _, ag = self.make()
        res = ag.run("S7", s7["message"])
        self.assertEqual(res["status"], "FALLBACK_MAX_ITERATIONS")
        self.assertEqual(res["llm_calls"], agent.MAX_ITERATIONS)
        self.assertLess(res["tokens_used"], agent.TOKEN_LIMIT)

    def test_token_limit_and_timeout_fallbacks(self):
        msg = agent.build_scenarios(self.orders)[0]["message"]
        self.assertEqual(self.make(token_limit=400)[1].run("T1", msg)["status"], "FALLBACK_TOKEN_LIMIT")
        self.assertEqual(self.make(timeout_s=20.0, latency={"check_policy": 25.0})[1].run("T2", msg)["status"], "FALLBACK_TIMEOUT")

    def test_crash_and_resume_runs_exactly_four_tools(self):
        s1 = agent.build_scenarios(self.orders)[0]
        box1, ag1 = self.make()
        with self.assertRaises(agent.CrashError):
            ag1.run("CR", s1["message"], crash_after=2)
        box2, ag2 = self.make()
        res = ag2.run("CR", s1["message"], resume=True)
        self.assertEqual(box1.executions + box2.executions, ["lookup_order", "check_policy", "compute_refund", "issue_refund"])
        self.assertEqual((res["status"], res["refund_amount"]), s1["expected"])

    def test_refund_never_issued_twice_after_resume(self):
        issued = [json.loads(line) for line in open(self.ledger.path) if '"CR"' in line and "ISSUED" in line]
        self.assertLessEqual(len(issued), 1)

    def test_empty_request_rejected_without_tools(self):
        box, ag = self.make()
        res = ag.run("E", "   ")
        self.assertEqual(box.executions, [])
        self.assertEqual(res["llm_calls"], 0)

    def test_zero_overlap_run_fails_at_least_one_question(self):
        pr = rag.PolicyRAG(chunker=lambda p: rag.chunk_by_window(p, 300, 0))
        ev = rag.evaluate_retrieval(pr)
        self.assertLess(int(ev["fact_ok"].sum()), len(ev))


if __name__ == "__main__":
    unittest.main()
