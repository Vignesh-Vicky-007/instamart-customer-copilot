"""Single entry point: python run_project.py  (runs Parts A -> D end to end)."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402  (after sys.path setup)

import data_pipeline  # noqa: E402
import agent as agent_mod  # noqa: E402
import guardrails  # noqa: E402
import rag  # noqa: E402
import segmentation  # noqa: E402
import triage  # noqa: E402
from config import MODELS_DIR, OUTPUT_DIR, PROCESSED_DIR, RAW_DIR  # noqa: E402

SAMPLE_QUESTIONS = [
    "What delivery time makes a completed order eligible for a full refund?",
    "Who must approve a refund above INR 1000?",
    "How fast must a fraud ticket reach the security team?",
]


def part_a():
    print("\n===== PART A: Data preparation & EDA =====")
    runpy.run_path(str(ROOT / "src" / "generate_data.py"), run_name="__main__")   # the provided generator
    customers, orders, tickets, products, profiles, audit, before, after = data_pipeline.clean_all(RAW_DIR)
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "eda_profile.txt").write_text(profiles + "\n\n" + data_pipeline.CLEANING_RULES)
    raw_c, raw_o, raw_t, _ = data_pipeline.load_raw(RAW_DIR)
    (OUTPUT_DIR / "data_cleaning_report.txt").write_text(data_pipeline.cleaning_report((raw_c, raw_o, raw_t), (customers, orders, tickets)))
    data_pipeline.eda_charts((raw_c, raw_o, raw_t), (customers, orders, tickets), products, OUTPUT_DIR / "eda")
    print("raw audit:", audit, "\nrows before:", before, "\nrows after: ", after)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data_pipeline.to_sqlite(customers, orders, tickets, products)
    customers.to_csv(PROCESSED_DIR / "customers_clean.csv", index=False)
    orders.to_csv(PROCESSED_DIR / "orders_clean.csv", index=False)
    tickets.to_csv(PROCESSED_DIR / "tickets_clean.csv", index=False)

    report, city_table, all_passed = data_pipeline.validate(customers, orders, tickets)
    print(report, "\n", city_table.to_string(index=False))
    (OUTPUT_DIR / "validation_report.txt").write_text(report)
    sql_report = data_pipeline.sql_analysis()
    (OUTPUT_DIR / "sql_analysis.txt").write_text(sql_report)
    print(sql_report)
    assert all_passed, "Pandas vs SQL validation failed"
    assert audit["exact_duplicate_rows"] == 119 and audit["raw_city_spellings"] == 24 and audit["pending_rows"] == 656, \
        "raw data does not match the teacher's checkpoints (119 duplicates, 24 city spellings, 656 pending)"
    return customers, orders, tickets, products


def part_b(customers, orders, products):
    print("\n===== PART B: Segmentation & churn =====")
    rfm = segmentation.build_rfm(orders)
    segmented, scores, best_k, profile = segmentation.segment(rfm)
    print(scores.round(3).to_string(index=False), f"\nOptimal K (max silhouette) = {best_k}\n", profile)

    metrics, best, model, data, baseline = segmentation.churn_model(orders, customers, products)
    print("\nLapse model comparison:\n", metrics.drop(columns="best_params").to_string(index=False),
          f"\nSelected: {best} (majority baseline accuracy {baseline:.3f})")
    retention = segmentation.retention_list(segmented, data, model)
    print(f"Retention list: {len(retention)} customers -> outputs/retention_list.csv")


def part_c(tickets):
    print("\n===== PART C: Guardrails & ticket triage =====")
    rows, calls, leaks = triage.run_guarded_pipeline(tickets, mode="few_shot")
    results = pd.DataFrame(rows)
    zero_rows, zero_calls, zero_leaks = triage.run_guarded_pipeline(tickets, mode="zero_shot")
    zero = pd.DataFrame(zero_rows)

    # validated ticket data (one row per ticket: decision + redacted text that is safe to prompt with)
    results[["ticket_id", "allowed", "reason", "tokens_approx", "redacted_text"]].to_csv(OUTPUT_DIR / "validated_tickets.csv", index=False)
    # classification results for both prompt styles, with the accuracy check against the known labels
    truth = tickets[["ticket_id", "true_category"]].assign(expected=lambda d: d["true_category"].map(triage.TRUE_LABEL_MAP))
    both = pd.concat([results, zero]).merge(truth, on="ticket_id")
    both = both[both["allowed"]][["ticket_id", "mode", "category", "priority", "action", "expected"]]
    both["correct"] = both["category"] == both["expected"]
    both.to_csv(OUTPUT_DIR / "ticket_classification.csv", index=False)
    (OUTPUT_DIR / "prompt_examples.txt").write_text(triage.prompt_examples())
    results.drop(columns=["predicted"]).to_csv(OUTPUT_DIR / "guardrail_results.csv", index=False)

    print(results["reason"].value_counts().to_string())
    blocked = (~results["allowed"]).sum()
    print(f"Model calls: {calls} | blocked: {blocked} | PII leaks into prompts: {leaks}")
    assert leaks == 0 and zero_leaks == 0 and calls == results["allowed"].sum() == zero_calls
    assert not results[results["reason"].isin(["empty_or_whitespace", "too_long"])]["allowed"].any()
    for mode, g in both.groupby("mode"):
        print(f"{mode} triage accuracy on allowed tickets: {g['correct'].mean():.3f}")
    print("Priority mix:", both[both['mode'] == 'few_shot']["priority"].value_counts().to_dict())
    print(open(OUTPUT_DIR / "prompt_examples.txt").read())


CHAT_DEMO = [
    "I received a damaged product. Can I get a refund?",
    "What delivery time makes a completed order eligible for a full refund?",
    "{late} was delivered very late, I want a refund.",
    "There is an unauthorized transaction on my card. I did not place it.",
    "Do you sell gift cards for international travel?",
    "   ",
]


def part_d(orders, tickets):
    print("\n===== PART D: RAG + refund agent =====")
    policies = rag.load_policies()
    print(f"Policy files: {list(policies)}")
    print("Chunks (sentence strategy):", len(rag.chunk_by_sentence(policies)),
          "| window strategy:", len(rag.chunk_by_window(policies)))
    policy_rag = rag.PolicyRAG()
    policy_rag.store.save(MODELS_DIR / "policy_vector_store.joblib")
    print(f"Vector store: {len(policy_rag.chunks)} chunks x {policy_rag.store.dense.shape[1]}-dim dense + sparse TF-IDF embeddings")
    for question in SAMPLE_QUESTIONS:
        print(f"Q: {question}\n  -> {policy_rag.answer(question)['answer'][:110]}...")
    print("Example grounded LLM prompt:\n" + policy_rag.answer(SAMPLE_QUESTIONS[0])["prompt"])
    ev = rag.evaluate_retrieval(policy_rag)
    ev.to_csv(OUTPUT_DIR / "policy_eval_results.csv", index=False)
    print(f"Policy retrieval eval: right doc {ev['doc_ok'].sum()}/{len(ev)}, fact found {ev['fact_ok'].sum()}/{len(ev)}")

    queue = rag.EscalationQueue(reset=True)                      # fresh outputs/escalations.csv for this run
    agent = rag.RefundAgent(orders, policy_rag, queue)
    decisions = pd.DataFrame([{"ticket_id": r["ticket_id"], **agent.decide(r["text"], order_id=r["order_id"] or None, ticket_id=r["ticket_id"])}
                              for _, r in tickets.iterrows()])
    decisions.to_csv(OUTPUT_DIR / "refund_decisions.csv", index=False)
    print(decisions["decision"].value_counts().to_string())
    print(f"Escalated to human agents: {decisions['case_id'].notna().sum()} cases -> outputs/escalations.csv")

    # chat demo (what the React page shows), including the brief's damaged-product example
    late = orders[(orders["status"] == "completed") & (orders["delivery_minutes"].astype(float) > 45) & (orders["order_amount"] <= 1000)]["order_id"].iloc[0]
    copilot = rag.CoPilot(orders, policy_rag, queue)
    lines = []
    for msg in CHAT_DEMO:
        out = copilot.respond(msg.format(late=late))
        lines.append(f"CUSTOMER: {msg.format(late=late)!r}\nCO-PILOT: {out['reply']}\n  [category={out.get('category')} decision={out.get('decision')} "
                     f"escalated={out['escalated']} case={out.get('case_id')} sources={out.get('sources')}]\n")
    (OUTPUT_DIR / "chat_demo.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def chunking_comparison(policies):
    """Chunk counts and retrieval quality for each strategy, including a ZERO-overlap run (where facts split across chunks)."""
    rows = []
    for name, chunker in [("sentence size=2 overlap=1 (default)", lambda p: rag.chunk_by_sentence(p, 2, 1)),
                          ("sentence size=2 overlap=0", lambda p: rag.chunk_by_sentence(p, 2, 0)),
                          ("window 300 chars overlap=60", lambda p: rag.chunk_by_window(p, 300, 60)),
                          ("window 300 chars overlap=0 (zero-overlap run)", lambda p: rag.chunk_by_window(p, 300, 0))]:
        pr = rag.PolicyRAG(policies, chunker=chunker)
        ev = rag.evaluate_retrieval(pr)
        rows.append({"strategy": name, "chunks": len(pr.chunks), "right_doc": int(ev["doc_ok"].sum()), "fact_found": int(ev["fact_ok"].sum()),
                     "questions": len(ev), "failed_questions": " | ".join(ev[~(ev["doc_ok"] & ev["fact_ok"])]["question"])})
    return pd.DataFrame(rows)


def part_d_agent(orders, policy_rag):
    """Seven refund scenarios + token-limit, timeout and crash-and-resume demonstrations. Writes agent_runs.jsonl / refund_ledger.jsonl / state/."""
    import shutil
    print("\n===== PART D: refund agent scenarios =====")
    shutil.rmtree(agent_mod.STATE_DIR, ignore_errors=True)
    ledger = agent_mod.Ledger()
    ledger.reset()
    scenarios = agent_mod.build_scenarios(orders)
    results, table = [], []
    for sc in scenarios:
        res = agent_mod.RefundAgent(agent_mod.ToolBox(orders, policy_rag, ledger), ledger).run(sc["id"], sc["message"])
        ok = (res["status"], res["refund_amount"]) == sc["expected"]
        results.append(res)
        table.append({"scenario": sc["id"], "title": sc["title"], "expected_status": sc["expected"][0], "expected_amount": sc["expected"][1],
                      "status": res["status"], "refund_amount": res["refund_amount"], "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"], "match": ok})
        assert ok, f"{sc['id']}: got {res['status']} {res['refund_amount']}, expected {sc['expected']}"
    pd.DataFrame(table).to_csv(OUTPUT_DIR / "agent_scenarios.csv", index=False)
    print(pd.DataFrame(table).drop(columns="title").to_string(index=False))

    s1 = scenarios[0]
    tok = agent_mod.RefundAgent(agent_mod.ToolBox(orders, policy_rag, ledger), ledger, token_limit=400).run("DEMO_TOKEN_LIMIT", s1["message"])
    tmo = agent_mod.RefundAgent(agent_mod.ToolBox(orders, policy_rag, ledger), ledger, timeout_s=20.0,
                                latency={"check_policy": 25.0}).run("DEMO_TIMEOUT", s1["message"])
    assert tok["status"] == "FALLBACK_TOKEN_LIMIT" and tmo["status"] == "FALLBACK_TIMEOUT"
    print(f"Token-limit demo (token_limit=400): {tok['status']} | Timeout demo (timeout 20s, slow check_policy): {tmo['status']}")

    box1 = agent_mod.ToolBox(orders, policy_rag, ledger)
    try:
        agent_mod.RefundAgent(box1, ledger).run("DEMO_CRASH_RESUME", s1["message"], crash_after=2)
        raise AssertionError("crash was not triggered")
    except agent_mod.CrashError as exc:
        print(f"Crash: {exc}; tools run before crash: {box1.executions}")
    box2 = agent_mod.ToolBox(orders, policy_rag, ledger)
    resumed = agent_mod.RefundAgent(box2, ledger).run("DEMO_CRASH_RESUME", s1["message"], resume=True)
    all_tools = box1.executions + box2.executions
    assert len(all_tools) == 4 and resumed["status"] == "REFUND_ISSUED" and resumed["refund_amount"] == s1["expected"][1]
    print(f"Resume: tools run after resume: {box2.executions}; total = {len(all_tools)} -> {all_tools}; result {resumed['status']} INR {resumed['refund_amount']}")
    agent_mod.write_runs(results + [tok, tmo, resumed])
    print("ledger lines:", sum(1 for _ in open(ledger.path)))

    comp = chunking_comparison(rag.load_policies())
    comp.to_csv(OUTPUT_DIR / "chunking_comparison.csv", index=False)
    print(comp.drop(columns="failed_questions").to_string(index=False))
    assert comp.iloc[-1]["failed_questions"], "the zero-overlap run should fail at least one question"


def part_tests():
    """QA: run tests/test_project.py (TC001-TC012 + negative/boundary tests) and save the results."""
    import unittest
    print("\n===== QA: test cases =====")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    with open(OUTPUT_DIR / "test_results.txt", "w") as fh:
        result = unittest.TextTestRunner(stream=fh, verbosity=2).run(suite)
    print((OUTPUT_DIR / "test_results.txt").read_text())
    assert result.wasSuccessful(), "test failures - see outputs/test_results.txt"


def main():
    customers, orders, tickets, products = part_a()
    part_b(customers, orders, products)
    part_c(tickets)
    part_d(orders, tickets)
    part_d_agent(orders, rag.PolicyRAG())
    part_tests()
    print("\nAll parts completed. See outputs/ and models/.")


if __name__ == "__main__":
    main()
