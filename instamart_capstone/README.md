# Instamart Customer Intelligence & Support Co-Pilot

## Overview
Raw Instamart-style exports are messy (duplicates, mixed spellings, impossible values), the retention team cannot see who will stop ordering, and the support desk is flooded with free-text tickets. This repo cleans the data (Part A), segments customers and predicts lapse (Part B), guards and triages tickets (Part C), and answers policy questions and decides refunds with a RAG assistant and a tool-using refund agent (Part D). Data is synthetic (`src/generate_data.py`). Everything runs offline: no network call and no API key is used anywhere. Everything is text and code (charts are not committed).

## Reproduce
```
pip install -r requirements.txt
python run_project.py      # generate data -> Parts A-D -> 49 tests; writes outputs/, models/, state/
python server.py           # optional React chatbot at http://localhost:8000 (React is loaded from a CDN)
```
Layout: `src/` code, `sql/queries.sql` Part A queries, `data/` raw + processed + policies, `outputs/` reports and CSV/JSONL results, `models/` joblib files, `state/` agent checkpoints, `tests/` unit tests.

## Part A - Data preparation and EDA
Loaded `customers.csv`, `orders.csv`, `tickets.csv`; profiled shape, dtypes, blanks and value counts with % (`outputs/eda_profile.txt`). Cleaning rules are listed in `outputs/data_cleaning_report.txt`. Cleaned tables are stored in SQLite (`data/processed/instamart.db`) and 10 of 10 Pandas-vs-SQL checks pass (`outputs/validation_report.txt`); joins are in `sql/queries.sql` (results in `outputs/sql_analysis.txt`).

| Measure | Before | After |
|---|---|---|
| Order rows | 13,084 | 12,309 |
| Exact duplicate order rows | 119 | 0 |
| Repeated order_id rows | 775 | 0 |
| City spellings | 24 | 6 |
| Gender spellings | 11 | 3 (Male/Female/Unknown) |
| Ages outside 10-100 or blank | 67 | 0 |
| Non-numeric order_amount | 120 | 0 |
| Negative delivery_minutes | 106 | 0 |
| Pending rows | 656 | 0 |

Outliers (IQR): order_amount 1,034 (8.0%), delivery_minutes 976 (9.3%). High amounts are real (electronics x quantity) and kept; only impossible values were corrected.

## Part B - Segmentation and churn
RFM from completed orders -> `StandardScaler` -> K-Means for K=2..8 (`outputs/kmeans_scores.csv`). Best K by silhouette = **4** (0.526; K=2: 0.465, K=3: 0.445, K=5: 0.480, K=6: 0.482, K=7: 0.416, K=8: 0.412).

| Segment | Customers | Recency (days) | Frequency | Monetary (INR) |
|---|---|---|---|---|
| Loyal / Active | 419 | 8.8 | 19.4 | 3,410 |
| Low-Engagement / Occasional | 436 | 28.9 | 3.8 | 1,058 |
| At-Risk / Lapsing | 182 | 109.0 | 2.2 | 686 |
| Champions / Premium | 138 | 24.1 | 6.8 | 7,130 |

Lapse model (no completed order in September; features from orders before it): logistic regression accuracy 0.728, precision 0.644, recall 0.652, F1 0.648 (majority baseline 0.616); random forest F1 0.641. Outputs: `customer_segments.csv`, `retention_list.csv` (top 100), `models/laps_model.joblib`.

## Part C - Guardrails and triage
Rules (top of `src/guardrails.py`): R1 empty/whitespace -> no model call; R2 over 200 approx. tokens -> no model call; R3 email/phone/order ID masked before any prompt; R4 leak check blocks a prompt that still holds PII. Zero-shot and few-shot prompts are in `outputs/prompt_examples.txt`; triage returns JSON `{category, priority, action}`.
Result on 164 tickets: 8 blocked (4 empty, 4 too long), 156 model calls, **0** PII leaks, triage accuracy 100% for both prompt styles (offline keyword classifier, so both give the same labels).

## Part D - RAG assistant and refund agent
**RAG.** Policies are chunked and embedded (TF-IDF + LSA), stored in a vector store and retrieved with a similarity floor; below it the case goes to a human. Retrieval picks the right policy for 14/14 evaluation questions.

| Chunking strategy | Chunks | Right doc | Fact found |
|---|---|---|---|
| sentence, size 2, overlap 1 (default) | 29 | 14/14 | 14/14 |
| sentence, size 2, overlap 0 | 19 | 14/14 | 14/14 |
| window 300 chars, overlap 60 | 13 | 14/14 | 14/14 |
| window 300 chars, **overlap 0** | 11 | 14/14 | **12/14** |

The zero-overlap run fails 2 questions ("On orders of what value is delivery free?", "After how many steps must an automated assistant hand over to a human?") because the answer is cut across a chunk boundary (`outputs/chunking_comparison.csv`).

**Refund agent** (`src/agent.py`). Loop: prompt (PII-checked) -> offline planner picks a tool -> run tool -> checkpoint. Tools: `lookup_order`, `check_policy`, `compute_refund`, `issue_refund`, `escalate_to_human`. Rules: only completed orders; delivery > 45 min -> full refund; above INR 1000 a human approves; fraud, unknown orders and damaged items go to a human. Defaults: `token_limit=6000`, 8 LLM calls, 30 s simulated timeout.

| Scenario | Situation | Status | Refund (INR) | LLM calls | Tokens |
|---|---|---|---|---|---|
| S1 | Late, amount <= 1000 (ORD-100048) | REFUND_ISSUED | 128.25 | 5 | 868 |
| S2 | Late, amount > 1000 (ORD-100145, INR 1999) | ESCALATED | 0.00 | 5 | 895 |
| S3 | Delivered on time | NO_REFUND | 0.00 | 4 | 645 |
| S4 | Cancelled order | NO_REFUND | 0.00 | 4 | 652 |
| S5 | Fraud report | ESCALATED | 0.00 | 2 | 266 |
| S6 | Unknown order ID | ESCALATED | 0.00 | 3 | 377 |
| S7 | 12 invalid order IDs, agent keeps retrying | FALLBACK_MAX_ITERATIONS | 0.00 | 8 | 1,465 |

S7 stops at the 8-call cap with 1,465 tokens, well under `token_limit=6000`, and is handed to a human.

| Demonstration | Setting | Status |
|---|---|---|
| Token limit | `token_limit=400` on S1 | FALLBACK_TOKEN_LIMIT |
| Timeout | 20 s budget, `check_policy` takes 25 s | FALLBACK_TIMEOUT |
| Crash and resume | crash after 2 tools, then resume S1 | REFUND_ISSUED, INR 128.25 |

Crash-and-resume tool executions: before the crash `lookup_order`, `check_policy`; after resume `compute_refund`, `issue_refund` -> **exactly 4** executions, and the refund is issued once.

Files: `outputs/agent_runs.jsonl` (one line per run: scenario_id, status, final_answer, llm_calls, tokens_used, steps), `outputs/refund_ledger.jsonl` (ISSUED and ESCALATED actions), `state/*.json` (checkpoints), `outputs/agent_scenarios.csv`.

**Ticket decisions** (`outputs/refund_decisions.csv`, 164 tickets): 78 HUMAN_REVIEW (logged in `outputs/escalations.csv`), 74 NO_REFUND, 8 REJECT_INPUT, 4 AUTO_APPROVE. **Chat**: greetings/thanks/bye get a friendly reply with no escalation; the React page calls `server.py`. **Tests**: 49/49 pass (`outputs/test_results.txt`).

## Limitations
- The LLM is an offline stand-in (rule-based planner, keyword classifier, extractive answers), so zero-shot and few-shot give identical labels and the 100% triage accuracy is not a measure of a real model.
- Tool latency and the timeout use a simulated clock.
- Embeddings are TF-IDF + LSA, not a neural model.
- The refund policy has no damaged-item rule, so those cases go to a human.
- Data is synthetic; the lapse model has modest accuracy (0.73).

## Next Steps
- Plug a real LLM into `triage.mock_llm`, `rag.default_llm` and `agent.plan`, keeping the guardrails and limits.
- Use a neural embedding model and a persistent vector database.
- Add a damaged-item rule and photo evidence to the refund policy.
- Retrain the lapse model with more features and monitor drift.
