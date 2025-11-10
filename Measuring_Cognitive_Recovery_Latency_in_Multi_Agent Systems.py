import os, time, random, json, pathlib
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

from langgraph.graph import StateGraph, START, END
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

# === VISUAL STYLE SETTINGS ===
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette(["#4169E1", "#9370DB", "#000000", "#FFFFFF"])  # royal blue, purple, black, white
sns.set_context("talk", font_scale=0.9)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "#000000"
plt.rcParams["grid.color"] = "#D3D3D3"
plt.rcParams["grid.alpha"] = 0.6
plt.rcParams["axes.labelcolor"] = "#000000"
plt.rcParams["xtick.color"] = "#000000"
plt.rcParams["ytick.color"] = "#000000"

random.seed(42)

# ===================== EXPERIMENT SETTINGS =====================
N_RUNS = 200
DRIFT_CONF_THRESHOLD = 0.6    # drift sensitivity; 
REFLEX_WEIGHTS = [
    ("auto-replan", 0.45),
    ("rollback", 0.25),
    ("tool-retry", 0.20),
    ("human-approve", 0.10)
]

# === Query pool ===
QUERY_POOL = [
  "LangGraph recovery reflexes", "agent orchestration reliability",
  "rollback sandbox audit snapshots", "tool retries and backoff",
  "consensus voting disagreement", "policy thresholds and approvals",
  "governance and risk tiers", "observability telemetry signals",
  "drift detection and confidence", "mttra and mtbf calculation",
  "normalized reliability index", "incident playbooks escalation",
  "global memory reconciliation", "safe mode fallback routes",
  "retrieval reranking grounding", "planning decomposition tools"
]

# ===================== DATASET LOADING =====================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

USE_HF = True # set True to use Hugging Face 'ag_news' instead of 20 Newsgroups

def load_corpus_20newsgroups(subset="train", remove=("headers","footers","quotes")):
    """
    Returns: texts (list[str]), titles (list[str]) where 'titles' are just category labels for display.
    """
    from sklearn.datasets import fetch_20newsgroups
    data = fetch_20newsgroups(subset=subset, remove=remove)
    texts = [t for t in data.data]
    titles = [data.target_names[y] for y in data.target]
    return texts, titles

def load_corpus_hf_agnews(split="train"):
    """
    Returns: texts (list[str]), titles (list[str]) from Hugging Face 'ag_news' (4 news categories).
    """
    from datasets import load_dataset
    ds = load_dataset("ag_news", split=split)
    texts = [x["text"] for x in ds]
    titles = [x["label"] for x in ds]  # numeric class IDs (0..3)
    return texts, titles

print("Loading real dataset...")
if USE_HF:
    texts, titles = load_corpus_hf_agnews(split="train")
    dataset_name = "ag_news (HF)"
else:
    texts, titles = load_corpus_20newsgroups(subset="train")
    dataset_name = "20newsgroups (scikit-learn)"

print(f"Dataset: {dataset_name} | Documents: {len(texts):,}")

# light cleanup to reduce empty docs
def _normalize(s: str) -> str:
    return (s or "").replace("\r"," ").replace("\n"," ").strip()

texts = [ _normalize(t) for t in texts ]
mask = [ (len(t) > 0) for t in texts ]
texts = [ t for t,m in zip(texts, mask) if m ]
titles = [ ti for ti,m in zip(titles, mask) if m ]

print(f"After cleanup: {len(texts):,} docs")

# ===================== BUILD TF-IDF RETRIEVER =====================
# Fit once globally; reuse for local_search()
tfidf = TfidfVectorizer(max_df=0.9, min_df=2, ngram_range=(1,2))
X = tfidf.fit_transform(texts)  # shape: [N_docs, vocab]

def local_search(query: str, k: int = 3):
    """
    Real retrieval: cosine similarity on TF-IDF between query and corpus.
    Returns list of top-k document texts.
    """
    qv = tfidf.transform([query])
    sims = cosine_similarity(qv, X)[0]  # vector of length N_docs
    if np.all(sims == 0):
        # Fallback: return 3 random docs to avoid degenerate empty results
        idx = np.random.choice(len(texts), size=min(k, len(texts)), replace=False)
    else:
        idx = np.argsort(-sims)[:k]
    return [texts[i] for i in idx], sims[idx].tolist()

def confidence_from_scores(scores):
    """
    Confidence is mean of top-k cosine similarities (already in [0,1]).
    """
    if not scores: return 0.0
    c = float(np.clip(np.mean(scores), 0.0, 1.0))
    return c

# ===================== TELEMETRY =====================
LOGDIR = pathlib.Path("telemetry"); LOGDIR.mkdir(exist_ok=True)
LOGFILE = LOGDIR / f"langgraph_run_{int(time.time())}.jsonl"

def jlog(**kv):
    kv["ts"] = time.time()
    with open(LOGFILE, "a") as f: f.write(json.dumps(kv) + "\n")

# ===================== LANGGRAPH NODES =====================
def reasoning_node(state):
    state["t_reason_start"] = time.time()
    query = state["query"]
    snippets, scores = local_search(query, k=3)
    state["snippets"] = snippets
    state["retrieval_scores"] = scores
    state["confidence"] = confidence_from_scores(scores)
    state["t_reason_end"] = time.time()
    jlog(run=state["run_id"], event="reasoning_done",
         conf=state["confidence"], dataset=dataset_name,
         top_scores=scores)
    return state

def check_drift_node(state):
    state["t_drift_check"] = time.time()
    # Drift if confidence too low OR random rare false-negative/edge case
    state["is_drift"] = (state["confidence"] < DRIFT_CONF_THRESHOLD) or (random.random() < 0.05)
    jlog(run=state["run_id"],
         event="fault_detected" if state["is_drift"] else "no_fault",
         conf=state["confidence"])
    return state

def recovery_node(state):
    state["t_recovery_start"] = time.time()
    if not state["is_drift"]:
        state.update({"recovery_mode": "no-drift", "recovery_delay": 0})
        jlog(run=state["run_id"], event="recovered", mode="no-drift")
        return state

    # Decision latency
    t_decide_start = time.time()
    modes, weights = zip(*REFLEX_WEIGHTS)
    mode = random.choices(modes, weights=weights, k=1)[0]
    time.sleep(random.uniform(0.2, 0.6))
    t_decide_end = time.time()

    # Execute reflex
    t_exec_start = time.time()
    if mode == "tool-retry":
        time.sleep(random.uniform(3.0, 5.0))
        _snip, _ = local_search(state["query"] + " recovery", k=3)
    elif mode == "auto-replan":
        time.sleep(random.uniform(4.0, 6.5))
        _snip, _ = local_search(state["query"] + " orchestration", k=3)
    elif mode == "rollback":
        time.sleep(random.uniform(5.5, 7.0))
        state["snippets"] = []
        _snip, _ = local_search(state["query"], k=3)
    elif mode == "human-approve":
        time.sleep(random.uniform(10.0, 12.5))
    t_exec_end = time.time()

    # Record metrics
    state["recovery_mode"] = mode
    state["T_detect"] = state["t_drift_check"] - state["t_reason_end"]
    state["T_decide"] = t_decide_end - t_decide_start
    state["T_execute"] = t_exec_end - t_exec_start
    state["recovery_delay"] = state["T_decide"] + state["T_execute"]

    jlog(run=state["run_id"], event="reflex_selected", mode=mode)
    jlog(run=state["run_id"], event="recovered", mode=mode,
         T_detect=state["T_detect"], T_decide=state["T_decide"],
         T_execute=state["T_execute"])
    return state

# ===================== GRAPH BUILD =====================
graph = StateGraph(dict)
graph.add_node("reasoning", reasoning_node)
graph.add_node("check_drift", check_drift_node)
graph.add_node("recovery", recovery_node)
graph.add_edge(START, "reasoning")
graph.add_edge("reasoning", "check_drift")
graph.add_edge("check_drift", "recovery")
graph.add_edge("recovery", END)
app = graph.compile()

# ===================== RUN EXPERIMENT =====================
records = []
print("Running experiment...")
for i in range(N_RUNS):
    init = {"run_id": i, "query": random.choice(QUERY_POOL)}
    final = app.invoke(init)
    records.append({
        "run_id": i,
        "is_drift": final["is_drift"],
        "recovery_mode": final["recovery_mode"],
        "delay_sec": final.get("recovery_delay", 0),
        "T_detect": final.get("T_detect", 0),
        "T_decide": final.get("T_decide", 0),
        "T_execute": final.get("T_execute", 0),
        "t_recovered": final.get("t_recovery_start", 0),
        "conf": final.get("confidence", np.nan)
    })
    if i % 20 == 0:
        print(f" Completed {i}/{N_RUNS} runs...")

df = pd.DataFrame(records)
print("\n Finished all runs.")

# ===================== METRICS =====================
valid = df[df.delay_sec > 0]["delay_sec"]
if len(valid) == 0:
    print("No drift/recovery events detected —> try lowering DRIFT_CONF_THRESHOLD to 0.45–0.55.")
else:
    mttr_a = valid.median(); mttr_std = valid.std()
    p90 = np.percentile(valid, 90)
    drift_rate = df.is_drift.mean()

    # MTBF based on recovery timestamps of drifted runs (rough heuristic)
    recov_times = df.loc[df.is_drift, "t_recovered"].sort_values().values
    if len(recov_times) >= 2:
        mtbf_intervals = np.diff(recov_times)
        mtbf = float(np.mean(mtbf_intervals))
        mtbf_std = float(np.std(mtbf_intervals))
    else:
        # fall back to total window / num_drifts if too few points
        total_time = df["t_recovered"].max() - df["t_recovered"].min()
        num_drifts = int(df["is_drift"].sum())
        mtbf = total_time / num_drifts if num_drifts > 0 else float("inf")
        mtbf_std = 0.0

    nrr = 1 - (mttr_a / max(mtbf, 1e-6))

    print(f"\nDataset: {dataset_name}")
    print(f"Median MTTR-A: {mttr_a:.2f} ± {mttr_std:.2f}s | P90: {p90:.2f}s | Drift rate: {drift_rate:.1%}")
    print(f"MTBF ≈ {mtbf:.2f} ± {mtbf_std:.2f}s | NRR ≈ {nrr:.3f}")

    summary = (
        df[df.delay_sec > 0]
        .groupby("recovery_mode")
        .agg(Median_MTTR_A=("delay_sec", "median"),
             Std_MTTR_A=("delay_sec", "std"),
             P90_MTTR_A=("delay_sec", lambda x: np.percentile(x, 90)),
             Count=("run_id", "count"))
        .reset_index()
    )
    print("\nSummary per recovery mode:\n", summary.round(2))
