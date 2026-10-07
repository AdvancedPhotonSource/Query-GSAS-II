"""
A/B eval: Snowflake Arctic Embed and Jina Embeddings v2 vs bge-base baseline,
on the same 40-question set used in the manuscript.
Uses direct numpy cosine search against the real production index.
Run with: /Users/b324240/miniconda3/bin/python evaluation/eval_more_candidates.py
"""
import json
import time
from pathlib import Path

import numpy as np

PROD_DB = "/Users/b324240/.GSASII/query_gsas2/chroma_db"
COLLECTION = "gsasii_docs"
QUESTIONS_PATH = Path(__file__).parent / "questions.json"
FETCH_K = 30
TOP_K = 6

MODELS = [
    ("jinaai/jina-embeddings-v2-base-en",  "jina-embed-v2-en (768-dim, DE/Jina AI) "),
]


def load_questions():
    with open(QUESTIONS_PATH) as f:
        return json.load(f)["questions"]


def hit(urls, gt_urls, k):
    for url in urls[:k]:
        for gt in gt_urls:
            if (url.rstrip("/") == gt.rstrip("/")
                    or url.startswith(gt.rstrip("/"))
                    or gt.startswith(url.rstrip("/"))):
                return True
    return False


def reciprocal_rank(urls, gt_urls):
    for i, url in enumerate(urls, 1):
        for gt in gt_urls:
            if (url.rstrip("/") == gt.rstrip("/")
                    or url.startswith(gt.rstrip("/"))
                    or gt.startswith(url.rstrip("/"))):
                return 1.0 / i
    return 0.0


def url_diverse_topk(sims, urls, fetch_k=FETCH_K, top_k=TOP_K):
    order = np.argsort(sims)[::-1][:fetch_k]
    seen = {}
    for idx in order:
        url = urls[idx]
        if url not in seen:
            seen[url] = sims[idx]
    ranked = sorted(seen.items(), key=lambda x: x[1], reverse=True)
    return [u for u, _ in ranked[:top_k]]


def run_model(model_name, label, doc_texts, doc_urls, questions):
    from fastembed import TextEmbedding

    print(f"\n{'='*60}")
    print(f"Model: {label}")
    print(f"{'='*60}")

    fe = TextEmbedding(model_name)

    n = len(doc_texts)
    t0 = time.time()
    batch = 256
    all_embs = []
    for i in range(0, n, batch):
        batch_embs = list(fe.embed(doc_texts[i:i + batch]))
        all_embs.extend(batch_embs)
        print(f"  Embedded {min(i+batch, n)}/{n} chunks...", end="\r")
    doc_matrix = np.array(all_embs, dtype=np.float32)
    print(f"  Embedding done in {time.time()-t0:.0f}s  shape={doc_matrix.shape}")

    cats = {}
    for q in questions:
        q_emb = np.array(next(fe.query_embed([q["question"]])), dtype=np.float32)
        sims = doc_matrix @ q_emb
        retrieved = url_diverse_topk(sims, doc_urls)

        cat = q["category"]
        if cat not in cats:
            cats[cat] = {"h1": 0, "h3": 0, "h6": 0, "rr": 0.0, "n": 0}
        cats[cat]["h1"] += hit(retrieved, q["ground_truth_urls"], 1)
        cats[cat]["h3"] += hit(retrieved, q["ground_truth_urls"], 3)
        cats[cat]["h6"] += hit(retrieved, q["ground_truth_urls"], 6)
        cats[cat]["rr"] += reciprocal_rank(retrieved, q["ground_truth_urls"])
        cats[cat]["n"] += 1

    total = {"h1": 0, "h3": 0, "h6": 0, "rr": 0.0, "n": 0}
    for v in cats.values():
        for k in total:
            total[k] += v[k]

    def fmt(d):
        n = d["n"]
        return (round(d["h1"] / n, 3), round(d["h3"] / n, 3),
                round(d["h6"] / n, 3), round(d["rr"] / n, 3))

    print(f"\n  {'Category':<22} {'n':>4} {'R@1':>6} {'R@3':>6} {'R@6':>6} {'MRR':>6}")
    print(f"  {'-'*52}")
    r1, r3, r6, mrr = fmt(total)
    print(f"  {'All':<22} {total['n']:>4} {r1:>6.3f} {r3:>6.3f} {r6:>6.3f} {mrr:>6.3f}")
    for cat in ["Rietveld", "Sequential", "Structure", "Calibration", "Scripting", "Export"]:
        if cat in cats:
            r1c, r3c, r6c, mrrc = fmt(cats[cat])
            print(f"  {cat:<22} {cats[cat]['n']:>4} {r1c:>6.3f} {r3c:>6.3f} {r6c:>6.3f} {mrrc:>6.3f}")

    return {"model": model_name, "label": label,
            "R@1": r1, "R@3": r3, "R@6": r6, "MRR": mrr}


def main():
    questions = load_questions()
    print(f"Loaded {len(questions)} questions")

    import chromadb
    print(f"Loading production index: {PROD_DB}")
    prod = chromadb.PersistentClient(path=PROD_DB)
    col = prod.get_collection(COLLECTION)
    data = col.get(include=["documents", "metadatas"])
    doc_texts = data["documents"]
    doc_urls = [m.get("url", "") for m in data["metadatas"]]
    print(f"Loaded {len(doc_texts)} chunks\n")

    all_results = []
    for model_name, label in MODELS:
        result = run_model(model_name, label, doc_texts, doc_urls, questions)
        all_results.append(result)

    bge_baseline = {"model": "BAAI/bge-base-en-v1.5",
                    "label": "bge-base-en-v1.5 (baseline, CN - excluded)",
                    "R@1": 0.500, "R@3": 0.825, "R@6": 0.975, "MRR": 0.661}
    prior = [
        {"model": "nomic-ai/nomic-embed-text-v1.5", "label": "nomic-embed-text-v1.5 (768-dim, US)",
         "R@1": 0.400, "R@3": 0.725, "R@6": 0.875, "MRR": 0.573},
        {"model": "mixedbread-ai/mxbai-embed-large-v1", "label": "mxbai-embed-large-v1 (1024-dim, DE)",
         "R@1": 0.425, "R@3": 0.650, "R@6": 0.875, "MRR": 0.579},
        {"model": "nomic-ai/nomic-embed-text-v1", "label": "nomic-embed-text-v1 (768-dim, US)",
         "R@1": 0.350, "R@3": 0.675, "R@6": 0.850, "MRR": 0.526},
        {"model": "snowflake/snowflake-arctic-embed-m", "label": "arctic-embed-m (768-dim, US) [prior run]",
         "R@1": 0.075, "R@3": 0.175, "R@6": 0.275, "MRR": 0.145},
        {"model": "snowflake/snowflake-arctic-embed-l", "label": "arctic-embed-l (1024-dim, US) [prior run]",
         "R@1": 0.100, "R@3": 0.150, "R@6": 0.200, "MRR": 0.133},
    ]

    print(f"\n\n{'='*60}")
    print("FINAL COMPARISON (5838-chunk full index)")
    print(f"{'='*60}")
    print(f"{'Model':<44} {'R@1':>6} {'R@3':>6} {'R@6':>6} {'MRR':>6}")
    print("-" * 64)
    for r in [bge_baseline] + prior + all_results:
        print(f"{r['label']:<44} {r['R@1']:>6.3f} {r['R@3']:>6.3f} {r['R@6']:>6.3f} {r['MRR']:>6.3f}")

    out = Path(__file__).parent / "eval_results_more_candidates.json"
    with open(out, "w") as f:
        json.dump([bge_baseline] + prior + all_results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
