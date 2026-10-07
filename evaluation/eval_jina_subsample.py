"""
Lightweight Jina-embeddings-v2-base-en eval on a reduced chunk subsample,
to avoid heavy memory pressure on this machine (swap was nearly full during
the full 5854-chunk run). Keeps all chunks whose URL matches a ground-truth
URL for the 40 questions, plus a random fill, capped at SAMPLE_SIZE total.
"""
import json
import random
import time
from pathlib import Path

import numpy as np

PROD_DB = "/Users/b324240/.GSASII/query_gsas2/chroma_db"
COLLECTION = "gsasii_docs"
QUESTIONS_PATH = Path(__file__).parent / "questions.json"
FETCH_K = 30
TOP_K = 6
SAMPLE_SIZE = 1200

random.seed(0)


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


def main():
    questions = load_questions()
    print(f"Loaded {len(questions)} questions")

    import chromadb
    print(f"Loading production index: {PROD_DB}")
    prod = chromadb.PersistentClient(path=PROD_DB)
    col = prod.get_collection(COLLECTION)
    data = col.get(include=["documents", "metadatas"])
    doc_texts_all = data["documents"]
    doc_urls_all = [m.get("url", "") for m in data["metadatas"]]
    print(f"Loaded {len(doc_texts_all)} chunks total")

    gt_urls_flat = set()
    for q in questions:
        for u in q["ground_truth_urls"]:
            gt_urls_flat.add(u.rstrip("/"))

    must_keep_idx = [
        i for i, u in enumerate(doc_urls_all)
        if any(u.rstrip("/") == gt or u.rstrip("/").startswith(gt) or gt.startswith(u.rstrip("/"))
               for gt in gt_urls_flat)
    ]
    must_keep_set = set(must_keep_idx)
    remaining = [i for i in range(len(doc_texts_all)) if i not in must_keep_set]
    fill_n = max(0, SAMPLE_SIZE - len(must_keep_idx))
    fill_idx = random.sample(remaining, min(fill_n, len(remaining)))
    keep_idx = must_keep_idx + fill_idx

    doc_texts = [doc_texts_all[i] for i in keep_idx]
    doc_urls = [doc_urls_all[i] for i in keep_idx]
    print(f"Subsample: {len(doc_texts)} chunks ({len(must_keep_idx)} ground-truth-matching + {len(fill_idx)} random fill)")

    from fastembed import TextEmbedding
    model_name = "jinaai/jina-embeddings-v2-base-en"
    print(f"\nLoading model: {model_name}")
    fe = TextEmbedding(model_name)

    t0 = time.time()
    batch = 64
    all_embs = []
    n = len(doc_texts)
    for i in range(0, n, batch):
        batch_embs = list(fe.embed(doc_texts[i:i + batch]))
        all_embs.extend(batch_embs)
        print(f"  Embedded {min(i+batch, n)}/{n} chunks...")
    doc_matrix = np.array(all_embs, dtype=np.float32)
    print(f"Embedding done in {time.time()-t0:.0f}s  shape={doc_matrix.shape}")

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

    r1, r3, r6, mrr = fmt(total)
    print(f"\n{'Category':<22} {'n':>4} {'R@1':>6} {'R@3':>6} {'R@6':>6} {'MRR':>6}")
    print("-" * 52)
    print(f"{'All':<22} {total['n']:>4} {r1:>6.3f} {r3:>6.3f} {r6:>6.3f} {mrr:>6.3f}")
    for cat in ["Rietveld", "Sequential", "Structure", "Calibration", "Scripting", "Export"]:
        if cat in cats:
            r1c, r3c, r6c, mrrc = fmt(cats[cat])
            print(f"{cat:<22} {cats[cat]['n']:>4} {r1c:>6.3f} {r3c:>6.3f} {r6c:>6.3f} {mrrc:>6.3f}")

    print("\nNOTE: subsample eval (1200 of 5854 chunks, all GT chunks retained).")
    print("Numbers are optimistic relative to full-index eval (less distractor")
    print("competition), so compare shape/ranking only, not absolute magnitude.")


if __name__ == "__main__":
    main()
