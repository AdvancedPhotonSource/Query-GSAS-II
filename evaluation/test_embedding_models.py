"""
Quick A/B comparison of embedding models on the 40-question eval set.
Uses a temporary in-memory ChromaDB to avoid touching the production index.
"""
import json
import time
from pathlib import Path

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
PROD_DB = "/Users/b324240/.GSASII/query_gsas2/chroma_db"
TOP_K = 6

def load_questions():
    with open(QUESTIONS_PATH) as f:
        return json.load(f)["questions"]

def hit_at_k(retrieved, gt_urls, k):
    for url in retrieved[:k]:
        for gt in gt_urls:
            if url.rstrip("/") == gt.rstrip("/") or url.startswith(gt.rstrip("/")) or gt.startswith(url.rstrip("/")):
                return True
    return False

def rr(retrieved, gt_urls):
    for i, url in enumerate(retrieved, 1):
        for gt in gt_urls:
            if url.rstrip("/") == gt.rstrip("/") or url.startswith(gt.rstrip("/")) or gt.startswith(url.rstrip("/")):
                return 1.0 / i
    return 0.0

def evaluate_with_ef(ef, questions, db_path=PROD_DB):
    import chromadb
    client = chromadb.PersistentClient(path=db_path)
    col = client.get_collection("gsasii_docs")
    docs = col.get(include=["documents", "metadatas"])

    # Build a temporary in-memory collection with the new EF
    tmp = chromadb.Client()
    tmp_col = tmp.get_or_create_collection("tmp", embedding_function=ef,
                                            metadata={"hnsw:space": "cosine"})
    print(f"  Adding {len(docs['ids'])} chunks to tmp collection...")
    batch = 500
    for i in range(0, len(docs['ids']), batch):
        tmp_col.add(
            ids=docs['ids'][i:i+batch],
            documents=docs['documents'][i:i+batch],
            metadatas=docs['metadatas'][i:i+batch],
        )

    hits1 = hits3 = hits6 = total_rr = 0
    for q in questions:
        results = tmp_col.query(query_texts=[q["question"]], n_results=TOP_K,
                                include=["metadatas"])
        urls = [m.get("url", "") for m in results["metadatas"][0]]
        hits1 += hit_at_k(urls, q["ground_truth_urls"], 1)
        hits3 += hit_at_k(urls, q["ground_truth_urls"], 3)
        hits6 += hit_at_k(urls, q["ground_truth_urls"], 6)
        total_rr += rr(urls, q["ground_truth_urls"])

    n = len(questions)
    return {
        "R@1": round(hits1/n, 3),
        "R@3": round(hits3/n, 3),
        "R@6": round(hits6/n, 3),
        "MRR": round(total_rr/n, 3),
        "hits": (hits1, hits3, hits6),
    }

def main():
    questions = load_questions()
    print(f"Loaded {len(questions)} questions\n")

    models_to_test = [
        ("BAAI/bge-small-en-v1.5", "BGE-small (384-dim, 67MB)"),
        ("BAAI/bge-base-en-v1.5",  "BGE-base  (768-dim, 210MB)"),
    ]

    from chromadb.utils.embedding_functions import FastEmbedEmbeddingFunction
    from fastembed import TextEmbedding

    results = {}
    for model_name, label in models_to_test:
        print(f"Testing {label}...")
        try:
            ef = FastEmbedEmbeddingFunction(model_name=model_name)
            t0 = time.time()
            metrics = evaluate_with_ef(ef, questions)
            elapsed = time.time() - t0
            results[label] = metrics
            print(f"  Done in {elapsed:.0f}s: R@1={metrics['R@1']} R@3={metrics['R@3']} R@6={metrics['R@6']} MRR={metrics['MRR']}\n")
        except Exception as e:
            print(f"  FAILED: {e}\n")

    print("=== COMPARISON ===")
    print(f"{'Model':<35} {'R@1':>6} {'R@3':>6} {'R@6':>6} {'MRR':>6}")
    print("-" * 60)
    print(f"{'MiniLM-L6-v2 (384-dim, 90MB) [current]':<35} {'0.325':>6} {'0.600':>6} {'0.725':>6} {'0.464':>6}")
    for label, m in results.items():
        print(f"{label:<35} {m['R@1']:>6} {m['R@3']:>6} {m['R@6']:>6} {m['MRR']:>6}")

if __name__ == "__main__":
    main()
