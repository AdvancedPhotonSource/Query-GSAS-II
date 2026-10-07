"""
Full 40-question evaluation against the live production ChromaDB index,
now re-embedded with nomic-embed-text-v1.5. Uses the real rag._retrieve()
code path (not a standalone numpy comparison) so these numbers reflect
exactly what a user would get.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gsas_query.rag import _retrieve

QUESTIONS_PATH = Path(__file__).parent / "questions.json"


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


def main():
    questions = load_questions()
    print(f"Loaded {len(questions)} questions\n")

    cats = {}
    per_question = []
    for q in questions:
        _, sources, _, _ = _retrieve(q["question"])
        urls = [s["url"] for s in sources]
        cat = q["category"]
        if cat not in cats:
            cats[cat] = {"h1": 0, "h3": 0, "h6": 0, "rr": 0.0, "n": 0}
        h1 = hit(urls, q["ground_truth_urls"], 1)
        h3 = hit(urls, q["ground_truth_urls"], 3)
        h6 = hit(urls, q["ground_truth_urls"], 6)
        rr = reciprocal_rank(urls, q["ground_truth_urls"])
        cats[cat]["h1"] += h1
        cats[cat]["h3"] += h3
        cats[cat]["h6"] += h6
        cats[cat]["rr"] += rr
        cats[cat]["n"] += 1
        per_question.append({
            "id": q["id"], "category": cat, "h1": h1, "h3": h3, "h6": h6,
            "rr": round(rr, 3), "top_url": urls[0] if urls else None,
        })
        print(f"  Q{q['id']:>2} [{cat:<11}] @1={int(h1)} @3={int(h3)} @6={int(h6)} rr={rr:.2f}  {urls[0] if urls else ''}")

    total = {"h1": 0, "h3": 0, "h6": 0, "rr": 0.0, "n": 0}
    for v in cats.values():
        for k in total:
            total[k] += v[k]

    def fmt(d):
        n = d["n"]
        return (round(d["h1"] / n, 3), round(d["h3"] / n, 3),
                round(d["h6"] / n, 3), round(d["rr"] / n, 3))

    print(f"\n{'Category':<22} {'n':>4} {'R@1':>6} {'R@3':>6} {'R@6':>6} {'MRR':>6}")
    print("-" * 52)
    r1, r3, r6, mrr = fmt(total)
    print(f"{'All':<22} {total['n']:>4} {r1:>6.3f} {r3:>6.3f} {r6:>6.3f} {mrr:>6.3f}")
    summary = {"All": {"n": total["n"], "recall@1": r1, "recall@3": r3, "recall@6": r6, "mrr": mrr}}
    for cat in ["Rietveld", "Sequential", "Structure", "Calibration", "Scripting", "Export"]:
        if cat in cats:
            r1c, r3c, r6c, mrrc = fmt(cats[cat])
            print(f"{cat:<22} {cats[cat]['n']:>4} {r1c:>6.3f} {r3c:>6.3f} {r6c:>6.3f} {mrrc:>6.3f}")
            summary[cat] = {"n": cats[cat]["n"], "recall@1": r1c, "recall@3": r3c, "recall@6": r6c, "mrr": mrrc}

    out = Path(__file__).parent / "eval_results_nomic_production.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "per_question": per_question}, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
