"""
Re-embed the production ChromaDB index with the new nomic-embed-text-v1.5
model (replacing the old BAAI/bge-base-en-v1.5 embeddings, which are
incompatible with the new query-time embedding function).

Designed for a memory-constrained machine:
  - tiny batch size (8 chunks at a time)
  - single-threaded ONNX session, memory arena/pattern disabled
  - incremental checkpoint file, so a kill/crash loses at most one batch
  - writes directly into a NEW collection (gsasii_docs_nomic) rather than
    mutating the live collection in place, so the old index stays usable
    until the swap is verified
"""
import json
import time
from pathlib import Path

import chromadb

PROD_DB = "/Users/b324240/.GSASII/query_gsas2/chroma_db"
OLD_COLLECTION = "gsasii_docs"
NEW_COLLECTION = "gsasii_docs_nomic"
CHECKPOINT = Path(__file__).parent / "reembed_checkpoint.json"
BATCH = 8


def get_low_memory_embedder():
    import onnxruntime as rt
    from tokenizers import Tokenizer
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from gsas_query._embed import get_model_dir, _download_model, _MODEL_FILES

    model_dir = get_model_dir()
    missing = [f for f in _MODEL_FILES if not (model_dir / f).exists()]
    if missing:
        _download_model(model_dir)

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tok.enable_padding()
    tok.enable_truncation(max_length=512)

    opts = rt.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.enable_mem_pattern = False
    opts.enable_cpu_mem_arena = False
    opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    sess = rt.InferenceSession(
        str(model_dir / "model.onnx"),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    has_token_type = any(inp.name == "token_type_ids" for inp in sess.get_inputs())

    import numpy as np

    def embed_batch(texts, prefix):
        prefixed = [prefix + t for t in texts]
        encodings = tok.encode_batch(prefixed)
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if has_token_type:
            feeds["token_type_ids"] = np.zeros_like(ids)
        last_hidden = sess.run(None, feeds)[0]
        fmask = mask[:, :, np.newaxis].astype(np.float32)
        emb = (last_hidden * fmask).sum(axis=1) / fmask.sum(axis=1).clip(1e-12)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-12)
        return emb.tolist()

    return embed_batch


def main():
    print("Connecting to production ChromaDB...", flush=True)
    client = chromadb.PersistentClient(path=PROD_DB)
    old_col = client.get_collection(OLD_COLLECTION)
    total = old_col.count()
    print(f"Source collection has {total} chunks", flush=True)

    new_col = client.get_or_create_collection(
        NEW_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    already_done = new_col.count()
    print(f"Target collection already has {already_done} chunks (resuming)", flush=True)

    embed_batch = get_low_memory_embedder()

    offset = already_done
    t_start = time.time()
    while offset < total:
        batch = old_col.get(
            include=["documents", "metadatas"],
            limit=BATCH,
            offset=offset,
        )
        if not batch["ids"]:
            break
        embeddings = embed_batch(batch["documents"], "search_document: ")
        new_col.upsert(
            ids=batch["ids"],
            documents=batch["documents"],
            metadatas=batch["metadatas"],
            embeddings=embeddings,
        )
        offset += len(batch["ids"])
        elapsed = time.time() - t_start
        rate = (offset - already_done) / elapsed if elapsed > 0 else 0
        eta_min = (total - offset) / rate / 60 if rate > 0 else float("inf")
        print(f"  {offset}/{total} done ({rate:.2f} chunks/s, ETA {eta_min:.1f} min)", flush=True)

    print(f"\nDone. New collection '{NEW_COLLECTION}' has {new_col.count()} chunks.", flush=True)


if __name__ == "__main__":
    main()
