"""
FastAPI + (Pinecone lub fallback in-memory) – full example (GCP/us‑central1)
Stack: SentenceTransformers all‑MiniLM‑L6‑v2 (384D) + Pinecone (cosine) **or** in‑memory (if Pinecone is not available)

Endpoints:
    - GET  /health
    - POST /upsert-items            (id, text, metadata)
    - POST /recommend/item          content-based recommendations text/ID
    - POST /recommend/user          recommendations from user profile ( vector average )
    - POST /recommend/session       session recommendations with MMR diversification

    # 1) add data ( shoes/clothes )
    curl -X POST http://localhost:8000/upsert-items \
         -H 'Content-Type: application/json' \
         -d '{
               "items": [
                 {"id":"p1","text":"Buty biegowe z dobrą amortyzacją","metadata":{"cat":"obuwie","price":299}},
                 {"id":"p2","text":"Buty trailowe z agresywnym bieżnikiem","metadata":{"cat":"obuwie","price":349}},
                 {"id":"p3","text":"Lekka koszulka do biegania","metadata":{"cat":"odziez","price":99}},
                 {"id":"p4","text":"Buty do biegania na asfalt, lekkie i szybkie","metadata":{"cat":"obuwie","price":399}},
                 {"id":"p5","text":"Skarpety kompresyjne do biegania","metadata":{"cat":"odziez","price":59}}
               ]
             }'

        Translation:
        # Running shoes with good amortization
        # Trailing shoes with aggressive rolling
        # Light tshirt for running
        # Shoes for running on asphalt, lightweight and fast
        # Compression socks for running


    # 2) Recommendations content based for description
    curl -X POST http://localhost:8000/recommend/item \
         -H 'Content-Type: application/json' \
         -d '{"text":"buty do biegania na asfalt z dobrą amortyzacją","k":5,
              "filter":{"cat":{"$eq":"obuwie"}}}'

        Translation:
        # shoes for running on asphalt with good amortization
        # shoes

    # 3) Recommendations from user profile
    curl -X POST http://localhost:8000/recommend/user \
         -H 'Content-Type: application/json' \
         -d '{"item_texts":["Buty biegowe na asfalt","Skarpety kompresyjne do biegania"],
              "k":5, "filter":{"cat":{"$in":["obuwie","odziez"]}}}'

        Translation:
        # Running shoes on asphalt, compression socks
        # shoes, clothes

    # 4) Session recommendations with MMR diversifications
    curl -X POST http://localhost:8000/recommend/session \
         -H 'Content-Type: application/json' \
         -d '{"query_text":"buty do biegania na asfalt","k":5,"base_top_k":30,"lambda":0.65,
              "filter":{"cat":{"$eq":"obuwie"}}}'

        Translation:
        # shoes for running on asphalt
        # shoes

"""

from __future__ import annotations
import os
import re
import sys
from typing import Any, Dict, List, Optional

# PRE-FLIGHT: check for ssl
try:
    import ssl as _ssl  # noqa: F401
    assert hasattr(_ssl, "SSLContext"), "SSL not supported – no SSLContext"
except Exception as e:
    msg = (
        "\n[ERROR] Python without 'ssl' – FastAPI i HTTPS won't work.\n"
    )
    print(msg, file=sys.stderr)
    raise SystemExit(msg) from e

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# SentenceTransformer Import with fallback
USING_ST_FALLBACK = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # sandbox / use light packet
    USING_ST_FALLBACK = True
    import hashlib

    class SentenceTransformer:  # noqa: N801
        """Light hash encoder fallback: (384D default).
        Caution: this is not semantic model, just for demo
        """
        def __init__(self, model_name: str = "fallback-hash-encoder", dim: Optional[int] = None):
            self.model_name = model_name
            try:
                self.dim = int(os.environ.get("EMBEDDING_DIM", "384")) if dim is None else int(dim)
            except Exception:
                self.dim = 384

        def _tok(self, text: str) -> List[str]:
            text = text.lower()
            # simple tokenization (letters/digits + polish dialectic symbols)
            tokens = re.findall(r"[a-zA-Z0-9ąćęłńóśźż]+", text, flags=re.UNICODE)
            return tokens

        def encode(self, text: str) -> np.ndarray:
            toks = self._tok(text)
            vec = np.zeros(self.dim, dtype=np.float32)
            # unigrams + bigrams: better for short phrases
            for i, tok in enumerate(toks):
                h = hashlib.sha256(tok.encode("utf-8")).hexdigest()
                idx = int(h[:8], 16) % self.dim
                vec[idx] += 1.0
                if i + 1 < len(toks):
                    bi = tok + "_" + toks[i + 1]
                    hb = hashlib.sha256(bi.encode("utf-8")).hexdigest()
                    idxb = int(hb[:8], 16) % self.dim
                    vec[idxb] += 0.5
            n = float(np.linalg.norm(vec))
            if n > 0:
                vec /= n
            return vec

# Import Pinecone with in‑memory fallback
USING_PINECONE = True
try:
    from pinecone import Pinecone, ServerlessSpec  # type: ignore
except Exception:
    USING_PINECONE = False
    Pinecone = object  # type: ignore
    ServerlessSpec = object  # type: ignore

    class _MemoryIndex:
        """Minimal, matching interface in-memory index.
        Supports: upsert, query, fetch + namespaces & simple filters ($eq, $in).
        """
        def __init__(self):
            # { namespace: { id: {"values": list[float], "metadata": dict} } }
            self.db: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # API compatible with pinecone.Index.upsert
        def upsert(self, vectors: List[tuple], namespace: Optional[str] = None):
            ns = namespace or "default"
            bucket = self.db.setdefault(ns, {})
            for _id, values, meta in vectors:
                bucket[_id] = {"values": values, "metadata": meta or {}}

        # API compatible with pinecone.Index.fetch
        def fetch(self, ids: List[str], namespace: Optional[str] = None):
            ns = namespace or "default"
            bucket = self.db.get(ns, {})
            out = {"vectors": {}}
            for _id in ids:
                if _id in bucket:
                    row = bucket[_id]
                    out["vectors"][_id] = {"id": _id, "values": row["values"], "metadata": row.get("metadata", {})}
            return out

        # Simple evaluator of filter (supports $eq i $in just for level 1)
        def _match_filter(self, meta: Dict[str, Any], flt: Optional[Dict[str, Any]]) -> bool:
            if not flt:
                return True
            for k, cond in flt.items():
                if isinstance(cond, dict):
                    if "$eq" in cond and meta.get(k) != cond["$eq"]:
                        return False
                    if "$in" in cond and meta.get(k) not in cond["$in"]:
                        return False
                else:
                    if meta.get(k) != cond:
                        return False
            return True

        # API compatible with pinecone.Index.query (subset)
        def query(self, *, vector: List[float], top_k: int, include_metadata: bool = False,
                  include_values: bool = False, namespace: Optional[str] = None, filter: Optional[Dict[str, Any]] = None):
            ns = namespace or "default"
            bucket = self.db.get(ns, {})
            if not bucket:
                return {"matches": []}

            q = np.array(vector, dtype=np.float32)
            qn = np.linalg.norm(q) + 1e-8
            q = q / qn

            matches = []
            for _id, row in bucket.items():
                if not self._match_filter(row.get("metadata", {}), filter):
                    continue
                v = np.array(row["values"], dtype=np.float32)
                vn = np.linalg.norm(v) + 1e-8
                v = v / vn
                score = float(np.dot(q, v))  # cosine for normalized
                m = {"id": _id, "score": score}
                if include_metadata:
                    m["metadata"] = row.get("metadata", {})
                if include_values:
                    m["values"] = row["values"]
                matches.append(m)

            matches.sort(key=lambda x: x["score"], reverse=True)
            return {"matches": matches[: top_k]}


# Configuration
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "recs-index")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "gpc")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-central1")  # GCP us-central1 (Standard/Enterprise)
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))  # MiniLM-L6-v2 / 384
METRIC = os.environ.get("VECTOR_METRIC", "cosine")
SKIP_PINECONE_STARTUP = os.environ.get("SKIP_PINECONE_STARTUP") == "1"

# Lazy-init globals (model + index backend)
app = FastAPI(title="Recommendations: FastAPI + (Pinecone or in‑memory)")
_model: Optional[SentenceTransformer] = None
_pc = None  # Pinecone client or None
_index = None  # pinecone.index.Index or _MemoryIndex

# Models Pydantic (request/response)
class ItemIn(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class UpsertRequest(BaseModel):
    items: List[ItemIn]
    namespace: Optional[str] = None

class RecommendItemRequest(BaseModel):
    text: Optional[str] = None
    id: Optional[str] = None
    k: int = 10
    filter: Optional[Dict[str, Any]] = None
    namespace: Optional[str] = None

class RecommendUserRequest(BaseModel):
    item_texts: List[str]
    weights: Optional[List[float]] = None
    k: int = 10
    filter: Optional[Dict[str, Any]] = None
    namespace: Optional[str] = None

class RecommendSessionRequest(BaseModel):
    query_text: str
    k: int = 10
    base_top_k: int = 30  # candidates for MMR
    lambda_: float = Field(0.7, ge=0.0, le=1.0, alias="lambda")
    filter: Optional[Dict[str, Any]] = None
    namespace: Optional[str] = None

class MatchOut(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RecommendResponse(BaseModel):
    results: List[MatchOut]


# Resources init
@app.on_event("startup")
def startup_event() -> None:
    global _model, _pc, _index

    # 1) Model embedings
    _model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 2) Index choosing
    use_memory = SKIP_PINECONE_STARTUP or (not USING_PINECONE)

    if use_memory:
        # Fallback
        _index = _MemoryIndex()
        return

    # 3) Pinecone
    if not PINECONE_API_KEY:
        raise RuntimeError(
            "No PINECONE_API_KEY – set env or  SKIP_PINECONE_STARTUP=1"
        )

    _pc = Pinecone(api_key=PINECONE_API_KEY)  # type: ignore[name-defined]

    existing = _pc.list_indexes().names()
    if PINECONE_INDEX_NAME not in existing:
        _pc.create_index(  # type: ignore[attr-defined]
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric=METRIC,
            spec=ServerlessSpec(  # type: ignore[name-defined]
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
            deletion_protection="disabled",
        )

    _index = _pc.Index(PINECONE_INDEX_NAME)


# Utils: embedding & normalization + MMR
def embed(text: str) -> np.ndarray:
    assert _model is not None
    v = _model.encode(text)
    # normalization – metrics cosine and local logic
    v = v / (np.linalg.norm(v) + 1e-8)
    return v.astype(np.float32)


def mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, cand_ids: List[str],
        lambda_: float = 0.7, k: int = 10) -> List[int]:
    """Returns indexes of candidates with MMR.
    query_vec: (D,)
    cand_vecs: (N, D) – *assuming vectors are normalized*
    """
    if len(cand_ids) == 0:
        return []
    k = min(k, len(cand_ids))

    # Similarity of query (cosine for normalized)
    sim_to_q = cand_vecs @ query_vec

    selected: List[int] = []
    remaining = list(range(len(cand_ids)))

    # 1. first: max similarity to query
    best_first = int(np.argmax(sim_to_q))
    selected.append(best_first)
    remaining.remove(best_first)

    # 2. following: max(lambda*sim(q, i) - (1-lambda)*max_j sim(i, j))
    while len(selected) < k and remaining:
        selected_vecs = cand_vecs[selected]
        sims_to_selected = cand_vecs[remaining] @ selected_vecs.T
        max_sims = sims_to_selected.max(axis=1)
        mmr_scores = lambda_ * sim_to_q[remaining] - (1.0 - lambda_) * max_sims
        pick_rel_idx = int(np.argmax(mmr_scores))
        pick = remaining[pick_rel_idx]
        selected.append(pick)
        remaining.remove(pick)

    return selected


# API Endpoints
@app.get("/health")
def health() -> Dict[str, str]:
    ssl_ok = "yes"
    return {
        "status": "ok",
        "index": PINECONE_INDEX_NAME,
        "region": PINECONE_REGION,
        "ssl": ssl_ok,
        "embedding_backend": "fallback" if USING_ST_FALLBACK else "sentence-transformers",
        "vector_index": "pinecone" if (USING_PINECONE and not SKIP_PINECONE_STARTUP) else "memory",
    }


@app.post("/upsert-items")
def upsert_items(req: UpsertRequest) -> Dict[str, Any]:
    if not req.items:
        raise HTTPException(400, "No upserts elemets")
    vectors = []
    for it in req.items:
        v = embed(it.text).tolist()
        vectors.append((it.id, v, it.metadata))
    _index.upsert(vectors, namespace=req.namespace)
    return {"upserted": len(vectors), "namespace": req.namespace or "default"}


@app.post("/recommend/item", response_model=RecommendResponse)
def recommend_item(req: RecommendItemRequest) -> RecommendResponse:
    if (req.text is None) == (req.id is None):
        raise HTTPException(400, "Pass exact one: text or id")

    namespace = req.namespace

    # 1) query vector
    if req.text is not None:
        qvec = embed(req.text).tolist()
    else:
        # fetch of vector value by id
        fetched = _index.fetch([req.id], namespace=namespace)
        vec_data = fetched.get("vectors", {}).get(req.id)
        if not vec_data:
            raise HTTPException(404, f"No vector for id={req.id}")
        qvec = vec_data["values"]

    # 2) query for index
    res = _index.query(
        vector=qvec, top_k=req.k, include_metadata=True, namespace=namespace, filter=req.filter
    )
    matches = [
        MatchOut(id=m["id"], score=float(m["score"]), metadata=m.get("metadata", {}))
        for m in res.get("matches", [])
    ]
    return RecommendResponse(results=matches)


@app.post("/recommend/user", response_model=RecommendResponse)
def recommend_user(req: RecommendUserRequest) -> RecommendResponse:
    if not req.item_texts:
        raise HTTPException(400, "List item_texts can not be empty")

    vecs = np.stack([embed(t) for t in req.item_texts])  # (N, D)
    if req.weights:
        if len(req.weights) != len(req.item_texts):
            raise HTTPException(400, "Weights has to match item_texts")
        w = np.array(req.weights, dtype=np.float32).reshape(-1, 1)
        uvec = (vecs * w).sum(axis=0) / (w.sum() + 1e-8)
    else:
        uvec = vecs.mean(axis=0)
    # normalization
    uvec = uvec / (np.linalg.norm(uvec) + 1e-8)

    res = _index.query(
        vector=uvec.tolist(),
        top_k=req.k,
        include_metadata=True,
        namespace=req.namespace,
        filter=req.filter,
    )
    matches = [
        MatchOut(id=m["id"], score=float(m["score"]), metadata=m.get("metadata", {}))
        for m in res.get("matches", [])
    ]
    return RecommendResponse(results=matches)


@app.post("/recommend/session", response_model=RecommendResponse)
def recommend_session(req: RecommendSessionRequest) -> RecommendResponse:
    qvec = embed(req.query_text)

    # 1) Retrieving candidate poll ( greater than k )
    base = _index.query(
        vector=qvec.tolist(),
        top_k=max(req.base_top_k, req.k),
        include_metadata=True,
        include_values=True,
        namespace=req.namespace,
        filter=req.filter,
    )

    matches = base.get("matches", [])
    if not matches:
        return RecommendResponse(results=[])

    cand_ids = [m["id"] for m in matches]
    cand_vecs = np.array([m["values"] for m in matches], dtype=np.float32)

    # 2) Make sure candidates are normalized
    norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-8
    cand_vecs = cand_vecs / norms

    # 3) MMR choosing
    chosen_idx = mmr(qvec, cand_vecs, cand_ids, lambda_=req.lambda_, k=req.k)
    picked = [matches[i] for i in chosen_idx]

    out = [
        MatchOut(id=m["id"], score=float(m["score"]), metadata=m.get("metadata", {}))
        for m in picked
    ]
    return RecommendResponse(results=out)


# Smoke check:  python app.py ( won't start server )
if __name__ == "__main__":
    print("[Self-test] Checking embed() & MMR…")
    # Test 1: vector length ≈ 1.0 after normalization
    os.environ["SKIP_PINECONE_STARTUP"] = "1"
    _model = SentenceTransformer(os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"))
    v1 = embed("Buty biegowe z dobrą amortyzacją")
    assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-3, "Embedding is not normalized"

    # Test 2: MMR returns k uniq indexes and prefers diversity
    q = v1
    cand = np.stack([
        v1,
        embed("Buty do biegania na asfalt z dobrą amortyzacją"), # Shoes for running on asphalt with good amortization
        embed("Lekka koszulka do biegania"), # Light tshirt for running
        embed("Skarpety kompresyjne do biegania"), # Compression socks for running
        embed("Buty trailowe na górskie ścieżki") # Trailing shoes for mountain trails
    ])
    ids = ["a","b","c","d","e"]
    out_idx = mmr(q, cand, ids, lambda_=0.65, k=3)
    assert len(out_idx) == len(set(out_idx)) == 3, "MMR returns duplicates or wrong cardinality"

    # Test 3 : fallback sentence-transformers – deterministic and dimension
    if USING_ST_FALLBACK:
        print("[Self-test] Fallback encoder – testing  deterministic and dimension")
        v2 = embed("Buty biegowe z dobrą amortyzacją") # Running shoes with good amortization
        v3 = embed("Buty biegowe z dobrą amortyzacją") # Running shoes with good amortization
        assert v2.shape[0] == int(os.environ.get("EMBEDDING_DIM", "384")), "Wrong fallback dimension"
        assert np.allclose(v2, v3, atol=1e-6), "Fallback encoder is not deterministic"

    # Test 4 (NOWY): fallback in‑memory index – upsert/query/fetch
    if not USING_PINECONE:
        print("[Self-test] Fallback in‑memory index – smoke test…")
        _index = _MemoryIndex()
        _index.upsert([("doc1", v1.tolist(), {"cat": "obuwie"})]) # shoes
        got = _index.fetch(["doc1"])  # should return values
        assert "doc1" in got.get("vectors", {}), "fetch did not return doc1"
        res = _index.query(vector=v1.tolist(), top_k=1, include_metadata=True, include_values=True)
        assert res.get("matches"), "query did not return results"
        m0 = res["matches"][0]
        assert m0["id"] == "doc1" and "values" in m0, "query/fetch not coherent with API"

    print("[Self-test] OK ✔")
    print("[Tip] To start API with fallback: export SKIP_PINECONE_STARTUP=1 && uvicorn app:app --port 8000")

