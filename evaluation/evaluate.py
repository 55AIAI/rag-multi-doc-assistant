"""Measure retrieval against an isolated API account and synthetic document."""
import json
import os
import statistics
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
API = os.getenv("API_URL", "http://localhost:8000")
questions = json.loads((ROOT / "datasets/questions.json").read_text(encoding="utf-8"))
username = "eval_" + uuid.uuid4().hex[:12]
email = username + "@example.test"
password = uuid.uuid4().hex + uuid.uuid4().hex

with httpx.Client(base_url=API, timeout=180) as client:
    client.post("/auth/register", json={"username": username, "email": email, "password": password}).raise_for_status()
    token = client.post("/auth/login", json={"email": email, "password": password})
    token.raise_for_status()
    client.headers["Authorization"] = "Bearer " + token.json()["access_token"]
    with (ROOT / "datasets/sample.txt").open("rb") as sample:
        uploaded = client.post("/documents", files={"file": ("sample.txt", sample, "text/plain")})
    uploaded.raise_for_status()
    document_id = uploaded.json()["id"]
    results = {}
    try:
        for mode in (False, True):
            rows = []
            for example in questions:
                started = time.perf_counter()
                response = client.post("/rag/search", json={"question": example["question"],
                    "document_ids": [document_id], "rerank": mode, "top_k": 3})
                response.raise_for_status()
                sources = response.json()["sources"]
                rank = next((index + 1 for index, source in enumerate(sources)
                    if source["filename"] == example["expected_source"]
                    and example["expected_answer"].lower() in source["preview"].lower()), None)
                rows.append({"question": example["question"], "rank": rank,
                             "retrieval_ms": response.json()["retrieval_ms"],
                             "http_latency_ms": round((time.perf_counter() - started) * 1000, 2)})
            results["rerank" if mode else "dense"] = {
                "hit_at_3": sum(row["rank"] is not None for row in rows) / len(rows),
                "mrr": sum(1 / row["rank"] if row["rank"] else 0 for row in rows) / len(rows),
                "mean_retrieval_ms": statistics.mean(row["retrieval_ms"] for row in rows),
                "mean_http_latency_ms": statistics.mean(row["http_latency_ms"] for row in rows),
                "examples": rows}
    finally:
        client.delete("/documents/" + document_id).raise_for_status()
ROOT.joinpath("results").mkdir(exist_ok=True)
output = ROOT / "results" / "latest.json"
output.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(output, json.dumps(results, indent=2))
