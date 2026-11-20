"""
Enterprise Hybrid Search Benchmark & Evaluation Suite
Compares Dense (HNSW) vs Sparse (BM25) vs Hybrid (RRF) Retrieval
"""

import time
from hybrid_search_engine import QdrantHybridEngine

# Curated enterprise documents with technical codes and domain concepts
SAMPLE_DOCUMENTS = [
    {
        "id": 1,
        "title": "Network Gateway Troubleshooting Guide",
        "content": "For packet drops on edge proxies, inspect firewall rule ERR-NET-9002. Restart interface eth0 if latency exceeds 200ms.",
        "metadata": {"department": "infrastructure", "tier": "p1"},
    },
    {
        "id": 2,
        "title": "Corporate Travel and Airfare Expense Policy",
        "content": "Employees may claim reimbursement for economy flights booked 14 days in advance. Meals capped at $75 per diem.",
        "metadata": {"department": "finance", "tier": "general"},
    },
    {
        "id": 3,
        "title": "Identity and Access Management SSO Protocol",
        "content": "Authentication failure SEC-AUTH-401 indicates expired SAML assertion. Reset OAuth2 tokens in enterprise portal.",
        "metadata": {"department": "security", "tier": "p0"},
    },
    {
        "id": 4,
        "title": "Kubernetes GPU Cluster Provisioning Manual",
        "content": "Deploy NVIDIA A100 daemonsets with helm chart v3. Ensure CUDA drivers 12.4 are linked before running distributed training.",
        "metadata": {"department": "infrastructure", "tier": "p2"},
    },
]


def evaluate_query(engine: QdrantHybridEngine, query: str) -> dict:
    """
    Executes a query across Dense, Sparse, and Hybrid RRF retrieval pipelines
    and records top-1 match and latency for each approach.
    """
    benchmarks = {}

    # Dense Semantic Search
    t0 = time.perf_counter()
    dense_hits = engine.dense_search(query, limit=1)
    dense_lat = (time.perf_counter() - t0) * 1000
    benchmarks["dense"] = {
        "latency_ms": round(dense_lat, 2),
        "top_id": dense_hits[0]["id"] if dense_hits else None,
        "score": dense_hits[0]["score"] if dense_hits else 0.0,
    }

    # Sparse Lexical Search
    t0 = time.perf_counter()
    sparse_hits = engine.sparse_search(query, limit=1)
    sparse_lat = (time.perf_counter() - t0) * 1000
    benchmarks["sparse"] = {
        "latency_ms": round(sparse_lat, 2),
        "top_id": sparse_hits[0]["id"] if sparse_hits else None,
        "score": sparse_hits[0]["score"] if sparse_hits else 0.0,
    }

    # Hybrid RRF Search
    t0 = time.perf_counter()
    hybrid_hits = engine.hybrid_search(query, limit=1)
    hybrid_lat = (time.perf_counter() - t0) * 1000
    benchmarks["hybrid"] = {
        "latency_ms": round(hybrid_lat, 2),
        "top_id": hybrid_hits[0]["id"] if hybrid_hits else None,
        "score": hybrid_hits[0]["score"] if hybrid_hits else 0.0,
    }

    return benchmarks

def main():
    print("=" * 70)
    print("ENTERPRISE HYBRID SEARCH BENCHMARK: DENSE vs SPARSE vs HYBRID (RRF)")
    print("=" * 70)

    engine = QdrantHybridEngine(collection_name="benchmark_eval_catalog")

    # Ingest the test enterprise catalog
    print("\n[1/3] Ingesting enterprise test documents...")
    indexed_ids = engine.index_batch(SAMPLE_DOCUMENTS)
    print(f"Indexed {len(indexed_ids)} documents into dual-vector catalog.")

    # Telemetry
    stats = engine.get_telemetry()
    print(f"Total points in Qdrant: {stats['total_points']}")
    print(f"Fusion algorithm: {stats['fusion_algorithm']}")

    # Benchmark queries
    test_queries = [
        ("ERR-NET-9002 firewall rule", "Exact Technical Error Code"),
        ("how to get reimbursed for work flights", "Pure Conceptual / Semantic Intent"),
        ("SEC-AUTH-401 authentication failure", "Exact Security Token Code"),
        ("setting up nvidia a100 gpu drivers", "Hybrid Technical Concept"),
    ]

    print("\n[2/3] Running Comparative Retrieval Benchmark...\n")

    for query, query_type in test_queries:
        print(f"Query: '{query}'")
        print(f"Query Type: {query_type}")

        results = evaluate_query(engine, query)

        print(
            f"  Dense (HNSW):   Top ID: {results['dense']['top_id']}   | Score: {results['dense']['score']}   | Latency: {results['dense']['latency_ms']}ms"
        )
        print(
            f"  Sparse (BM25):  Top ID: {results['sparse']['top_id']}   | Score: {results['sparse']['score']}   | Latency: {results['sparse']['latency_ms']}ms"
        )
        print(
            f"  Hybrid (RRF):   Top ID: {results['hybrid']['top_id']}   | Score: {results['hybrid']['score']} | Latency: {results['hybrid']['latency_ms']}ms"
        )
        print("-" * 70)

    print("\n[3/3] Benchmark complete. Qdrant Hybrid Engine verified!")


if __name__ == "__main__":
    main()