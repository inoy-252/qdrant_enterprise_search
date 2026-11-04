# Qdrant Enterprise Vector Search & Caching Engine

High-performance semantic vector retrieval and caching engine leveraging Qdrant and Google Gemini dense embeddings.

---

## Architecture Overview

Traditional relational databases and linear SQLite scans degrade to $O(N)$ latency when searching through large-scale document corpora or cached vector spaces. This engine implements:

- **HNSW (Hierarchical Navigable Small World) Indexing:** Multi-layer skip-graph vector traversal delivering sub-millisecond similarity lookups.
- **Dense Embeddings:** 768-dimensional representations generated via Google Gemini (`gemini-embedding-001`).
- **Disk-Backed Persistence:** Embedded local storage without external server dependencies.

---

## Quickstart

### Prerequisites
- Python 3.10+
- Google Gemini API Key

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## Author
Yasir Ahmad Lone (@inoy-252)
