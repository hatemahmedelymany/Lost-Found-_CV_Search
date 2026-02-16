# Lost & Found - CV Image Search (CLIP + FastAPI + MariaDB + Docker)

Computer Vision project for **image-based retrieval** of lost & found items using **CLIP embeddings** and category-aware matching.  
Backend is **FastAPI** + **MariaDB**, fully **Dockerized**. (Optional Flutter Web UI can consume the API.)

---
## Evaluation (CPU)
- Dataset: 10 items × 20 augmentations = 200 image queries
- Recall@1 = 0.90
- Recall@10 = 0.90
- Latency: p50 = 2.44s, p95 = 2.56s

## Quick Start (Docker)
> This repo keeps backend files inside the `app/` folder.

### 1) Run
```bash
docker compose -f app/docker-compose.yml up --build


