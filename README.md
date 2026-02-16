# Lost & Found - CV Image Search (CLIP + FastAPI + MariaDB + Docker)

Computer Vision project for **image-based retrieval** of lost & found items using **CLIP embeddings** and category-aware matching.  
Backend is **FastAPI** + **MariaDB**, fully **Dockerized**. (Optional Flutter Web UI can consume the API.)

---

## Quick Start (Docker)

> This repo keeps backend files inside the `app/` folder.

### 1) Run
```bash
docker compose -f app/docker-compose.yml up --build
