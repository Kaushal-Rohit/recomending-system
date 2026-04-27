# Cross-Platform User-Behaviour Recommendation System

A recommendation engine that learns from what users watch, listen to, and search across multiple platforms (YouTube, Spotify, Google, Netflix, etc.) and returns personalised content recommendations.

> 📖 See **[PROJECT_JOURNAL.md](PROJECT_JOURNAL.md)** for a full step-by-step account of how this project was built, all design decisions, and how each algorithm works.

---

## Features

- 🎬 Track video watches, music listens, and search queries across platforms
- 🤖 **Content-based filtering** — recommends items similar to what you already liked (TF-IDF + cosine similarity)
- 👥 **Collaborative filtering** — recommends items liked by users with a similar taste profile
- 🔀 **Hybrid recommendations** — blends both algorithms for better coverage
- 🌐 **REST API** (Flask) — easy to integrate with any front-end or data pipeline

---

## Project Structure

```
recomending-system/
├── PROJECT_JOURNAL.md       ← full project documentation and progress log
├── README.md
├── requirements.txt
├── main.py                  ← start the API server
├── data/
│   ├── user_behavior.csv    ← user interaction events
│   └── content_catalog.csv  ← item metadata (videos, music, articles, podcasts)
├── src/
│   ├── data_models.py       ← typed dataclasses for events and catalogue items
│   ├── data_loader.py       ← CSV loading, validation, user-item matrix builder
│   ├── recommender.py       ← content-based and collaborative engines
│   └── api.py               ← Flask REST API
└── tests/
    └── test_recommender.py  ← unit tests
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the server
```bash
python main.py
```

### 3. Get recommendations
```bash
# Content-based (what you watch/listen to)
curl http://localhost:5000/recommend/content/u001?n=5

# Collaborative (what similar users enjoy)
curl http://localhost:5000/recommend/collaborative/u001?n=5

# Hybrid (blend of both)
curl http://localhost:5000/recommend/hybrid/u001?n=5&alpha=0.5

# Your behaviour profile
curl http://localhost:5000/users/u001/profile

# All content in the catalogue
curl http://localhost:5000/catalog
```

### 4. Ingest a new event
```bash
curl -X POST http://localhost:5000/events \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u001",
    "item_id": "item_042",
    "event_type": "watch",
    "platform": "youtube",
    "duration_sec": 320
  }'
```

Supported `event_type` values: `watch`, `listen`, `search`, `like`, `skip`

---

## Run Tests
```bash
python -m pytest tests/ -v
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/events` | Ingest a user behaviour event |
| GET | `/recommend/content/<user_id>` | Content-based recommendations |
| GET | `/recommend/collaborative/<user_id>` | Collaborative recommendations |
| GET | `/recommend/hybrid/<user_id>` | Hybrid recommendations |
| GET | `/users/<user_id>/profile` | User behaviour profile summary |
| GET | `/catalog` | List all catalogue items |

---

## How It Works

See **[PROJECT_JOURNAL.md](PROJECT_JOURNAL.md)** §6 for a full algorithm walkthrough.