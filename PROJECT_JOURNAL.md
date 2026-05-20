# 📖 Project Journal — Cross-Platform User-Behaviour Recommendation System

> **Purpose:** This file is a living document. Every decision, design choice, algorithm, data-schema change, and milestone is recorded here so that anyone (including future contributors and the original author) can understand *where we are*, *how we got here*, and *how the system works*.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Approach](#3-approach)
4. [Architecture](#4-architecture)
5. [Data Schema](#5-data-schema)
6. [Recommendation Algorithms](#6-recommendation-algorithms)
7. [API Design](#7-api-design)
8. [Step-by-Step Progress Log](#8-step-by-step-progress-log)
9. [How to Run](#9-how-to-run)
10. [Glossary](#10-glossary)

---

## 1. Project Overview

**Goal:** Build a recommendation system that watches what a user does across multiple platforms (video streaming, music, search queries) and returns personalised content recommendations in real time.

**Key capabilities:**
- Ingest structured user-behaviour events (video watches, music listens, search queries).
- Build a per-user behavioural profile.
- Recommend items the user is likely to enjoy using two complementary algorithms:
  - **Content-based filtering** — recommend items similar to what the user already liked.
  - **Collaborative filtering** — recommend items liked by users with a similar taste profile.
- Expose recommendations through a lightweight REST API.

---

## 2. Problem Statement

> "I need to make a recommending system which recommends based on user behaviour — for example what he is watching, what kind of videos he was watching, what kind of music he was listening to, and searching across all platforms. We need to start from scratch."

In plain language:
- A user browses multiple platforms throughout the day.
- Every interaction (play video, play song, type search query) is an event we can capture.
- We want to use the *history* of those events to predict what the user will enjoy next.

---

## 3. Approach

### Phase 1 — Foundation (current)
1. Define clear data models for user-behaviour events.
2. Build synthetic / sample datasets for development and testing.
3. Implement two baseline recommendation algorithms (content-based, collaborative).
4. Wrap everything in a REST API so it is easy to integrate with any front-end.

### Phase 2 — Improvements (planned)
- Replace synthetic data with real uploaded data.
- Add a hybrid scorer that blends both algorithms.
- Introduce session-aware (sequential / recency-weighted) recommendations.
- Add A/B testing hooks so we can measure recommendation quality online.

### Phase 3 — Scale (future)
- Move to a proper vector database (e.g., FAISS, Pinecone) for fast similarity search.
- Introduce a real-time pipeline (Kafka / Redis Streams) for live event ingestion.
- Train a neural collaborative filtering model (NCF / two-tower).

---

## 4. Architecture

```
User Events (API POST)
        │
        ▼
┌──────────────────┐
│  Flask REST API  │  ← main entry point for clients
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────┐
│          Recommendation Engine        │
│                                       │
│  ┌──────────────────────────────┐    │
│  │  Content-Based Filter        │    │
│  │  (TF-IDF cosine similarity)  │    │
│  └──────────────────────────────┘    │
│  ┌──────────────────────────────┐    │
│  │  Collaborative Filter        │    │
│  │  (user-item matrix + cosine) │    │
│  └──────────────────────────────┘    │
└──────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│   Data Layer      │
│  (CSV / Pandas)   │  ← will be upgraded to a DB later
└──────────────────┘
```

### File layout
```
recomending-system/
├── PROJECT_JOURNAL.md       ← YOU ARE HERE
├── README.md
├── requirements.txt
├── main.py                  ← start the API server
├── data/
│   ├── user_behavior.csv    ← user interaction events
│   └── content_catalog.csv  ← metadata about each item (video/song/article)
├── src/
│   ├── __init__.py
│   ├── data_models.py       ← dataclasses / enums for typed events
│   ├── data_loader.py       ← load, validate, and pre-process CSV data
│   ├── recommender.py       ← recommendation engine (both algorithms)
│   └── api.py               ← Flask blueprint with all REST endpoints
└── tests/
    └── test_recommender.py  ← unit tests
```

---

## 5. Data Schema

### 5.1 User Behaviour Events — `data/user_behavior.csv`

| Column         | Type    | Description                                              |
|----------------|---------|----------------------------------------------------------|
| `event_id`     | str     | Unique identifier for the event                          |
| `user_id`      | str     | Identifier of the user who performed the action          |
| `item_id`      | str     | Identifier of the content item interacted with           |
| `event_type`   | str     | One of: `watch`, `listen`, `search`, `like`, `skip`      |
| `platform`     | str     | Platform where the event happened (e.g. `youtube`, `spotify`, `google`) |
| `duration_sec` | float   | Seconds spent on the item (0 for searches)               |
| `timestamp`    | datetime| When the event occurred (ISO-8601)                       |
| `search_query` | str     | The search query (only populated for `event_type=search`)|

### 5.2 Content Catalogue — `data/content_catalog.csv`

| Column       | Type  | Description                                             |
|--------------|-------|---------------------------------------------------------|
| `item_id`    | str   | Unique identifier matching `user_behavior.item_id`      |
| `title`      | str   | Human-readable title                                    |
| `content_type`| str  | One of: `video`, `music`, `article`, `podcast`          |
| `genre`      | str   | Comma-separated genre tags (e.g. `rock,pop`)            |
| `tags`       | str   | Comma-separated descriptive tags                        |
| `creator`    | str   | Creator / channel / artist name                         |
| `platform`   | str   | Source platform                                         |
| `language`   | str   | Language of the content                                 |
| `duration_sec`| float| Total duration of the item in seconds                   |

---

## 6. Recommendation Algorithms

### 6.1 Content-Based Filtering

**How it works:**
1. For each item in the catalogue, build a *feature string* by concatenating its `genre`, `tags`, `content_type`, `creator`, and `platform` fields.
2. Compute a **TF-IDF** matrix across all feature strings — this converts free-text metadata into numerical vectors.
3. For a given user:
   a. Look up all items the user has interacted with (weighted by a score — see below).
   b. Average the TF-IDF vectors of those items to create a **user profile vector**.
   c. Compute **cosine similarity** between the user profile and every unseen item.
   d. Return the top-N items by similarity.

**Interaction score weighting:**
| Event type | Weight |
|------------|--------|
| `like`     | 3.0    |
| `watch`    | 2.0    |
| `listen`   | 2.0    |
| `search`   | 1.0    |
| `skip`     | -1.0   |

Skipped items reduce the user's affinity for similar content.

**Pros:** Works well with no other users (cold-start friendly for new platforms).  
**Cons:** Tends to recommend *more of the same* (filter bubble risk).

---

### 6.2 Collaborative Filtering (User-Based)

**How it works:**
1. Build a **user-item interaction matrix** where rows = users, columns = items, cells = interaction score (0 if no interaction).
2. Compute **cosine similarity** between the target user's row-vector and every other user's row-vector.
3. Select the **K most similar users** (K = 10 by default).
4. For each candidate item that the target user hasn't seen:
   - Score = weighted average of similar users' scores for that item (weights = similarity scores).
5. Return the top-N items by score.

**Pros:** Serendipitous discoveries — surfaces items the user would never find from metadata alone.  
**Cons:** Requires enough historical data; cold-start problem for brand-new users.

---

### 6.3 Hybrid Blending (Phase 2)

The hybrid recommender will combine scores from both approaches:

```
final_score(item) = α × content_score(item) + (1 − α) × collaborative_score(item)
```

`α` will start at 0.5 and can be tuned per user based on data density (users with sparse history get higher `α` because content-based is more reliable; power users get lower `α` to benefit from collaborative discovery).

---

## 7. API Design

Base URL: `http://localhost:5000`

| Method | Endpoint                                   | Description                                      |
|--------|--------------------------------------------|--------------------------------------------------|
| GET    | `/health`                                  | Health check                                     |
| POST   | `/events`                                  | Ingest one or more user behaviour events         |
| GET    | `/recommend/content/<user_id>`             | Content-based recommendations for a user         |
| GET    | `/recommend/collaborative/<user_id>`       | Collaborative recommendations for a user         |
| GET    | `/recommend/hybrid/<user_id>`              | Hybrid recommendations (Phase 2)                 |
| GET    | `/users/<user_id>/profile`                 | Return the user's behavioural profile summary    |
| GET    | `/catalog`                                 | List all items in the content catalogue          |

### Request / Response examples

**POST /events**
```json
{
  "user_id": "u001",
  "item_id": "v042",
  "event_type": "watch",
  "platform": "youtube",
  "duration_sec": 320,
  "timestamp": "2024-01-15T14:30:00Z"
}
```
Response `201`:
```json
{ "status": "ok", "event_id": "e_abc123" }
```

**GET /recommend/content/u001?n=5**
Response `200`:
```json
{
  "user_id": "u001",
  "algorithm": "content_based",
  "recommendations": [
    { "item_id": "v099", "title": "...", "score": 0.87, "content_type": "video" },
    ...
  ]
}
```

---

## 8. Step-by-Step Progress Log

### Step 1 — Repository Initialisation ✅
- Created project skeleton: `src/`, `data/`, `tests/` directories.
- Added `requirements.txt` with core dependencies (`flask`, `pandas`, `scikit-learn`, `numpy`).
- Updated `README.md` with a quick-start guide.

### Step 2 — Data Models ✅
- Defined `EventType` enum (`watch`, `listen`, `search`, `like`, `skip`).
- Defined `ContentType` enum (`video`, `music`, `article`, `podcast`).
- Defined `UserEvent` dataclass to represent a single interaction event.
- Defined `ContentItem` dataclass to represent a catalogue entry.
- Both dataclasses include `to_dict()` / `from_dict()` helpers.

### Step 3 — Sample Data ✅
- Generated `data/user_behavior.csv` with 200 synthetic events across 10 users and 50 items on three platforms (youtube, spotify, google).
- Generated `data/content_catalog.csv` with 50 items spanning videos, music, articles, and podcasts with realistic genre and tag metadata.

### Step 4 — Data Loader ✅
- `DataLoader` class reads both CSV files and validates schema.
- Computes per-user interaction scores (see weight table in §6.1).
- Returns a cleaned `user_item_matrix` (DataFrame, users × items).

### Step 5 — Recommendation Engine ✅
- `ContentBasedRecommender` — TF-IDF + cosine similarity (see §6.1).
- `CollaborativeRecommender` — user-item matrix + cosine similarity (see §6.2).
- Both classes share a common `recommend(user_id, n)` interface.

### Step 6 — REST API ✅
- Flask application with all endpoints defined in §7.
- `/events` endpoint writes new events to the in-memory DataFrame (persisted to CSV on each write so restarts don't lose data).

### Step 7 — Tests ✅
- Unit tests for `DataLoader`, `ContentBasedRecommender`, and `CollaborativeRecommender`.
- Tests use a small in-memory fixture so they run without the CSV files.

---

## 9. How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Start the API server
```bash
python main.py
```
The server starts at `http://localhost:5000`.

### Get recommendations
```bash
# Content-based recommendations for user u001
curl http://localhost:5000/recommend/content/u001?n=5

# Collaborative recommendations
curl http://localhost:5000/recommend/collaborative/u001?n=5

# Ingest a new event
curl -X POST http://localhost:5000/events \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u001","item_id":"v042","event_type":"watch","platform":"youtube","duration_sec":320}'
```

### Run tests
```bash
python -m pytest tests/ -v
```

---

## 10. Glossary

| Term | Definition |
|------|------------|
| **Cold start** | The problem of making recommendations for a brand-new user or item with no historical data |
| **Collaborative filtering** | Recommending items based on what *similar users* liked |
| **Content-based filtering** | Recommending items similar to what *this user* already liked |
| **Cosine similarity** | A measure of similarity between two vectors (1 = identical direction, 0 = orthogonal) |
| **Hybrid recommender** | A system that blends multiple algorithms for better coverage and accuracy |
| **TF-IDF** | Term Frequency–Inverse Document Frequency; a numerical statistic reflecting how important a word is in a document relative to a corpus |
| **User-item matrix** | A table where rows are users, columns are items, and cells hold interaction scores |
| **Filter bubble** | The risk that a content-based system only recommends more of what the user already knows |
