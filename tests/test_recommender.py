"""
Unit tests for the recommendation system.

Tests cover:
- DataLoader schema validation and matrix building
- ContentBasedRecommender (fit + recommend)
- CollaborativeRecommender (recommend)
- Flask API endpoints
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader
from src.data_models import ContentType, EventType, UserEvent, ContentItem
from src.recommender import CollaborativeRecommender, ContentBasedRecommender


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_catalog_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "item_id": "v001", "title": "Python Tutorial", "content_type": "video",
            "genre": "tech,education", "tags": "python,programming,beginner",
            "creator": "TechChannel", "platform": "youtube", "language": "en", "duration_sec": 600,
        },
        {
            "item_id": "v002", "title": "Machine Learning Basics", "content_type": "video",
            "genre": "tech,education", "tags": "ml,ai,data",
            "creator": "AILab", "platform": "youtube", "language": "en", "duration_sec": 900,
        },
        {
            "item_id": "m001", "title": "Lo-Fi Beats", "content_type": "music",
            "genre": "electronic,chill", "tags": "lofi,study,relax",
            "creator": "ChillArtist", "platform": "spotify", "language": "en", "duration_sec": 300,
        },
        {
            "item_id": "m002", "title": "Jazz Vibes", "content_type": "music",
            "genre": "jazz,blues", "tags": "jazz,instrumental,classic",
            "creator": "JazzMaster", "platform": "spotify", "language": "en", "duration_sec": 240,
        },
        {
            "item_id": "a001", "title": "AI News Today", "content_type": "article",
            "genre": "technology,ai", "tags": "ai,news,research",
            "creator": "TechCrunch", "platform": "medium", "language": "en", "duration_sec": 0,
        },
    ])


def _make_behavior_df() -> pd.DataFrame:
    return pd.DataFrame([
        # User u001 watches tech videos and listens to lo-fi
        {"event_id": "e001", "user_id": "u001", "item_id": "v001", "event_type": "watch",
         "platform": "youtube", "duration_sec": 580, "timestamp": "2024-01-01T10:00:00+00:00", "search_query": ""},
        {"event_id": "e002", "user_id": "u001", "item_id": "m001", "event_type": "listen",
         "platform": "spotify", "duration_sec": 290, "timestamp": "2024-01-02T12:00:00+00:00", "search_query": ""},
        {"event_id": "e003", "user_id": "u001", "item_id": "v001", "event_type": "like",
         "platform": "youtube", "duration_sec": 0, "timestamp": "2024-01-02T10:05:00+00:00", "search_query": ""},
        # User u002 watches ML content
        {"event_id": "e004", "user_id": "u002", "item_id": "v002", "event_type": "watch",
         "platform": "youtube", "duration_sec": 880, "timestamp": "2024-01-03T09:00:00+00:00", "search_query": ""},
        {"event_id": "e005", "user_id": "u002", "item_id": "a001", "event_type": "search",
         "platform": "google", "duration_sec": 0, "timestamp": "2024-01-03T09:30:00+00:00", "search_query": "machine learning news"},
        {"event_id": "e006", "user_id": "u002", "item_id": "v001", "event_type": "watch",
         "platform": "youtube", "duration_sec": 600, "timestamp": "2024-01-04T11:00:00+00:00", "search_query": ""},
        # User u003 likes jazz
        {"event_id": "e007", "user_id": "u003", "item_id": "m002", "event_type": "listen",
         "platform": "spotify", "duration_sec": 230, "timestamp": "2024-01-05T08:00:00+00:00", "search_query": ""},
        {"event_id": "e008", "user_id": "u003", "item_id": "m001", "event_type": "skip",
         "platform": "spotify", "duration_sec": 10, "timestamp": "2024-01-05T08:05:00+00:00", "search_query": ""},
    ])


@pytest.fixture
def loader():
    return DataLoader(
        behavior_df=_make_behavior_df(),
        catalog_df=_make_catalog_df(),
    )


@pytest.fixture
def content_rec(loader):
    rec = ContentBasedRecommender(loader)
    rec.fit()
    return rec


@pytest.fixture
def collab_rec(loader):
    return CollaborativeRecommender(loader, n_neighbors=3)


# ---------------------------------------------------------------------------
# DataLoader tests
# ---------------------------------------------------------------------------


class TestDataLoader:
    def test_loads_catalog(self, loader):
        items = loader.get_catalog_items()
        assert len(items) == 5

    def test_catalog_item_types(self, loader):
        items = loader.get_catalog_items()
        types = {item.content_type for item in items}
        assert ContentType.VIDEO in types
        assert ContentType.MUSIC in types

    def test_user_item_matrix_shape(self, loader):
        matrix = loader.get_user_item_matrix()
        # 3 users and the interacted items are columns
        assert matrix.shape[0] == 3
        assert matrix.shape[1] >= 1

    def test_user_item_matrix_scores_non_negative(self, loader):
        matrix = loader.get_user_item_matrix()
        assert (matrix.values >= 0).all()

    def test_user_history_returns_events(self, loader):
        history = loader.get_user_history("u001")
        assert len(history) == 3

    def test_user_history_empty_for_unknown(self, loader):
        history = loader.get_user_history("unknown_user")
        assert history == []

    def test_add_event_updates_matrix(self, loader):
        before = loader.get_user_item_matrix()
        event = UserEvent(
            user_id="u001",
            item_id="a001",
            event_type=EventType.WATCH,
            platform="medium",
        )
        loader.add_event(event)
        after = loader.get_user_item_matrix()
        # The matrix must now include a001 for u001
        assert "a001" in after.columns
        assert after.loc["u001", "a001"] > 0

    def test_all_user_ids(self, loader):
        users = loader.all_user_ids()
        assert "u001" in users
        assert "u002" in users
        assert "u003" in users

    def test_all_item_ids(self, loader):
        item_ids = loader.all_item_ids()
        assert "v001" in item_ids


# ---------------------------------------------------------------------------
# ContentBasedRecommender tests
# ---------------------------------------------------------------------------


class TestContentBasedRecommender:
    def test_fit_succeeds(self, loader):
        rec = ContentBasedRecommender(loader)
        result = rec.fit()
        assert result is rec  # returns self

    def test_recommend_returns_list(self, content_rec):
        recs = content_rec.recommend("u001", n=3)
        assert isinstance(recs, list)
        assert len(recs) <= 3

    def test_recommend_excludes_seen_items(self, content_rec):
        recs = content_rec.recommend("u001", n=10)
        seen = {"v001", "m001"}
        returned_ids = {r["item_id"] for r in recs}
        assert returned_ids.isdisjoint(seen)

    def test_recommend_contains_required_keys(self, content_rec):
        recs = content_rec.recommend("u001", n=1)
        if recs:
            keys = set(recs[0].keys())
            assert {"item_id", "title", "score", "content_type"}.issubset(keys)

    def test_recommend_unknown_user_returns_fallback(self, content_rec):
        recs = content_rec.recommend("unknown_user", n=3)
        assert isinstance(recs, list)

    def test_recommend_scores_between_0_and_1(self, content_rec):
        recs = content_rec.recommend("u001", n=10)
        for r in recs:
            assert 0.0 <= r["score"] <= 1.0

    def test_tech_user_gets_tech_recommendations(self, content_rec):
        # u001 watched tech video + liked it → should get tech items back
        recs = content_rec.recommend("u001", n=5)
        genres = [r["genre"] for r in recs]
        tech_related = any("tech" in g or "education" in g or "ai" in g for g in genres)
        assert tech_related


# ---------------------------------------------------------------------------
# CollaborativeRecommender tests
# ---------------------------------------------------------------------------


class TestCollaborativeRecommender:
    def test_recommend_returns_list(self, collab_rec):
        recs = collab_rec.recommend("u001", n=3)
        assert isinstance(recs, list)

    def test_recommend_excludes_seen_items(self, collab_rec, loader):
        seen = set(loader.get_user_item_matrix().columns[loader.get_user_item_matrix().loc["u001"] != 0].tolist())
        recs = collab_rec.recommend("u001", n=10)
        returned_ids = {r["item_id"] for r in recs}
        assert returned_ids.isdisjoint(seen)

    def test_recommend_unknown_user_returns_fallback(self, collab_rec):
        recs = collab_rec.recommend("ghost_user", n=3)
        assert isinstance(recs, list)

    def test_recommend_contains_required_keys(self, collab_rec):
        recs = collab_rec.recommend("u001", n=1)
        if recs:
            keys = set(recs[0].keys())
            assert {"item_id", "title", "score"}.issubset(keys)


# ---------------------------------------------------------------------------
# Flask API tests
# ---------------------------------------------------------------------------


class TestFlaskAPI:
    @pytest.fixture
    def client(self, loader):
        from src.api import create_app
        app = create_app(loader=loader)
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"

    def test_catalog(self, client):
        resp = client.get("/catalog")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 5

    def test_recommend_content(self, client):
        resp = client.get("/recommend/content/u001?n=3")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["algorithm"] == "content_based"
        assert isinstance(data["recommendations"], list)

    def test_recommend_collaborative(self, client):
        resp = client.get("/recommend/collaborative/u001?n=3")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["algorithm"] == "collaborative"

    def test_recommend_hybrid(self, client):
        resp = client.get("/recommend/hybrid/u001?n=3&alpha=0.6")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["algorithm"] == "hybrid"
        assert data["alpha"] == pytest.approx(0.6, abs=1e-4)

    def test_ingest_event_valid(self, client):
        payload = {
            "user_id": "u001",
            "item_id": "a001",
            "event_type": "watch",
            "platform": "medium",
            "duration_sec": 120,
        }
        resp = client.post("/events", json=payload)
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "event_id" in data

    def test_ingest_event_missing_field(self, client):
        payload = {"user_id": "u001", "item_id": "a001"}  # missing event_type, platform
        resp = client.post("/events", json=payload)
        assert resp.status_code == 400

    def test_ingest_event_bad_json(self, client):
        resp = client.post("/events", data="not json", content_type="text/plain")
        assert resp.status_code == 400

    def test_user_profile(self, client):
        resp = client.get("/users/u001/profile")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["user_id"] == "u001"
        assert data["total_events"] == 3

    def test_user_profile_not_found(self, client):
        resp = client.get("/users/nobody/profile")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DataModels tests
# ---------------------------------------------------------------------------


class TestDataModels:
    def test_user_event_roundtrip(self):
        event = UserEvent(
            user_id="u001",
            item_id="v001",
            event_type=EventType.WATCH,
            platform="youtube",
            duration_sec=300,
        )
        d = event.to_dict()
        restored = UserEvent.from_dict(d)
        assert restored.user_id == event.user_id
        assert restored.item_id == event.item_id
        assert restored.event_type == event.event_type

    def test_content_item_feature_string(self):
        item = ContentItem(
            item_id="v001",
            title="Test Video",
            content_type=ContentType.VIDEO,
            genre="tech,education",
            tags="python,beginner",
            creator="TestChannel",
            platform="youtube",
        )
        fs = item.feature_string
        assert "tech" in fs
        assert "education" in fs
        assert "python" in fs
        assert "TestChannel" in fs

    def test_content_item_roundtrip(self):
        item = ContentItem(
            item_id="v001",
            title="Test Video",
            content_type=ContentType.VIDEO,
            genre="tech",
            tags="python",
            creator="Creator",
            platform="youtube",
        )
        d = item.to_dict()
        restored = ContentItem.from_dict(d)
        assert restored.item_id == item.item_id
        assert restored.content_type == item.content_type
