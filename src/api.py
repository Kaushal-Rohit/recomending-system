"""
Flask REST API for the recommendation system.

Endpoints:
  GET  /health                           — health check
  POST /events                           — ingest a user behaviour event
  GET  /recommend/content/<user_id>      — content-based recommendations
  GET  /recommend/collaborative/<user_id>— collaborative recommendations
  GET  /users/<user_id>/profile          — user behaviour profile summary
  GET  /catalog                          — list all catalogue items
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from http import HTTPStatus

from flask import Flask, jsonify, request

from src.data_loader import DataLoader
from src.data_models import UserEvent
from src.recommender import CollaborativeRecommender, ContentBasedRecommender

logger = logging.getLogger(__name__)


def create_app(loader: DataLoader | None = None) -> Flask:
    """Application factory — accepts an optional pre-built DataLoader (useful in tests)."""
    app = Flask(__name__)

    # ------------------------------------------------------------------
    # Initialise data layer and recommenders
    # ------------------------------------------------------------------
    if loader is None:
        loader = DataLoader()
    content_rec = ContentBasedRecommender(loader)
    collab_rec = CollaborativeRecommender(loader)

    # Fit content-based model on startup
    content_rec.fit()

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()})

    @app.post("/events")
    def ingest_event():
        """
        Ingest one user-behaviour event.

        Body (JSON):
          user_id      string  required
          item_id      string  required
          event_type   string  required  (watch|listen|search|like|skip)
          platform     string  required
          duration_sec float   optional  default 0
          search_query string  optional
          timestamp    string  optional  ISO-8601
        """
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Request body must be valid JSON."}), HTTPStatus.BAD_REQUEST

        required = ("user_id", "item_id", "event_type", "platform")
        missing = [f for f in required if not data.get(f)]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), HTTPStatus.BAD_REQUEST

        try:
            event = UserEvent.from_dict(data)
        except (KeyError, ValueError):
            return jsonify({"error": "Invalid field value. Check 'event_type' is one of: watch, listen, search, like, skip."}), HTTPStatus.BAD_REQUEST

        loader.add_event(event)
        # Re-fit content recommender so new item interactions are reflected
        content_rec.fit()

        return jsonify({"status": "ok", "event_id": event.event_id}), HTTPStatus.CREATED

    @app.get("/recommend/content/<user_id>")
    def recommend_content(user_id: str):
        """Return content-based recommendations for *user_id*."""
        n = _parse_n(request.args.get("n", 10))
        recs = content_rec.recommend(user_id, n=n)
        return jsonify({
            "user_id": user_id,
            "algorithm": "content_based",
            "count": len(recs),
            "recommendations": recs,
        })

    @app.get("/recommend/collaborative/<user_id>")
    def recommend_collaborative(user_id: str):
        """Return collaborative-filtering recommendations for *user_id*."""
        n = _parse_n(request.args.get("n", 10))
        recs = collab_rec.recommend(user_id, n=n)
        return jsonify({
            "user_id": user_id,
            "algorithm": "collaborative",
            "count": len(recs),
            "recommendations": recs,
        })

    @app.get("/recommend/hybrid/<user_id>")
    def recommend_hybrid(user_id: str):
        """
        Hybrid recommendations — blends content-based and collaborative scores.

        Query params:
          n     int    number of results (default 10)
          alpha float  weight for content-based score (default 0.5)
        """
        n = _parse_n(request.args.get("n", 10))
        try:
            alpha = float(request.args.get("alpha", 0.5))
            alpha = max(0.0, min(1.0, alpha))
        except ValueError:
            alpha = 0.5

        content_recs = {r["item_id"]: r["score"] for r in content_rec.recommend(user_id, n=n * 3)}
        collab_recs = {r["item_id"]: r["score"] for r in collab_rec.recommend(user_id, n=n * 3)}

        all_item_ids = set(content_recs) | set(collab_recs)
        blended: list[dict] = []

        catalog = {item.item_id: item for item in loader.get_catalog_items()}
        for item_id in all_item_ids:
            cs = content_recs.get(item_id, 0.0)
            cf = collab_recs.get(item_id, 0.0)
            hybrid_score = alpha * cs + (1 - alpha) * cf
            if item_id in catalog:
                item = catalog[item_id]
                blended.append({
                    "item_id": item_id,
                    "title": item.title,
                    "content_type": item.content_type.value if hasattr(item.content_type, "value") else item.content_type,
                    "genre": item.genre,
                    "creator": item.creator,
                    "platform": item.platform,
                    "score": round(hybrid_score, 4),
                    "content_score": round(cs, 4),
                    "collab_score": round(cf, 4),
                })

        blended.sort(key=lambda x: x["score"], reverse=True)
        return jsonify({
            "user_id": user_id,
            "algorithm": "hybrid",
            "alpha": alpha,
            "count": len(blended[:n]),
            "recommendations": blended[:n],
        })

    @app.get("/users/<user_id>/profile")
    def user_profile(user_id: str):
        """Return a summary of the user's behavioural profile."""
        events = loader.get_user_history(user_id)
        if not events:
            return jsonify({"error": f"No history found for user '{user_id}'."}), HTTPStatus.NOT_FOUND

        from collections import Counter

        event_counts = Counter(
            e.event_type.value if hasattr(e.event_type, "value") else e.event_type
            for e in events
        )
        platform_counts = Counter(e.platform for e in events)
        total_watch_time = sum(
            e.duration_sec
            for e in events
            if (e.event_type.value if hasattr(e.event_type, "value") else e.event_type)
            in ("watch", "listen")
        )
        searches = [
            e.search_query
            for e in events
            if (e.event_type.value if hasattr(e.event_type, "value") else e.event_type) == "search"
            and e.search_query
        ]

        return jsonify({
            "user_id": user_id,
            "total_events": len(events),
            "event_breakdown": dict(event_counts),
            "platform_breakdown": dict(platform_counts),
            "total_watch_listen_time_sec": total_watch_time,
            "recent_searches": searches[-10:],
        })

    @app.get("/catalog")
    def catalog():
        """Return all items in the content catalogue."""
        items = loader.get_catalog_items()
        return jsonify({
            "count": len(items),
            "items": [item.to_dict() for item in items],
        })

    return app


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_n(value) -> int:
    try:
        n = int(value)
        return max(1, min(n, 100))
    except (TypeError, ValueError):
        return 10
