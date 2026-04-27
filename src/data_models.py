"""
Data models for the recommendation system.

Defines enums and dataclasses used throughout the project to represent
user-behaviour events and catalogue items in a typed, validated way.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class EventType(str, Enum):
    """Types of user interaction events tracked across platforms."""

    WATCH = "watch"
    LISTEN = "listen"
    SEARCH = "search"
    LIKE = "like"
    SKIP = "skip"


class ContentType(str, Enum):
    """Types of content items in the catalogue."""

    VIDEO = "video"
    MUSIC = "music"
    ARTICLE = "article"
    PODCAST = "podcast"


# Interaction score weights used by the recommendation engine.
# Higher weight → stronger positive signal; negative → user disliked this.
EVENT_WEIGHTS: dict[str, float] = {
    EventType.LIKE: 3.0,
    EventType.WATCH: 2.0,
    EventType.LISTEN: 2.0,
    EventType.SEARCH: 1.0,
    EventType.SKIP: -1.0,
}


@dataclass
class UserEvent:
    """A single user-interaction event (e.g. watched a video, searched a query)."""

    user_id: str
    item_id: str
    event_type: EventType
    platform: str
    duration_sec: float = 0.0
    search_query: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_id: str = field(default_factory=lambda: f"e_{uuid.uuid4().hex[:8]}")

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "item_id": self.item_id,
            "event_type": self.event_type.value if isinstance(self.event_type, EventType) else self.event_type,
            "platform": self.platform,
            "duration_sec": self.duration_sec,
            "search_query": self.search_query or "",
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserEvent":
        return cls(
            event_id=data.get("event_id", f"e_{uuid.uuid4().hex[:8]}"),
            user_id=data["user_id"],
            item_id=data["item_id"],
            event_type=EventType(data["event_type"]),
            platform=data["platform"],
            duration_sec=float(data.get("duration_sec", 0.0)),
            search_query=data.get("search_query") or None,
            timestamp=datetime.fromisoformat(str(data["timestamp"])) if data.get("timestamp") else datetime.now(timezone.utc),
        )


@dataclass
class ContentItem:
    """A single item in the content catalogue (video, song, article, podcast)."""

    item_id: str
    title: str
    content_type: ContentType
    genre: str
    tags: str
    creator: str
    platform: str
    language: str = "en"
    duration_sec: float = 0.0

    @property
    def feature_string(self) -> str:
        """Concatenated feature text used by the TF-IDF vectoriser."""
        return " ".join(
            filter(
                None,
                [
                    self.content_type.value if isinstance(self.content_type, ContentType) else self.content_type,
                    self.genre.replace(",", " "),
                    self.tags.replace(",", " "),
                    self.creator,
                    self.platform,
                    self.language,
                ],
            )
        )

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "title": self.title,
            "content_type": self.content_type.value if isinstance(self.content_type, ContentType) else self.content_type,
            "genre": self.genre,
            "tags": self.tags,
            "creator": self.creator,
            "platform": self.platform,
            "language": self.language,
            "duration_sec": self.duration_sec,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContentItem":
        return cls(
            item_id=data["item_id"],
            title=data["title"],
            content_type=ContentType(data["content_type"]),
            genre=data.get("genre", ""),
            tags=data.get("tags", ""),
            creator=data.get("creator", ""),
            platform=data.get("platform", ""),
            language=data.get("language", "en"),
            duration_sec=float(data.get("duration_sec", 0.0)),
        )
