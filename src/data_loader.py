"""
Data loader and pre-processor.

Reads user-behaviour events and the content catalogue from CSV files,
validates the schema, and builds the derived structures (e.g. user-item
interaction matrix) used by the recommendation algorithms.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from src.data_models import EVENT_WEIGHTS, ContentItem, EventType, UserEvent

_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class DataLoader:
    """Load and pre-process data from CSV files (or inject DataFrames directly for testing)."""

    BEHAVIOR_COLUMNS = [
        "event_id",
        "user_id",
        "item_id",
        "event_type",
        "platform",
        "duration_sec",
        "timestamp",
        "search_query",
    ]

    CATALOG_COLUMNS = [
        "item_id",
        "title",
        "content_type",
        "genre",
        "tags",
        "creator",
        "platform",
        "language",
        "duration_sec",
    ]

    def __init__(
        self,
        behavior_df: Optional[pd.DataFrame] = None,
        catalog_df: Optional[pd.DataFrame] = None,
        data_dir: str = _DEFAULT_DATA_DIR,
    ):
        self._data_dir = data_dir
        self._behavior_path = os.path.join(data_dir, "user_behavior.csv")
        self._catalog_path = os.path.join(data_dir, "content_catalog.csv")

        if behavior_df is not None and catalog_df is not None:
            self.behavior = behavior_df.copy()
            self.catalog = catalog_df.copy()
            self._persist_enabled = False  # in-memory only (e.g. tests)
        else:
            self.behavior = self._load_behavior()
            self.catalog = self._load_catalog()
            self._persist_enabled = True

        self._user_item_matrix: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_user_item_matrix(self) -> pd.DataFrame:
        """Return (and cache) a users × items interaction-score matrix."""
        if self._user_item_matrix is None:
            self._user_item_matrix = self._build_user_item_matrix()
        return self._user_item_matrix

    def get_catalog_items(self) -> list[ContentItem]:
        """Return all catalogue entries as ContentItem objects."""
        return [ContentItem.from_dict(row) for _, row in self.catalog.iterrows()]

    def get_user_history(self, user_id: str) -> list[UserEvent]:
        """Return all events for a given user as UserEvent objects."""
        rows = self.behavior[self.behavior["user_id"] == user_id]
        return [UserEvent.from_dict(row) for _, row in rows.iterrows()]

    def add_event(self, event: UserEvent) -> None:
        """Append a new event and invalidate the cached interaction matrix."""
        new_row = pd.DataFrame([event.to_dict()])
        self.behavior = pd.concat([self.behavior, new_row], ignore_index=True)
        self._user_item_matrix = None  # invalidate cache
        self._persist_behavior()

    def all_user_ids(self) -> list[str]:
        return self.behavior["user_id"].unique().tolist()

    def all_item_ids(self) -> list[str]:
        return self.catalog["item_id"].unique().tolist()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_behavior(self) -> pd.DataFrame:
        df = pd.read_csv(self._behavior_path, dtype=str)
        df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce").fillna(0.0)
        df["search_query"] = df["search_query"].fillna("")
        # Keep only known event types; drop malformed rows silently
        valid = [e.value for e in EventType]
        df = df[df["event_type"].isin(valid)].reset_index(drop=True)
        return df

    def _load_catalog(self) -> pd.DataFrame:
        df = pd.read_csv(self._catalog_path, dtype=str)
        df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce").fillna(0.0)
        for col in ("genre", "tags", "creator", "platform", "language"):
            df[col] = df[col].fillna("")
        return df

    def _build_user_item_matrix(self) -> pd.DataFrame:
        """
        Build a user × item matrix where each cell holds the *total interaction
        score* accumulated by that user for that item.

        Score = Σ (event_weight × recency_factor) for each event on that item.
        """
        weight_map = {k.value if isinstance(k, EventType) else k: v for k, v in EVENT_WEIGHTS.items()}

        df = self.behavior.copy()
        df["score"] = df["event_type"].map(weight_map).fillna(0.0)

        matrix = (
            df.groupby(["user_id", "item_id"])["score"]
            .sum()
            .unstack(fill_value=0.0)
        )
        # Clip at 0 to avoid very negative values dominating cosine similarity
        matrix = matrix.clip(lower=0.0)
        return matrix

    def _persist_behavior(self) -> None:
        """Write the current in-memory behavior DataFrame back to CSV."""
        if not self._persist_enabled:
            return
        try:
            os.makedirs(self._data_dir, exist_ok=True)
            self.behavior.to_csv(self._behavior_path, index=False)
        except OSError:
            pass  # Non-critical — in-memory state is still valid
