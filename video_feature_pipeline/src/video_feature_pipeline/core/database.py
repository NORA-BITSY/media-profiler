from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import sqlite_utils

from video_feature_pipeline.core.config import get_settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    source TEXT,
    local_path TEXT NOT NULL,
    duration REAL,
    fps REAL,
    width INTEGER,
    height INTEGER,
    download_ts TIMESTAMP,
    process_ts TIMESTAMP,
    status TEXT DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS spatial_features (
    video_id TEXT PRIMARY KEY REFERENCES videos(video_id),
    shoulder_width_mean REAL,
    shoulder_width_std REAL,
    torso_height_mean REAL,
    torso_height_std REAL,
    limb_lengths_json TEXT,
    face_landmark_distances_json TEXT,
    physical_signature_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS identities (
    identity_id TEXT PRIMARY KEY,
    video_id TEXT REFERENCES videos(video_id),
    cluster_id INTEGER,
    embedding_id TEXT,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS co_occurrences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_id_a TEXT NOT NULL,
    identity_id_b TEXT NOT NULL,
    video_id TEXT NOT NULL,
    frame_idx INTEGER NOT NULL,
    timestamp REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(identity_id_a, identity_id_b, video_id, frame_idx)
);

CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
CREATE INDEX IF NOT EXISTS idx_videos_source ON videos(source);
CREATE INDEX IF NOT EXISTS idx_identities_video_id ON identities(video_id);
CREATE INDEX IF NOT EXISTS idx_identities_embedding_id ON identities(embedding_id);
CREATE INDEX IF NOT EXISTS idx_co_occurrences_identity_a ON co_occurrences(identity_id_a);
CREATE INDEX IF NOT EXISTS idx_co_occurrences_identity_b ON co_occurrences(identity_id_b);
CREATE INDEX IF NOT EXISTS idx_co_occurrences_video_id ON co_occurrences(video_id);
"""


class Database:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or get_settings().paths.sqlite_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: sqlite_utils.Database | None = None
        self._init_db()

    def _init_db(self) -> None:
        self._db = sqlite_utils.Database(self.db_path)
        self._db.executescript(SCHEMA)

    @property
    def db(self) -> sqlite_utils.Database:
        if self._db is None:
            self._db = sqlite_utils.Database(self.db_path)
        return self._db

    @contextmanager
    def transaction(self):
        with self.db.conn:
            yield self.db

    def upsert_video(self, video_data: dict[str, Any]) -> None:
        self.db["videos"].upsert(video_data, pk="video_id")

    def get_video(self, video_id: str) -> dict[str, Any] | None:
        try:
            return self.db["videos"].get(video_id)
        except sqlite_utils.db.NotFoundError:
            return None

    def update_video_status(self, video_id: str, status: str) -> None:
        self.db["videos"].update(video_id, {"status": status})

    def list_videos(self, status: str | None = None, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        query = "SELECT * FROM videos"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY download_ts DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        return list(self.db.query(query, params))

    def upsert_spatial_features(self, video_id: str, features: dict[str, Any]) -> None:
        data = {"video_id": video_id, **features}
        self.db["spatial_features"].upsert(data, pk="video_id")

    def get_spatial_features(self, video_id: str) -> dict[str, Any] | None:
        try:
            return self.db["spatial_features"].get(video_id)
        except sqlite_utils.db.NotFoundError:
            return None

    def upsert_identity(self, identity_data: dict[str, Any]) -> None:
        self.db["identities"].upsert(identity_data, pk="identity_id")

    def get_identities_by_video(self, video_id: str) -> list[dict[str, Any]]:
        return list(self.db["identities"].rows_where("video_id = ?", [video_id]))

    def get_identity_by_embedding_id(self, embedding_id: str) -> dict[str, Any] | None:
        try:
            return self.db["identities"].get(embedding_id)
        except sqlite_utils.db.NotFoundError:
            return None

    def search_identities_by_embedding_ids(self, embedding_ids: list[str]) -> list[dict[str, Any]]:
        if not embedding_ids:
            return []
        placeholders = ",".join("?" * len(embedding_ids))
        return list(self.db.query(f"SELECT * FROM identities WHERE embedding_id IN ({placeholders})", embedding_ids))

    def add_co_occurrence(self, identity_id_a: str, identity_id_b: str, video_id: str, frame_idx: int, timestamp: float) -> None:
        if identity_id_a > identity_id_b:
            identity_id_a, identity_id_b = identity_id_b, identity_id_a
        self.db["co_occurrences"].insert({
            "identity_id_a": identity_id_a,
            "identity_id_b": identity_id_b,
            "video_id": video_id,
            "frame_idx": frame_idx,
            "timestamp": timestamp,
        }, ignore=True)

    def get_co_occurrences(self, identity_id: str) -> list[dict[str, Any]]:
        return list(self.db.query(
            """SELECT * FROM co_occurrences 
               WHERE identity_id_a = ? OR identity_id_b = ?""",
            [identity_id, identity_id]
        ))

    def get_co_occurrence_partners(self, identity_id: str) -> list[dict[str, Any]]:
        return list(self.db.query(
            """SELECT 
                CASE WHEN identity_id_a = ? THEN identity_id_b ELSE identity_id_a END as partner_id,
                COUNT(*) as co_count,
                MIN(created_at) as first_seen,
                MAX(created_at) as last_seen
               FROM co_occurrences 
               WHERE identity_id_a = ? OR identity_id_b = ?
               GROUP BY partner_id
               ORDER BY co_count DESC""",
            [identity_id, identity_id, identity_id]
        ))

    def get_faces_in_frame(self, video_id: str, frame_idx: int, window: int = 0) -> list[dict[str, Any]]:
        min_frame = frame_idx - window
        max_frame = frame_idx + window
        return list(self.db.query(
            """SELECT i.* FROM identities i
               WHERE i.video_id = ? AND i.frame_idx BETWEEN ? AND ?""",
            [video_id, min_frame, max_frame]
        ))

    def get_stats(self) -> dict[str, int]:
        return {
            "total_videos": self.db["videos"].count,
            "pending_videos": self.db["videos"].count_where("status = ?", ["pending"]),
            "processing_videos": self.db["videos"].count_where("status = ?", ["processing"]),
            "done_videos": self.db["videos"].count_where("status = ?", ["done"]),
            "failed_videos": self.db["videos"].count_where("status = ?", ["failed"]),
            "total_identities": self.db["identities"].count,
            "videos_with_spatial": self.db["spatial_features"].count,
            "total_co_occurrences": self.db["co_occurrences"].count,
        }

    def search_by_physical_signature(
        self,
        ratio_filters: dict[str, tuple[float, float]] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search videos by physical signature ratios.
        
        Args:
            ratio_filters: Dict of ratio_name -> (min, max) range
            limit: Max results
        """
        if not ratio_filters:
            return list(self.db.query(
                "SELECT v.video_id, v.source, sf.physical_signature_json "
                "FROM spatial_features sf JOIN videos v ON v.video_id = sf.video_id "
                "WHERE sf.physical_signature_json IS NOT NULL LIMIT ?", [limit]
            ))
        
        conditions = []
        params = []
        for ratio_name, (min_val, max_val) in ratio_filters.items():
            conditions.append(f"json_extract(physical_signature_json, '$.{ratio_name}') BETWEEN ? AND ?")
            params.extend([min_val, max_val])
        
        where_clause = " AND ".join(conditions)
        params.append(limit)
        
        query = f"""
            SELECT v.video_id, v.source, sf.physical_signature_json
            FROM spatial_features sf JOIN videos v ON v.video_id = sf.video_id
            WHERE sf.physical_signature_json IS NOT NULL AND {where_clause}
            LIMIT ?
        """
        return list(self.db.query(query, params))

_db_instance: Database | None = None


def get_database() -> Database:
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance