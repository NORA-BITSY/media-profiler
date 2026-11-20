"""Database operations for photo enrichment records.

Supports SQLite (default, local-first) and PostgreSQL (via Immich instance).
Schema mirrors sql/enrich_schema.sql.
"""

import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union
from uuid import UUID

from .metadata import DBRecord, asset_to_record
from .models import Asset, AssetStatus, EntityType, NEREntity, SidecarStatus, VLMResult


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO datetime string from the DB, returning None on failure."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


class PhotoIndexDB:
    """SQLite-backed store for enrichment records keyed by asset."""

    def __init__(self, db_path: Union[str, Path] = "./photo_index.db") -> None:
        self.db_path = Path(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS enrich_assets (
                    asset_id      TEXT PRIMARY KEY,
                    content_sha256 TEXT,
                    file_path     TEXT NOT NULL,
                    file_name     TEXT NOT NULL,
                    file_size     INTEGER,
                    mime_type     TEXT,
                    status        TEXT DEFAULT 'pending',
                    indexed_at    TEXT,
                    analyzed_at   TEXT,
                    enriched_at   TEXT,
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS enrich_exif (
                    asset_id        TEXT PRIMARY KEY,
                    raw_json        TEXT NOT NULL,
                    structured_json TEXT,
                    FOREIGN KEY (asset_id) REFERENCES enrich_assets(asset_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS enrich_faces (
                    id            TEXT PRIMARY KEY,
                    asset_id      TEXT NOT NULL,
                    bbox_json     TEXT,
                    landmarks_json TEXT,
                    landmarks_468_json TEXT,
                    landmarks_3d_json TEXT,
                    embedding_json TEXT NOT NULL,
                    blendshapes_json TEXT,
                    expression_vector_json TEXT,
                    identity_vector_json TEXT,
                    head_pose_json TEXT,
                    eye_gaze_json TEXT,
                    iris_landmarks_json TEXT,
                    det_score     REAL,
                    age           INTEGER,
                    gender        INTEGER,
                    model_name    TEXT,
                    detected_at   TEXT,
                    FOREIGN KEY (asset_id) REFERENCES enrich_assets(asset_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_enrich_faces_asset
                    ON enrich_faces(asset_id);

                CREATE TABLE IF NOT EXISTS enrich_vlm (
                    asset_id        TEXT PRIMARY KEY,
                    model           TEXT,
                    caption         TEXT,
                    description     TEXT,
                    raw_response    TEXT,
                    processing_ms   INTEGER,
                    analyzed_at     TEXT,
                    FOREIGN KEY (asset_id) REFERENCES enrich_assets(asset_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS enrich_entities (
                    id            TEXT PRIMARY KEY,
                    asset_id      TEXT NOT NULL,
                    text          TEXT NOT NULL,
                    entity_type   TEXT NOT NULL,
                    label         TEXT,
                    confidence    REAL,
                    bbox_json     TEXT,
                    source        TEXT DEFAULT 'vlm',
                    start_char    INTEGER,
                    end_char      INTEGER,
                    FOREIGN KEY (asset_id) REFERENCES enrich_assets(asset_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS enrich_sidecars (
                    asset_id      TEXT PRIMARY KEY,
                    sidecar_path  TEXT,
                    status        TEXT,
                    FOREIGN KEY (asset_id) REFERENCES enrich_assets(asset_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_enrich_entities_asset
                    ON enrich_entities(asset_id);
                CREATE INDEX IF NOT EXISTS idx_enrich_entities_type
                    ON enrich_entities(entity_type);
                CREATE INDEX IF NOT EXISTS idx_enrich_assets_path
                    ON enrich_assets(file_path);
                CREATE INDEX IF NOT EXISTS idx_enrich_assets_sha
                    ON enrich_assets(content_sha256);

                CREATE TABLE IF NOT EXISTS enrich_tags (
                    id            TEXT PRIMARY KEY,
                    asset_id      TEXT NOT NULL,
                    tag_name      TEXT NOT NULL,
                    source        TEXT DEFAULT 'manual',
                    FOREIGN KEY (asset_id) REFERENCES enrich_assets(asset_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_enrich_tags_asset
                    ON enrich_tags(asset_id);
                CREATE INDEX IF NOT EXISTS idx_enrich_tags_name
                    ON enrich_tags(tag_name);

                CREATE TABLE IF NOT EXISTS enrich_persons (
                    id            TEXT PRIMARY KEY,
                    name          TEXT NOT NULL UNIQUE,
                    embedding_json TEXT NOT NULL,
                    created_at    TEXT
                );

                CREATE TABLE IF NOT EXISTS enrich_face_matches (
                    id            TEXT PRIMARY KEY,
                    face_id       TEXT NOT NULL,
                    person_id     TEXT NOT NULL,
                    similarity    REAL NOT NULL,
                    matched_at    TEXT,
                    FOREIGN KEY (face_id) REFERENCES enrich_faces(id) ON DELETE CASCADE,
                    FOREIGN KEY (person_id) REFERENCES enrich_persons(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_enrich_face_matches_face
                    ON enrich_face_matches(face_id);
                CREATE INDEX IF NOT EXISTS idx_enrich_face_matches_person
                    ON enrich_face_matches(person_id);

                -- Body pose detection results
                CREATE TABLE IF NOT EXISTS enrich_body_poses (
                    id            TEXT PRIMARY KEY,
                    asset_id      TEXT NOT NULL,
                    landmark_set  TEXT NOT NULL,
                    landmarks_2d_json TEXT NOT NULL,
                    landmarks_3d_json TEXT,
                    bbox_json     TEXT,
                    det_score     REAL,
                    model_name    TEXT,
                    detected_at   TEXT,
                    raw_response_json TEXT,
                    FOREIGN KEY (asset_id) REFERENCES enrich_assets(asset_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_enrich_body_poses_asset
                    ON enrich_body_poses(asset_id);
                CREATE INDEX IF NOT EXISTS idx_enrich_body_poses_model
                    ON enrich_body_poses(model_name);

                -- Body measurements (per-asset and per-performer)
                CREATE TABLE IF NOT EXISTS enrich_body_measurements (
                    id            TEXT PRIMARY KEY,
                    asset_id      TEXT,
                    performer_id  TEXT,
                    measurement_type TEXT NOT NULL,
                    value         REAL NOT NULL,
                    unit          TEXT NOT NULL DEFAULT 'cm',
                    confidence    REAL NOT NULL DEFAULT 0.0,
                    method        TEXT NOT NULL DEFAULT 'landmark_2d',
                    reference_scale REAL,
                    landmarks_used_json TEXT,
                    computed_at   TEXT,
                    notes         TEXT,
                    FOREIGN KEY (asset_id) REFERENCES enrich_assets(asset_id) ON DELETE CASCADE,
                    FOREIGN KEY (performer_id) REFERENCES enrich_performers(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_enrich_body_measurements_asset
                    ON enrich_body_measurements(asset_id);
                CREATE INDEX IF NOT EXISTS idx_enrich_body_measurements_performer
                    ON enrich_body_measurements(performer_id);
                CREATE INDEX IF NOT EXISTS idx_enrich_body_measurements_type
                    ON enrich_body_measurements(measurement_type);

                -- Facial biometrics (aggregated per performer)
                CREATE TABLE IF NOT EXISTS enrich_facial_biometrics (
                    id                          TEXT PRIMARY KEY,
                    performer_id                TEXT NOT NULL,
                    identity_embedding_json     TEXT NOT NULL,
                    identity_embedding_std_json TEXT,
                    identity_quality_mean       REAL DEFAULT 0.0,
                    shape_coefficients_json     TEXT,
                    expression_coefficients_mean_json TEXT,
                    expression_coefficients_std_json TEXT,
                    canonical_landmarks_68_json TEXT,
                    canonical_landmarks_68_cov_json TEXT,
                    canonical_landmarks_468_json TEXT,
                    facial_measurements_json    TEXT DEFAULT '{}',
                    skin_tone_lab_json          TEXT,
                    skin_texture_descriptor_json TEXT,
                    source_image_count          INTEGER DEFAULT 0,
                    pose_coverage_json          TEXT DEFAULT '{}',
                    expression_coverage_json    TEXT DEFAULT '{}',
                    updated_at                  TEXT,
                    model_versions_json         TEXT DEFAULT '{}',
                    FOREIGN KEY (performer_id) REFERENCES enrich_performers(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_enrich_facial_biometrics_performer
                    ON enrich_facial_biometrics(performer_id);

                -- Performers (extended known persons)
                CREATE TABLE IF NOT EXISTS enrich_performers (
                    id                      TEXT PRIMARY KEY,
                    name                    TEXT NOT NULL UNIQUE,
                    aliases_json            TEXT DEFAULT '[]',
                    face_embedding_json     TEXT,
                    face_embedding_quality  REAL DEFAULT 0.0,
                    measurements_json       TEXT DEFAULT '{}',
                    measurement_confidence_json TEXT DEFAULT '{}',
                    body_type               TEXT,
                    height_cm               REAL,
                    height_confidence       REAL DEFAULT 0.0,
                    weight_kg_estimate      REAL,
                    facial_biometrics_id    TEXT,
                    asset_count             INTEGER DEFAULT 0,
                    first_seen              TEXT,
                    last_seen               TEXT,
                    source_tags_json        TEXT DEFAULT '[]',
                    notes                   TEXT,
                    created_at              TEXT NOT NULL,
                    updated_at              TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_enrich_performers_name
                    ON enrich_performers(name);

                -- Performer-asset linkage
                CREATE TABLE IF NOT EXISTS enrich_performer_assets (
                    performer_id    TEXT NOT NULL,
                    asset_id        TEXT NOT NULL,
                    matched_at      TEXT NOT NULL,
                    match_method    TEXT,
                    match_score     REAL,
                    PRIMARY KEY (performer_id, asset_id),
                    FOREIGN KEY (performer_id) REFERENCES enrich_performers(id) ON DELETE CASCADE,
                    FOREIGN KEY (asset_id) REFERENCES enrich_assets(asset_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_enrich_performer_assets_asset
                    ON enrich_performer_assets(asset_id);
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def upsert(self, asset: Asset) -> None:
        """Insert or update an asset and all related enrichment tables."""
        record = asset_to_record(asset)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO enrich_assets
                (asset_id, content_sha256, file_path, file_name, file_size, mime_type,
                 status, indexed_at, analyzed_at, enriched_at, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(asset_id) DO UPDATE SET
                  content_sha256=excluded.content_sha256,
                  file_path=excluded.file_path,
                  file_name=excluded.file_name,
                  file_size=excluded.file_size,
                  mime_type=excluded.mime_type,
                  status=excluded.status,
                  indexed_at=excluded.indexed_at,
                  analyzed_at=excluded.analyzed_at,
                  enriched_at=excluded.enriched_at,
                  error_message=excluded.error_message
                """,
                (
                    record.asset_id, record.content_sha256, record.file_path,
                    record.file_name, record.file_size, record.mime_type,
                    asset.status.value if hasattr(asset.status, "value") else asset.status,
                    asset.indexed_at.isoformat() if asset.indexed_at else None,
                    asset.analyzed_at.isoformat() if asset.analyzed_at else None,
                    record.enriched_at, asset.error_message,
                ),
            )
            self._upsert_exif(conn, record)
            self._upsert_vlm(conn, record)
            self._upsert_entities(conn, record)
            self._upsert_faces(conn, record)
            self._upsert_sidecar(conn, record)
            self._upsert_tags(conn, record)

    def _upsert_exif(self, conn: sqlite3.Connection, record: DBRecord) -> None:
        conn.execute(
            """
            INSERT INTO enrich_exif (asset_id, raw_json, structured_json)
            VALUES (?, ?, ?)
            ON CONFLICT(asset_id) DO UPDATE SET
                raw_json=excluded.raw_json,
                structured_json=excluded.structured_json
            """,
            (record.asset_id, json.dumps(record.exif_raw), json.dumps(record.exif_structured)),
        )

    def _upsert_vlm(self, conn: sqlite3.Connection, record: DBRecord) -> None:
        if not record.vlm_caption and not record.vlm_description:
            return
        conn.execute(
            """
            INSERT INTO enrich_vlm
            (asset_id, model, caption, description, analyzed_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(asset_id) DO UPDATE SET
              model=excluded.model,
              caption=excluded.caption,
              description=excluded.description,
              analyzed_at=excluded.analyzed_at
            """,
            (
                record.asset_id, record.vlm_model, record.vlm_caption,
                record.vlm_description, record.enriched_at,
            ),
        )

    def _upsert_entities(self, conn: sqlite3.Connection, record: DBRecord) -> None:
        conn.execute("DELETE FROM enrich_entities WHERE asset_id = ?", (record.asset_id,))
        try:
            entities = json.loads(record.ner_entities_json)
        except (json.JSONDecodeError, TypeError):
            entities = []
        for e in entities:
            conn.execute(
                """
                INSERT INTO enrich_entities
                (id, asset_id, text, entity_type, label, confidence, bbox_json, source, start_char, end_char)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(e.get("id", "")),
                    record.asset_id,
                    e.get("text", ""),
                    e.get("entity_type", ""),
                    e.get("label"),
                    e.get("confidence"),
                    json.dumps(e.get("bbox")) if e.get("bbox") else None,
                    e.get("source", "vlm"),
                    e.get("start_char"),
                    e.get("end_char"),
                ),
            )

    def _upsert_faces(self, conn: sqlite3.Connection, record: DBRecord) -> None:
        conn.execute("DELETE FROM enrich_faces WHERE asset_id = ?", (record.asset_id,))
        try:
            faces = json.loads(record.face_embeddings_json)
        except (json.JSONDecodeError, TypeError):
            faces = []
        for f in faces:
            conn.execute(
                """
                INSERT INTO enrich_faces
                (id, asset_id, bbox_json, landmarks_json, landmarks_468_json, landmarks_3d_json,
                 embedding_json, blendshapes_json, expression_vector_json, identity_vector_json,
                 head_pose_json, eye_gaze_json, iris_landmarks_json, det_score, age, gender, model_name, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(f.get("id", "")),
                    record.asset_id,
                    json.dumps(f.get("bbox")) if f.get("bbox") else None,
                    json.dumps(f.get("landmarks")) if f.get("landmarks") else None,
                    json.dumps(f.get("landmarks_468")) if f.get("landmarks_468") else None,
                    json.dumps(f.get("landmarks_3d")) if f.get("landmarks_3d") else None,
                    json.dumps(f.get("embedding")) if f.get("embedding") else None,
                    json.dumps(f.get("blendshapes")) if f.get("blendshapes") else None,
                    json.dumps(f.get("expression_vector")) if f.get("expression_vector") else None,
                    json.dumps(f.get("identity_vector")) if f.get("identity_vector") else None,
                    json.dumps(f.get("head_pose")) if f.get("head_pose") else None,
                    json.dumps(f.get("eye_gaze")) if f.get("eye_gaze") else None,
                    json.dumps(f.get("iris_landmarks")) if f.get("iris_landmarks") else None,
                    f.get("det_score"),
                    f.get("age"),
                    f.get("gender"),
                    f.get("model_name", "buffalo_l"),
                    f.get("detected_at"),
                ),
            )

    def _upsert_sidecar(self, conn: sqlite3.Connection, record: DBRecord) -> None:
        conn.execute(
            """
            INSERT INTO enrich_sidecars (asset_id, sidecar_path, status)
            VALUES (?, ?, ?)
            ON CONFLICT(asset_id) DO UPDATE SET
              sidecar_path=excluded.sidecar_path,
              status=excluded.status
            """,
            (record.asset_id, None, record.sidecar_status),
        )

    def _upsert_tags(self, conn: sqlite3.Connection, record: DBRecord) -> None:
        conn.execute("DELETE FROM enrich_tags WHERE asset_id = ?", (record.asset_id,))
        try:
            tags = json.loads(record.tags_json)
        except (json.JSONDecodeError, TypeError):
            tags = []
        for tag in tags:
            conn.execute(
                """
                INSERT INTO enrich_tags
                (id, asset_id, tag_name, source)
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    record.asset_id,
                    tag,
                    "manual",
                ),
            )

    def get(self, asset_id: str) -> Optional[Asset]:
        """Retrieve an asset by ID with all enrichment data."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enrich_assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            if not row:
                return None
            asset = Asset(
                id=UUID(row["asset_id"]) if row["asset_id"] else None,
                path=row["file_path"],
                filename=row["file_name"],
                content_sha256=row["content_sha256"],
                file_size=row["file_size"],
                mime_type=row["mime_type"],
                status=AssetStatus(row["status"]) if row["status"] else AssetStatus.PENDING,
                indexed_at=_parse_dt(row["indexed_at"]),
                analyzed_at=_parse_dt(row["analyzed_at"]),
                enriched_at=_parse_dt(row["enriched_at"]),
                error_message=row["error_message"],
            )
            # Load exif
            exif_row = conn.execute(
                "SELECT raw_json, structured_json FROM enrich_exif WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            if exif_row:
                raw = json.loads(exif_row["raw_json"]) if exif_row["raw_json"] else {}
                structured = json.loads(exif_row["structured_json"]) if exif_row["structured_json"] else {}
                from .models import ExifMetadata
                if structured:
                    asset.exif = ExifMetadata(**structured, raw=raw)
                else:
                    asset.exif = ExifMetadata(raw=raw)

            # Load vlm
            vlm_row = conn.execute(
                "SELECT model, caption, description FROM enrich_vlm WHERE asset_id = ?",
                (asset_id,),
            ).fetchone()
            if vlm_row:
                asset.vlm = VLMResult(
                    model=vlm_row["model"],
                    caption=vlm_row["caption"],
                    detailed_description=vlm_row["description"],
                )

            # Load entities
            ent_rows = conn.execute(
                "SELECT * FROM enrich_entities WHERE asset_id = ?", (asset_id,)
            ).fetchall()
            for er in ent_rows:
                asset.ner_entities.append(
                    NEREntity(
                        text=er["text"],
                        entity_type=EntityType(er["entity_type"]),
                        label=er["label"],
                        confidence=er["confidence"],
                        source=er["source"],
                    )
                )

            # Load faces
            face_rows = conn.execute(
                "SELECT * FROM enrich_faces WHERE asset_id = ?", (asset_id,)
            ).fetchall()
            for fr in face_rows:
                from .models import FaceEmbedding
                asset.face_embeddings.append(
                    FaceEmbedding(
                        id=UUID(fr["id"]) if fr["id"] else None,
                        asset_id=UUID(fr["asset_id"]) if fr["asset_id"] else None,
                        bbox=json.loads(fr["bbox_json"]) if fr["bbox_json"] else None,
                        landmarks=json.loads(fr["landmarks_json"]) if fr["landmarks_json"] else None,
                        embedding=json.loads(fr["embedding_json"]) if fr["embedding_json"] else None,
                        det_score=fr["det_score"],
                        age=fr["age"],
                        gender=fr["gender"],
                        model_name=fr["model_name"],
                        detected_at=_parse_dt(fr["detected_at"]),
                    )
                )

            # Load tags
            tag_rows = conn.execute(
                "SELECT * FROM enrich_tags WHERE asset_id = ?", (asset_id,)
            ).fetchall()
            for tr in tag_rows:
                asset.ner_entities.append(
                    NEREntity(
                        id=UUID(tr["id"]) if tr["id"] else None,
                        text=tr["tag_name"],
                        entity_type=EntityType.FACE,
                        label=tr["tag_name"],
                        confidence=1.0,
                        source=tr["source"],
                    )
                )

            # Load sidecar
            sc_row = conn.execute(
                "SELECT status FROM enrich_sidecars WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            if sc_row and sc_row["status"]:
                asset.sidecar_status = SidecarStatus(sc_row["status"])

            return asset

    def list_all(self, limit: int = 100, offset: int = 0) -> List[Asset]:
        """List all indexed assets."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM enrich_assets LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
            return [self._row_to_asset(r, conn) for r in rows]

    def _row_to_asset(self, row: sqlite3.Row, conn: sqlite3.Connection) -> Asset:
        asset = Asset(
            id=UUID(row["asset_id"]) if row["asset_id"] else None,
            path=row["file_path"],
            filename=row["file_name"],
            content_sha256=row["content_sha256"],
            file_size=row["file_size"],
            mime_type=row["mime_type"],
            status=AssetStatus(row["status"]) if row["status"] else AssetStatus.PENDING,
            indexed_at=_parse_dt(row["indexed_at"]),
            analyzed_at=_parse_dt(row["analyzed_at"]),
            enriched_at=_parse_dt(row["enriched_at"]),
            error_message=row["error_message"],
        )
        return asset

    def delete(self, asset_id: str) -> bool:
        """Delete an asset and all related enrichment data."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM enrich_assets WHERE asset_id = ?", (asset_id,)
            )
            return cur.rowcount > 0

    # Face Recognition Methods

    def create_person(self, name: str, embedding: List[float]) -> str:
        """Create a known person with their face embedding.
        
        Args:
            name: Person's name (must be unique)
            embedding: Face embedding vector
            
        Returns:
            person_id: UUID of the created person
        """
        person_id = str(uuid.uuid4())
        embedding_json = json.dumps(embedding)
        created_at = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO enrich_persons (id, name, embedding_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (person_id, name, embedding_json, created_at),
            )
        return person_id

    def get_person(self, person_id: str) -> Optional[Dict]:
        """Get a person by their ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enrich_persons WHERE id = ?", (person_id,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def get_person_by_name(self, name: str) -> Optional[Dict]:
        """Get a person by their name."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enrich_persons WHERE name = ?", (name,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def list_persons(self) -> List[Dict]:
        """List all known persons."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM enrich_persons ORDER BY created_at DESC").fetchall()
            return [dict(row) for row in rows]

    def _cosine_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        dot = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = math.sqrt(sum(a * a for a in emb1))
        norm2 = math.sqrt(sum(b * b for b in emb2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def match_face_to_person(
        self, face_embedding: List[float], threshold: float = 0.6
    ) -> Optional[Tuple[str, float]]:
        """Match a face embedding against all known persons using cosine similarity.
        
        Args:
            face_embedding: Face embedding vector to match
            threshold: Minimum cosine similarity threshold (default 0.6)
            
        Returns:
            Tuple of (person_id, similarity) if match found, None otherwise
        """
        persons = self.list_persons()
        best_match = None
        best_similarity = 0.0
        
        for person in persons:
            person_embedding = json.loads(person["embedding_json"])
            similarity = self._cosine_similarity(face_embedding, person_embedding)
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = person["id"]
        
        if best_match:
            return (best_match, best_similarity)
        return None

    def match_faces_in_asset(
        self, asset_id: str, threshold: float = 0.6
    ) -> List[Dict]:
        """Match all faces in an asset to known persons.
        
        Args:
            asset_id: Asset ID to match faces from
            threshold: Minimum cosine similarity threshold
            
        Returns:
            List of match results with face_id, person_id, person_name, similarity
        """
        with self._connect() as conn:
            face_rows = conn.execute(
                "SELECT * FROM enrich_faces WHERE asset_id = ?", (asset_id,)
            ).fetchall()
        
        matches = []
        for face_row in face_rows:
            face_embedding = json.loads(face_row["embedding_json"])
            match = self.match_face_to_person(face_embedding, threshold)
            if match:
                person_id, similarity = match
                person = self.get_person(person_id)
                matches.append({
                    "face_id": face_row["id"],
                    "person_id": person_id,
                    "person_name": person["name"] if person else None,
                    "similarity": similarity,
                })
        return matches

    def link_face_to_person(
        self, face_id: str, person_id: str, similarity: float
    ) -> None:
        """Store a face-to-person match in enrich_face_matches.
        
        Args:
            face_id: Face embedding ID
            person_id: Known person ID
            similarity: Cosine similarity score
        """
        match_id = str(uuid.uuid4())
        matched_at = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO enrich_face_matches (id, face_id, person_id, similarity, matched_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (match_id, face_id, person_id, similarity, matched_at),
            )

    def get_faces_by_person(self, person_id: str) -> List[Dict]:
        """Get all faces matched to a specific person.
        
        Args:
            person_id: Known person ID
            
        Returns:
            List of face records with match info
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT f.*, m.similarity, m.matched_at, p.name as person_name
                FROM enrich_face_matches m
                JOIN enrich_faces f ON m.face_id = f.id
                JOIN enrich_persons p ON m.person_id = p.id
                WHERE m.person_id = ?
                ORDER BY m.matched_at DESC
                """,
                (person_id,)
            )
            return [dict(row) for row in rows]

    # Body pose methods
    def upsert_body_pose(self, asset_id: str, pose_data: Dict) -> str:
        """Insert or update body pose for an asset."""
        pose_id = str(uuid.uuid4())
        detected_at = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO enrich_body_poses
                (id, asset_id, landmark_set, landmarks_2d_json, landmarks_3d_json, bbox_json,
                 det_score, model_name, detected_at, raw_response_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  landmark_set=excluded.landmark_set,
                  landmarks_2d_json=excluded.landmarks_2d_json,
                  landmarks_3d_json=excluded.landmarks_3d_json,
                  bbox_json=excluded.bbox_json,
                  det_score=excluded.det_score,
                  model_name=excluded.model_name,
                  detected_at=excluded.detected_at,
                  raw_response_json=excluded.raw_response_json
                """,
                (
                    pose_id, asset_id, pose_data.get("landmark_set"),
                    json.dumps(pose_data.get("landmarks_2d")),
                    json.dumps(pose_data.get("landmarks_3d")) if pose_data.get("landmarks_3d") else None,
                    json.dumps(pose_data.get("bbox")) if pose_data.get("bbox") else None,
                    pose_data.get("det_score"),
                    pose_data.get("model_name"),
                    detected_at,
                    json.dumps(pose_data.get("raw_response")) if pose_data.get("raw_response") else None,
                ),
            )
        return pose_id

    def get_body_poses(self, asset_id: str) -> List[Dict]:
        """Get all body poses for an asset."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM enrich_body_poses WHERE asset_id = ?", (asset_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    # Body measurement methods
    def upsert_body_measurement(self, measurement: Dict) -> str:
        """Insert or update a body measurement."""
        meas_id = str(uuid.uuid4())
        computed_at = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO enrich_body_measurements
                (id, asset_id, performer_id, measurement_type, value, unit, confidence,
                 method, reference_scale, landmarks_used_json, computed_at, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    meas_id,
                    measurement.get("asset_id"),
                    measurement.get("performer_id"),
                    measurement.get("measurement_type"),
                    measurement.get("value"),
                    measurement.get("unit", "cm"),
                    measurement.get("confidence", 0.0),
                    measurement.get("method", "landmark_2d"),
                    measurement.get("reference_scale"),
                    json.dumps(measurement.get("landmarks_used")) if measurement.get("landmarks_used") else None,
                    computed_at,
                    measurement.get("notes"),
                ),
            )
        return meas_id

    def get_body_measurements(self, asset_id: str = None, performer_id: str = None, measurement_type: str = None) -> List[Dict]:
        """Get body measurements with optional filters."""
        query = "SELECT * FROM enrich_body_measurements WHERE 1=1"
        params = []
        if asset_id:
            query += " AND asset_id = ?"
            params.append(asset_id)
        if performer_id:
            query += " AND performer_id = ?"
            params.append(performer_id)
        if measurement_type:
            query += " AND measurement_type = ?"
            params.append(measurement_type)
        query += " ORDER BY computed_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                r = dict(row)
                # Parse JSON fields
                if r.get("landmarks_used_json"):
                    import json
                    r["landmarks_used_json"] = json.loads(r["landmarks_used_json"])
                results.append(r)
            return results

    # Performer methods
    def create_performer(self, name: str, aliases: List[str] = None, face_embedding: List[float] = None,
                         face_embedding_quality: float = 0.0, notes: str = None) -> str:
        """Create a new performer profile."""
        performer_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        aliases_json = json.dumps(aliases or [])
        face_embedding_json = json.dumps(face_embedding) if face_embedding else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO enrich_performers
                (id, name, aliases_json, face_embedding_json, face_embedding_quality,
                 created_at, updated_at, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (performer_id, name, aliases_json, face_embedding_json, face_embedding_quality,
                 now, now, notes),
            )
        return performer_id

    def get_performer(self, performer_id: str) -> Optional[Dict]:
        """Get performer by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enrich_performers WHERE id = ?", (performer_id,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def get_performer_by_name(self, name: str) -> Optional[Dict]:
        """Get performer by name."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enrich_performers WHERE name = ?", (name,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def list_performers(self) -> List[Dict]:
        """List all performers."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM enrich_performers ORDER BY created_at DESC").fetchall()
            return [dict(row) for row in rows]

    def update_performer(self, performer_id: str, updates: Dict) -> bool:
        """Update performer fields."""
        if not updates:
            return False
        updates["updated_at"] = datetime.utcnow().isoformat()
        set_clause = ", ".join(f"{k}=?" for k in updates.keys())
        values = list(updates.values()) + [performer_id]
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE enrich_performers SET {set_clause} WHERE id = ?", values
            )
            return cursor.rowcount > 0

    def link_performer_to_asset(self, performer_id: str, asset_id: str, match_method: str = "face",
                                 match_score: float = None) -> None:
        """Link a performer to an asset."""
        matched_at = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO enrich_performer_assets
                (performer_id, asset_id, matched_at, match_method, match_score)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(performer_id, asset_id) DO UPDATE SET
                  matched_at=excluded.matched_at,
                  match_method=excluded.match_method,
                  match_score=excluded.match_score
                """,
                (performer_id, asset_id, matched_at, match_method, match_score),
            )

    def get_performer_assets(self, performer_id: str) -> List[Dict]:
        """Get all assets linked to a performer."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT a.*, pa.matched_at, pa.match_method, pa.match_score
                FROM enrich_assets a
                JOIN enrich_performer_assets pa ON a.asset_id = pa.asset_id
                WHERE pa.performer_id = ?
                ORDER BY pa.matched_at DESC
                """,
                (performer_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_asset_performers(self, asset_id: str) -> List[Dict]:
        """Get all performers linked to an asset."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT p.*, pa.matched_at, pa.match_method, pa.match_score
                FROM enrich_performers p
                JOIN enrich_performer_assets pa ON p.id = pa.performer_id
                WHERE pa.asset_id = ?
                ORDER BY pa.matched_at DESC
                """,
                (asset_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    # Facial biometrics methods
    def upsert_facial_biometrics(self, biometrics: Dict) -> str:
        """Insert or update facial biometrics for a performer."""
        bio_id = str(uuid.uuid4())
        updated_at = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO enrich_facial_biometrics
                (id, performer_id, identity_embedding_json, identity_embedding_std_json,
                 identity_quality_mean, shape_coefficients_json,
                 expression_coefficients_mean_json, expression_coefficients_std_json,
                 canonical_landmarks_68_json, canonical_landmarks_68_cov_json,
                 canonical_landmarks_468_json, facial_measurements_json,
                 skin_tone_lab_json, skin_texture_descriptor_json,
                 source_image_count, pose_coverage_json, expression_coverage_json,
                 updated_at, model_versions_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  identity_embedding_json=excluded.identity_embedding_json,
                  identity_embedding_std_json=excluded.identity_embedding_std_json,
                  identity_quality_mean=excluded.identity_quality_mean,
                  shape_coefficients_json=excluded.shape_coefficients_json,
                  expression_coefficients_mean_json=excluded.expression_coefficients_mean_json,
                  expression_coefficients_std_json=excluded.expression_coefficients_std_json,
                  canonical_landmarks_68_json=excluded.canonical_landmarks_68_json,
                  canonical_landmarks_68_cov_json=excluded.canonical_landmarks_68_cov_json,
                  canonical_landmarks_468_json=excluded.canonical_landmarks_468_json,
                  facial_measurements_json=excluded.facial_measurements_json,
                  skin_tone_lab_json=excluded.skin_tone_lab_json,
                  skin_texture_descriptor_json=excluded.skin_texture_descriptor_json,
                  source_image_count=excluded.source_image_count,
                  pose_coverage_json=excluded.pose_coverage_json,
                  expression_coverage_json=excluded.expression_coverage_json,
                  updated_at=excluded.updated_at,
                  model_versions_json=excluded.model_versions_json
                """,
                (
                    bio_id,
                    biometrics.get("performer_id"),
                    json.dumps(biometrics.get("identity_embedding")),
                    json.dumps(biometrics.get("identity_embedding_std")) if biometrics.get("identity_embedding_std") else None,
                    biometrics.get("identity_quality_mean", 0.0),
                    json.dumps(biometrics.get("shape_coefficients")) if biometrics.get("shape_coefficients") else None,
                    json.dumps(biometrics.get("expression_coefficients_mean")) if biometrics.get("expression_coefficients_mean") else None,
                    json.dumps(biometrics.get("expression_coefficients_std")) if biometrics.get("expression_coefficients_std") else None,
                    json.dumps(biometrics.get("canonical_landmarks_68")) if biometrics.get("canonical_landmarks_68") else None,
                    json.dumps(biometrics.get("canonical_landmarks_68_cov")) if biometrics.get("canonical_landmarks_68_cov") else None,
                    json.dumps(biometrics.get("canonical_landmarks_468")) if biometrics.get("canonical_landmarks_468") else None,
                    json.dumps(biometrics.get("facial_measurements", {})),
                    json.dumps(biometrics.get("skin_tone_lab")) if biometrics.get("skin_tone_lab") else None,
                    json.dumps(biometrics.get("skin_texture_descriptor")) if biometrics.get("skin_texture_descriptor") else None,
                    biometrics.get("source_image_count", 0),
                    json.dumps(biometrics.get("pose_coverage", {})),
                    json.dumps(biometrics.get("expression_coverage", {})),
                    updated_at,
                    json.dumps(biometrics.get("model_versions", {})),
                ),
            )
        return bio_id

    def get_facial_biometrics(self, performer_id: str) -> Optional[Dict]:
        """Get facial biometrics for a performer."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enrich_facial_biometrics WHERE performer_id = ?", (performer_id,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def get_faces_for_asset(self, asset_id: str) -> List[Dict]:
        """Get all face records for an asset as dicts."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM enrich_faces WHERE asset_id = ?", (asset_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_asset(self, asset_id: str) -> Optional[Asset]:
        """Retrieve an asset by ID (alias for get)."""
        return self.get(asset_id)

    # Tag management methods
    def add_tag(self, asset_id: str, tag_name: str, source: str = "manual") -> bool:
        """Add a tag to an asset."""
        with self._connect() as conn:
            # Check if asset exists
            asset = conn.execute(
                "SELECT 1 FROM enrich_assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            if not asset:
                return False
            
            # Check if tag already exists
            existing = conn.execute(
                "SELECT 1 FROM enrich_tags WHERE asset_id = ? AND tag_name = ?", 
                (asset_id, tag_name)
            ).fetchone()
            if existing:
                return False
            
            tag_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO enrich_tags (id, asset_id, tag_name, source)
                VALUES (?, ?, ?, ?)
                """,
                (tag_id, asset_id, tag_name, source)
            )
            return True

    def remove_tag(self, asset_id: str, tag_name: str) -> bool:
        """Remove a tag from an asset."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM enrich_tags WHERE asset_id = ? AND tag_name = ?",
                (asset_id, tag_name)
            )
            return cursor.rowcount > 0

    def get_tags(self, asset_id: str) -> List[str]:
        """Get all tags for an asset."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT tag_name FROM enrich_tags WHERE asset_id = ? ORDER BY tag_name",
                (asset_id,)
            ).fetchall()
            return [row["tag_name"] for row in rows]

    def find_assets_by_tag(self, tag_name: str, limit: int = 100, offset: int = 0) -> List[Asset]:
        """Find all assets with a specific tag."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT a.* FROM enrich_assets a
                JOIN enrich_tags t ON a.asset_id = t.asset_id
                WHERE t.tag_name = ?
                ORDER BY a.enriched_at DESC
                LIMIT ? OFFSET ?
                """,
                (tag_name, limit, offset)
            ).fetchall()
            return [self._row_to_asset(r, conn) for r in rows]

    def import_tags_from_file(self, file_path: str) -> int:
        """Bulk import tags from a file.
        
        File format (TSV/CSV): asset_id\ttag1,tag2,tag3 or path\ttag1,tag2
        
        Returns:
            Number of tags successfully imported
        """
        import csv
        count = 0
        with open(file_path, 'r') as f:
            # Force tab delimiter for TSV files (more reliable than sniffer)
            delimiter = '\t'
            f.seek(0)
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                if len(row) < 2:
                    continue
                identifier = row[0].strip()
                tags_str = row[1].strip()
                
                # Find asset by ID or path
                asset_id = None
                if '-' in identifier and len(identifier) > 30:  # Likely a UUID
                    with self._connect() as conn:
                        asset_row = conn.execute(
                            "SELECT asset_id FROM enrich_assets WHERE asset_id = ?", 
                            (identifier,)
                        ).fetchone()
                        if asset_row:
                            asset_id = asset_row["asset_id"]
                else:
                    # Try by path/filename
                    with self._connect() as conn:
                        asset_row = conn.execute(
                            "SELECT asset_id FROM enrich_assets WHERE file_path LIKE ? OR file_name = ?", 
                            (f"%{identifier}%", identifier)
                        ).fetchone()
                        if asset_row:
                            asset_id = asset_row["asset_id"]
                
                if not asset_id:
                    continue
                
                tags = [t.strip() for t in tags_str.split(',') if t.strip()]
                for tag in tags:
                    if self.add_tag(asset_id, tag, source="file"):
                        count += 1
        return count