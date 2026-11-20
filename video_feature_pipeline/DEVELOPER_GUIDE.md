# Developer Guide

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│   Ingestion     │────▶│  Feature Extraction  │────▶│  Storage & Index │
│   (yt-dlp)      │     │  (MediaPipe,         │     │  (ChromaDB +     │
│   File Watcher  │     │   InsightFace, VLM)  │     │   SQLite +       │
└─────────────────┘     └──────────────────────┘     │   Neo4j)         │
       │                         │                    └──────────────────┘
       ▼                         ▼                           │
  Local video files         Feature vectors:              Vector DB:
  organized by source       - 3D skeletal landmarks       - Face embeddings
                            - Euclidean distances         - Semantic summaries
                            - Face embeddings             - Metadata
                            - Semantic descriptions       Relational DB:
                                                          - File paths
                                                          - Spatial coordinates
                                                          - Processing status
                                                          Graph DB:
                                                          - Identity nodes
                                                          - Co-occurrence edges
```

## Core Modules

### 1. Ingestion Layer (`src/video_feature_pipeline/ingestion/`)

**Downloader** (`downloader.py`)
- Wrapper around `yt-dlp` Python API
- Downloads to structured path: `/data/videos/{source}/{video_id}.{ext}`
- Progress hooks with tqdm
- Manifest generation (JSON)
- Source detection: youtube, tiktok, twitter, instagram

**File Watcher** (`watcher.py`)
- `watchdog` observer on `/data/videos/`
- Debounced event handling (2s default)
- Filters video extensions: mp4, mkv, mov, avi, webm, flv
- Async processor callback

**SocQ Client** (`socq_client.py`)
- Async HTTP client for SocQ API
- Profile search, video enumeration, profile info
- Configurable via `harvest.socq_api_key`

### 2. Feature Extraction Layer (`src/video_feature_pipeline/extraction/`)

**Spatial Extractor** (`spatial.py`)
- MediaPipe Tasks API `HolisticLandmarker`
- 33 pose landmarks + 478 face landmarks
- 10 limb measurements + 4 face distances
- **7 invariant ratios** (camera-independent):
  - `forearm_to_upper_arm_L/R`
  - `shin_to_thigh_L/R`
  - `torso_to_shoulder`
  - `arm_span_to_height`
  - `leg_length_to_height`
- Frame sampling at configurable FPS

**Identity Extractor** (`identity.py`)
- InsightFace `FaceAnalysis` (buffalo_l/s)
- 512-d normalized embeddings
- Per-frame face detection at configurable FPS
- DBSCAN clustering (eps=0.4, min_samples=2)
- Output: `IdentityCluster` with centroid, count, avg_confidence

**Semantic Extractor** (`semantic.py`)
- Ollama VLM (moondream2, llava:7b)
- Keyframe extraction at 0.5 FPS (max 20)
- Prompt: setting, people count, activity, visual attributes
- SentenceTransformer `all-MiniLM-L6-v2` for 384-d semantic embedding

### 3. Storage Layer (`src/video_feature_pipeline/storage/`)

**Vector Database** (`vector_db.py`)
- ChromaDB persistent client
- Collections: `face_embeddings` (512-d), `semantic_embeddings` (384-d)
- Cosine similarity (HNSW)
- Batch insert methods: `add_face_embeddings_batch`, `add_semantic_embeddings_batch`
- Search with threshold filtering

**Graph Database** (`graph_db.py`)
- Neo4j driver (optional)
- Nodes: `Identity {id, video_id, cluster_id}`
- Edges: `CO_OCCURS_WITH {video_id, frame_idx, weight, timestamp}`
- Queries: partners, network (BFS), GraphML export

**SQLite Database** (`core/database.py`)
- Tables: `videos`, `spatial_features`, `identities`, `co_occurrences`
- Indexes on status, source, video_id, embedding_id
- JSON columns for flexible feature storage
- Physical signature search via `json_extract`

### 4. Core Pipeline (`src/video_feature_pipeline/core/`)

**Processor** (`processor.py`)
- Orchestrates all extractors
- **Adaptive frame rate**: base FPS → high-confidence re-analysis
- Harvest trigger: high-confidence new identities → SocQ scrape
- Syncs to SQLite, ChromaDB, Neo4j
- Sync/async processing methods

**Config** (`config.py`)
- Pydantic Settings v2
- YAML + environment variables
- Nested configs: paths, ingestion, extraction, search, harvest, logging

**Logging** (`logging.py`)
- Structlog with JSON/console output
- Context variables, log levels, file rotation

## Data Flow

```
process_video(video_path)
    │
    ├─▶ _get_video_metadata() → duration, fps, resolution
    │
    ├▶ upsert_video(status="processing")
    │
    ├▶ SpatialExtractor.extract_from_video()
    │     ├▶ MediaPipe Holistic per frame
    │     ├▶ Compute limb lengths + face distances
    │     ├▶ Compute 7 invariant ratios
    │     └▶ Aggregate stats (mean/std) → SpatialFeatures
    │
    ├▶ _save_spatial_to_db() → spatial_features table
    │
    ├▶ IdentityExtractor.extract_from_video()
    │     ├▶ InsightFace per frame
    │     ├▶ Collect embeddings + metadata
    │     ├▶ DBSCAN clustering
    │     └▶ IdentityCluster list
    │
    ├▶ _save_identities_to_db()
    │     ├▶ ChromaDB face_embeddings
    │     ├▶ SQLite identities
    │     ├▶ Co-occurrence detection (frame ±5)
    │     │     ├▶ SQLite co_occurrences
    │     │     └▶ Neo4j CO_OCCURS_WITH
    │     └▶ Neo4j Identity nodes
    │
    ├▶ SemanticExtractor.extract_from_video()
    │     ├▶ Keyframes at 0.5 FPS
    │     ├▶ Ollama VLM per keyframe
    │     ├▶ Concatenate descriptions
    │     ├▶ SentenceTransformer embedding
    │     └▶ SemanticFeatures
    │
    ├▶ _save_semantic_to_db() → ChromaDB semantic_embeddings
    │
    ├▶ Harvest trigger (if enabled)
    │     ├▶ Check cluster.avg_confidence >= min_confidence
    │     ├▶ Verify not already tracked
    │     └▶ SocQClient.get_profile_videos() → download
    │
    └▶ upsert_video(status="done")
```

## Configuration

### config.yaml Structure

```yaml
paths:
  data_root: "./data"
  videos_dir: "./data/videos"
  db_dir: "./data/chroma_db"
  sqlite_path: "./data/pipeline.db"

ingestion:
  ytdlp_format: "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
  max_concurrent: 3
  download_timeout: 300

extraction:
  frame_sample_rate: 1.0
  keyframe_rate: 0.5
  mediapipe_complexity: 1
  insightface_model: "buffalo_l"
  insightface_det_size: [640, 640]
  vlm_model: "moondream2"
  vlm_host: "http://localhost:11434"
  vlm_timeout: 30
  max_keyframes: 20
  # Adaptive frame rate
  adaptive_frame_rate: false
  base_frame_rate: 0.5
  high_confidence_frame_rate: 2.0
  identity_confidence_threshold: 0.8

search:
  default_top_k: 10
  face_similarity_threshold: 0.6
  semantic_similarity_threshold: 0.5
  spatial_weight: 0.3
  face_weight: 0.5
  semantic_weight: 0.2

harvest:
  enabled: false
  socq_api_url: "https://api.socq.io"
  socq_api_key: ""
  min_confidence: 0.85
  max_concurrent: 3
  auto_scrape_on_new_identity: true

logging:
  level: "INFO"
  format: "json"
  file: "./data/pipeline.log"
```

### Environment Variables

All settings support `VFP_<SECTION>_<KEY>` format:
```bash
VFP_EXTRACTION_FRAME_SAMPLE_RATE=2.0
VFP_HARVEST_ENABLED=true
VFP_HARVEST_SOCQ_API_KEY=your_key
VFP_LOGGING_LEVEL=DEBUG
```

## Extending the Pipeline

### Adding a New Extractor

1. Create `src/video_feature_pipeline/extraction/new_extractor.py`:
```python
from dataclasses import dataclass
import numpy as np

@dataclass
class NewFeatures:
    video_id: str
    custom_vector: np.ndarray
    metadata: dict

class NewExtractor:
    def __init__(self, config_param: int = 1):
        pass
    
    def extract_from_video(self, video_path: Path, video_id: str) -> NewFeatures:
        # Implementation
        return NewFeatures(...)
```

2. Register in `extraction/__init__.py`
3. Add to `Processor.__init__` and `process_video`
4. Add storage method in `core/processor.py` and `storage/vector_db.py`

### Adding a New Search Type

1. Add query parameters to `cli/main.py` search command
2. Implement search method in `storage/vector_db.py` or `core/database.py`
3. Handle result formatting in CLI

## Testing

```bash
# Run pytest
cd video_feature_pipeline
./venv/bin/pytest tests/ -v

# Linting
./venv/bin/ruff check src/
./venv/bin/mypy src/

# Format
./venv/bin/ruff format src/
```

## Performance Tuning

### ChromaDB Batch Operations
```python
# Use batch methods for bulk inserts
vector_db.add_face_embeddings_batch(
    embeddings=emb_list,
    video_ids=vid_list,
    cluster_ids=cid_list,
    confidences=conf_list,
    frame_indices=frame_list,
    timestamps=ts_list,
)
```

### Adaptive Frame Rate
```yaml
extraction:
  adaptive_frame_rate: true
  base_frame_rate: 0.5          # Initial scan
  high_confidence_frame_rate: 2.0  # Detailed analysis
  identity_confidence_threshold: 0.8
```

### ONNX Quantization
```bash
# Dynamic quantization (recommended)
python scripts/quantize_onnx.py --model buffalo_l --output ./models/quantized

# QDQ format (best for ONNX Runtime)
python scripts/quantize_onnx.py --model buffalo_l --qdq --output ./models/quantized
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| MediaPipe model not found | Runs auto-download on first use; check `~/.cache/mediapipe/` |
| InsightFace slow | Run quantization script; use `buffalo_s` for speed |
| Ollama connection refused | Start Ollama: `ollama serve`; pull model: `ollama pull moondream2` |
| Neo4j unavailable | Graph features disabled automatically; check `neo4j_uri` config |
| yt-dlp fails | Update: `pip install -U yt-dlp`; install JS runtime for YouTube |
| Memory OOM | Reduce `frame_sample_rate`, `max_keyframes`; use batch methods |

## Deployment Checklist

- [ ] Configure `config.yaml` for environment
- [ ] Set `VFP_HARVEST_SOCQ_API_KEY` if using harvest
- [ ] Start Ollama for VLM: `ollama serve && ollama pull moondream2`
- [ ] Start Neo4j for graph features (optional)
- [ ] Run quantization: `python scripts/quantize_onnx.py --model buffalo_l --qdq`
- [ ] Set up systemd/service for `pipeline watch --input /data/videos`
- [ ] Configure log rotation for `./data/pipeline.log`