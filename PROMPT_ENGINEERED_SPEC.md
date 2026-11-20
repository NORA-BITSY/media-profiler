# Automated Video Feature Extraction & Vector Indexing Pipeline
## Prompt-Engineered Specification for CLI Agent Execution

---

### Project Overview
**Title:** Automated Video Feature Extraction & Vector Indexing Pipeline  
**Objective:** Build a Python-based CLI application that ingests video files from local storage, extracts spatial and visual feature vectors using local open-source computer vision models, and indexes them in a vector database for similarity search and retrieval.

**Constraint:** Use ONLY local, open-source libraries. No cloud APIs, no external services. All processing runs offline on local hardware.

---

### Technical Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│   Ingestion     │────▶│  Feature Extraction  │────▶│  Storage & Index │
│   (yt-dlp)      │     │  (MediaPipe,         │     │  (ChromaDB +     │
│   File Watcher  │     │   InsightFace, VLM)  │     │   SQLite)        │
└─────────────────┘     └──────────────────────┘     └──────────────────┘
        │                         │                          │
        ▼                         ▼                          ▼
  Local video files         Feature vectors:              Vector DB:
  organized by source       - 3D skeletal landmarks       - Face embeddings
                            - Euclidean distances         - Semantic summaries
                            - Face embeddings             - Metadata
                            - Semantic descriptions       Relational DB:
                                                          - File paths
                                                          - Spatial coordinates
                                                          - Processing status
```

---

### Component Specifications

#### 1. Ingestion Layer
**Module:** `src/ingestion/`

| Component | Specification |
|-----------|---------------|
| **Video Downloader** | Wrapper around `yt-dlp` Python API. Downloads from URL to structured local path: `/data/videos/{source}/{video_id}.{ext}` |
| **File Watcher** | `watchdog` observer monitoring `/data/videos/` for new files. Triggers processing pipeline on `on_created` event. |
| **Supported Sources** | YouTube, TikTok, Twitter/X, Instagram, generic HTTP(S) video URLs |
| **Output** | JSON manifest: `{video_id, source, local_path, metadata, download_timestamp}` |

**CLI Commands:**
```bash
# Download single URL
pipeline ingest --url "https://..." --source youtube

# Batch download from file (one URL per line)
pipeline ingest --batch urls.txt

# Start file watcher daemon
pipeline watch --input-dir /data/videos
```

#### 2. Feature Extraction Layer
**Module:** `src/extraction/`

##### 2.1 Spatial Coordinates (MediaPipe Holistic)
- **Model:** `mediapipe.tasks.vision.HolisticLandmarker` (or `mp.solutions.holistic`)
- **Output per frame:** 33 pose landmarks (x, y, z, visibility), 468 face landmarks, 21 hand landmarks each
- **Derived Features:** Euclidean distances between key landmark pairs:
  - Shoulder width (landmark 11 ↔ 12)
  - Torso height (mid-shoulder ↔ mid-hip)
  - Limb lengths (shoulder ↔ elbow ↔ wrist, hip ↔ knee ↔ ankle)
  - Inter-ocular distance (face landmarks 33 ↔ 263)
- **Aggregation:** Per-video statistics (mean, std, min, max) of each distance → CSV row per video

##### 2.2 Identity Embeddings (InsightFace)
- **Model:** `insightface.app.FaceAnalysis` with `buffalo_l` or `buffalo_s` model pack
- **Output per face:** 512-d normalized embedding vector
- **Processing:** Sample frames at 1 FPS (configurable). Detect faces, extract embeddings.
- **Deduplication:** Cluster embeddings per video (DBSCAN, eps=0.4) to get unique identities
- **Storage:** One vector per unique identity per video in ChromaDB

##### 2.3 Semantic Labeling (Local VLM via Ollama)
- **Backend:** Ollama serving `moondream2` (1.6B params) or `llava:7b`
- **Input:** Keyframes extracted at 0.5 FPS (middle frame of each 2-second window)
- **Prompt Template:**
  ```
  Describe this video frame in one sentence. Focus on: 
  - Setting/environment
  - Number of people visible
  - Primary activity/action
  - Notable visual attributes (clothing colors, objects)
  ```
- **Output:** Concatenated summaries → single text description per video
- **Embedding:** Sentence transformer (`all-MiniLM-L6-v2`) for semantic vector storage

#### 3. Storage & Indexing Layer
**Module:** `src/storage/`

##### 3.1 Vector Database (ChromaDB)
- **Collections:**
  - `face_embeddings`: 512-d vectors, metadata `{video_id, face_cluster_id, timestamp}`
  - `semantic_embeddings`: 384-d vectors, metadata `{video_id, description}`

##### 3.2 Relational Database (SQLite)
**Schema:**
```sql
CREATE TABLE videos (
    video_id TEXT PRIMARY KEY,
    source TEXT,
    local_path TEXT NOT NULL,
    duration REAL,
    fps REAL,
    width INTEGER,
    height INTEGER,
    download_ts TIMESTAMP,
    process_ts TIMESTAMP,
    status TEXT DEFAULT 'pending'  -- pending, processing, done, failed
);

CREATE TABLE spatial_features (
    video_id TEXT PRIMARY KEY REFERENCES videos(video_id),
    shoulder_width_mean REAL,
    shoulder_width_std REAL,
    torso_height_mean REAL,
    limb_lengths_json TEXT,  -- JSON of all derived distances
    face_landmark_distances_json TEXT
);

CREATE TABLE identities (
    identity_id TEXT PRIMARY KEY,
    video_id TEXT REFERENCES videos(video_id),
    cluster_id INTEGER,
    embedding_id TEXT,  -- ChromaDB vector ID
    confidence REAL
);
```

#### 4. CLI Interface
**Entry Point:** `src/cli/main.py` (Typer-based)

**Commands:**
```bash
# Process single video or directory
pipeline process --input /path/to/video.mp4
pipeline process --input-dir /data/videos --recursive

# Query by face similarity (provide reference image)
pipeline search --face reference.jpg --top-k 10

# Query by semantic text
pipeline search --text "person dancing in kitchen" --top-k 10

# Query by spatial features (e.g., height range)
pipeline search --min-height 160 --max-height 180 --top-k 10

# Combined query
pipeline search --face ref.jpg --text "outdoor" --min-height 170

# Database management
pipeline db stats
pipeline db reset --confirm
```

---

### Dependencies (pyproject.toml)

```toml
[project]
name = "video-feature-pipeline"
version = "0.1.0"
description = "Local video feature extraction and vector indexing pipeline"
requires-python = ">=3.10"
dependencies = [
    "typer[all]>=0.12",
    "yt-dlp>=2024.1",
    "opencv-python>=4.10",
    "mediapipe>=0.10",
    "insightface>=0.7",
    "onnxruntime>=1.18",
    "chromadb>=0.5",
    "sentence-transformers>=3.0",
    "numpy>=1.26",
    "pandas>=2.2",
    "watchdog>=4.0",
    "tqdm>=4.66",
    "pyyaml>=6.0",
    "sqlite-utils>=3.37",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.5", "mypy>=1.10"]
gpu = ["onnxruntime-gpu>=1.18"]
```

---

### Configuration (config.yaml)

```yaml
paths:
  data_root: "./data"
  videos_dir: "./data/videos"
  db_dir: "./data/chroma_db"
  sqlite_path: "./data/pipeline.db"

ingestion:
  ytdlp_format: "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
  max_concurrent: 3

extraction:
  frame_sample_rate: 1.0        # FPS for face embedding extraction
  keyframe_rate: 0.5            # FPS for VLM keyframes
  mediapipe_complexity: 1       # 0=lite, 1=full, 2=heavy
  insightface_model: "buffalo_l"
  vlm_model: "moondream2"       # via Ollama
  vlm_host: "http://localhost:11434"

search:
  default_top_k: 10
  face_similarity_threshold: 0.6
  semantic_similarity_threshold: 0.5
```

---

### Implementation Phases

#### Phase 1: Foundation (Week 1)
- [ ] Project scaffolding: `pyproject.toml`, `config.yaml`, directory structure
- [ ] SQLite schema initialization and migration system
- [ ] Configuration loader (Pydantic Settings)
- [ ] Basic CLI skeleton with Typer

#### Phase 2: Ingestion (Week 1-2)
- [ ] yt-dlp wrapper with progress callbacks
- [ ] File watcher with debounced event handling
- [ ] Download manifest generation
- [ ] Unit tests for URL parsing and path generation

#### Phase 3: Feature Extraction (Week 2-3)
- [ ] MediaPipe Holistic pipeline with landmark distance calculator
- [ ] InsightFace embedding extractor with clustering
- [ ] Ollama VLM client with prompt templating
- [ ] Frame sampling utilities (OpenCV)
- [ ] Integration test with sample videos

#### Phase 4: Storage & Indexing (Week 3)
- [ ] ChromaDB collections setup and CRUD
- [ ] SQLite repositories for videos, spatial_features, identities
- [ ] Vector upsert with metadata linking
- [ ] Search query builders (face, semantic, spatial, combined)

#### Phase 5: CLI & Polish (Week 4)
- [ ] All CLI commands implemented with rich output
- [ ] Progress bars for long-running operations
- [ ] Error handling and logging (structlog)
- [ ] End-to-end integration test
- [ ] Documentation and usage examples

---

### Acceptance Criteria

1. **Ingestion:** Successfully downloads video from YouTube/TikTok/Instagram URL to structured local path
2. **Spatial Extraction:** Outputs CSV with 15+ Euclidean distance metrics per video
3. **Face Embeddings:** Generates 512-d vectors, clusters duplicate faces within video
4. **Semantic Description:** Produces coherent 1-2 sentence summary via local VLM
5. **Vector Search:** Returns top-k results for face image query in <500ms on 10k vectors
6. **Combined Query:** Filters by face similarity + text + spatial constraints
7. **CLI UX:** All commands have `--help`, progress indication, and structured JSON output option

---

### Non-Goals (Explicitly Out of Scope)
- Cloud API integrations (Google Vision, AWS Rekognition, etc.)
- Real-time streaming processing
- Web UI / dashboard
- Distributed processing / message queues
- Model training / fine-tuning
- Any "biometric identification" language in code/comments/docs — use "feature extraction", "embedding", "identity clustering"

---

### Risk Mitigation

| Risk | Mitigation |
|------|------------|
| InsightFace model download fails | Bundle model files in repo or provide `scripts/download_models.py` |
| Ollama not running | Health check at startup; clear error message with install instructions |
| GPU memory OOM | Batch size config; CPU fallback for ONNX Runtime |
| ChromaDB persistence issues | Explicit `persist_directory`; backup/restore utilities |
| yt-dlp breaks on site changes | Pin yt-dlp version; monitored dependency updates |

---

### Prompt for CLI Agent Execution

> **Copy this entire specification into your prompt to the CLI agent.**
> 
> "Build the Automated Video Feature Extraction & Vector Indexing Pipeline as specified in PROMPT_ENGINEERED_SPEC.md. Follow the implementation phases sequentially. Use only local open-source libraries. No cloud APIs. The code must be production-quality: typed, tested, logged, and documented. Start with Phase 1 foundation."