# Video Feature Extraction & Vector Indexing Pipeline

A production-grade, fully local computer vision pipeline for automated video analysis. Extracts spatial measurements, identity embeddings, and semantic descriptions from video files, then indexes them in a searchable vector database with relationship graphs.

## Features

### Core Extraction
- **Spatial Analysis** — MediaPipe Holistic (33 pose + 478 face landmarks) → 10 limb measurements, 4 face distances, **7 camera-invariant ratios**
- **Identity Embeddings** — InsightFace (buffalo_l/s) → 512-d face embeddings with DBSCAN clustering
- **Semantic Descriptions** — Ollama VLM (moondream2/llava) + SentenceTransformer → 384-d semantic vectors

### Search & Retrieval
- **Text Search** — Natural language queries via semantic embeddings
- **Physical Signature Search** — Filter by invariant body ratios (forearm/upper-arm, shin/thigh, torso/shoulder)
- **Face Similarity** — Embedding-based identity matching (reference image support planned)
- **Combined Queries** — Chain text + physical + face filters

### Relationship Intelligence
- **Co-occurrence Graph** — Automatic detection of identities appearing in same frames
- **Neo4j Integration** — Persistent graph with partner queries, network exploration, GraphML export
- **Identity Web** — Track associations across videos and time

### Automation
- **File Watcher** — Auto-process new videos dropped in directory
- **Closed-Loop Harvesting** — High-confidence new identities trigger SocQ profile scraping → download → process
- **Adaptive Frame Rate** — Base-rate scan → high-confidence re-analysis at higher FPS

### Performance
- **ONNX Quantization** — Dynamic/static/QDQ INT8 quantization (2-4x speedup)
- **Batch Embeddings** — Optimized ChromaDB batch inserts
- **Local-First** — No cloud APIs, all models run offline

## Quick Start

```bash
# 1. Setup
git clone <repo> video_feature_pipeline
cd video_feature_pipeline
python3 -m venv venv && source venv/bin/activate
pip install -e .

# 2. System dependencies
sudo apt-get install -y ffmpeg
# For YouTube: install deno or nodejs

# 3. Start services
ollama serve & ollama pull moondream2
# Optional: Neo4j for graphs
docker run -d -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j

# 4. Configure
# Edit config.yaml (paths, API keys, model settings)

# 5. Run
pipeline ingest --url "https://youtube.com/watch?v=..." --source youtube
pipeline process ./data/videos/youtube/VIDEO_ID.mp4
pipeline search --text "person dancing" --top-k 10
```

## Architecture

```
┌─────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│  Ingestion  │───▶│  Feature Extraction  │───▶│  Storage/Index  │
│  (yt-dlp)   │    │  • MediaPipe         │    │  • ChromaDB     │
│  File Watch │    │  • InsightFace       │    │  • SQLite       │
│  SocQ API   │    │  • Ollama VLM        │    │  • Neo4j        │
└─────────────┘    └──────────────────────┘    └─────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         Spatial         Identity      Semantic
         (ratios)        (embeddings)  (descriptions)
```

## Documentation

| Guide | Description |
|-------|-------------|
| [User Guide](USER_GUIDE.md) | CLI commands, configuration, workflows, troubleshooting |
| [Developer Guide](DEVELOPER_GUIDE.md) | Architecture, data flow, extending, testing, performance |
| [Prompt-Engineered Spec](PROMPT_ENGINEERED_SPEC.md) | Original technical specification for CLI agent execution |

## CLI Commands

```bash
# Ingestion
pipeline ingest --url "https://youtube.com/..." --source youtube
pipeline ingest --batch urls.txt

# Processing
pipeline process video.mp4
pipeline process ./videos/ --recursive

# Search
pipeline search --text "person dancing"
pipeline search --forearm-ratio-min 0.8 --forearm-ratio-max 1.2
pipeline search --text "outdoor" --shin-ratio-min 1.0

# Graphs
pipeline graph-stats
pipeline graph-partners IDENTITY_ID
pipeline graph-network IDENTITY_ID --depth 2
pipeline graph-export network.graphml

# Harvesting
pipeline harvest-status
pipeline harvest-configure --enabled --api-key KEY
pipeline harvest-trigger IDENTITY_ID

# Monitoring
pipeline watch --input ./data/videos
pipeline stats

# Maintenance
pipeline reset --yes
```

## Configuration

Key settings in `config.yaml`:

```yaml
extraction:
  frame_sample_rate: 1.0           # FPS for analysis
  adaptive_frame_rate: false       # Enable adaptive two-pass
  base_frame_rate: 0.5
  high_confidence_frame_rate: 2.0
  insightface_model: "buffalo_l"   # or buffalo_s for speed
  vlm_model: "moondream2"          # Ollama model

harvest:
  enabled: false
  socq_api_key: ""                 # Required for auto-harvest
  min_confidence: 0.85
  auto_scrape_on_new_identity: true
```

Environment overrides: `VFP_EXTRACTION_FRAME_SAMPLE_RATE=2.0`, `VFP_HARVEST_SOCQ_API_KEY=key`

## Performance Optimization

| Technique | Speedup | Command |
|-----------|---------|---------|
| ONNX Quantization (QDQ) | 2-4x | `python scripts/quantize_onnx.py --model buffalo_l --qdq` |
| Batch Embeddings | 10-50x | Use `add_face_embeddings_batch()` |
| Adaptive Frame Rate | 2-5x | `adaptive_frame_rate: true` |
| Smaller Model | 2-3x | `insightface_model: "buffalo_s"` |

## Requirements

- Python 3.10+
- FFmpeg (system)
- Ollama (for VLM)
- Optional: Neo4j (graphs), deno/nodejs (YouTube)

## License

MIT