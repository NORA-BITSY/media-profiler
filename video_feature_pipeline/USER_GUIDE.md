# User Guide

## Quick Start

### 1. Installation

```bash
# Clone and setup
cd video_feature_pipeline
python3 -m venv venv
source venv/bin/activate
pip install -e .

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install -y ffmpeg

# For YouTube support, install a JS runtime
# Option 1: deno (recommended by yt-dlp)
curl -fsSL https://deno.land/install.sh | sh
# Option 2: nodejs
sudo apt-get install -y nodejs npm
```

### 2. Configuration

```bash
# Edit config.yaml for your environment
cp config.yaml config.local.yaml  # optional local override
# Edit paths, API keys, model settings
```

Key settings to review:
- `paths.videos_dir`: Where downloaded videos are stored
- `harvest.socq_api_key`: Required for automatic harvesting
- `extraction.vlm_host`: Ollama endpoint (default: http://localhost:11434)

### 3. Start Required Services

```bash
# Start Ollama for VLM (semantic descriptions)
ollama serve &
ollama pull moondream2  # or llava:7b

# Optional: Start Neo4j for relationship graphs
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

### 4. Basic Usage

```bash
# Download a video
pipeline ingest --url "https://youtube.com/watch?v=..." --source youtube

# Process video through feature extraction
pipeline process ./data/videos/youtube/VIDEO_ID.mp4

# Search by text description
pipeline search --text "person dancing in kitchen" --top-k 10

# Search by physical attributes
pipeline search --forearm-ratio-min 0.8 --forearm-ratio-max 1.2

# View statistics
pipeline stats

# Start file watcher (auto-process new videos)
pipeline watch --input ./data/videos
```

## Commands Reference

### `pipeline ingest` — Download Videos

```bash
# Single URL
pipeline ingest --url "https://youtube.com/watch?v=dQw4w9WgXcQ" --source youtube

# Batch from file (one URL per line)
pipeline ingest --batch urls.txt --source tiktok

# Custom output directory
pipeline ingest --url "..." --output ./my_videos
```

**Options:**
| Flag | Description |
|------|-------------|
| `--url, -u` | Single video URL |
| `--batch, -b` | File with URLs (one per line) |
| `--source, -s` | Platform: youtube, tiktok, twitter, instagram |
| `--output, -o` | Custom output directory |

### `pipeline process` — Feature Extraction

```bash
# Single file
pipeline process ./data/videos/youtube/dQw4w9WgXcQ.mp4

# Directory (non-recursive)
pipeline process ./data/videos/youtube/

# Recursive directory scan
pipeline process ./data/videos/ --recursive

# Custom video ID and source
pipeline process video.mp4 --id custom_id --source my_source
```

**What happens during processing:**
1. Video metadata extraction (duration, fps, resolution)
2. **Spatial analysis**: MediaPipe holistic landmarks → 10 limb measurements + 4 face distances + 7 invariant ratios
3. **Identity analysis**: InsightFace embeddings → face clusters with confidence
4. **Semantic analysis**: Keyframes → Ollama VLM descriptions → sentence embeddings
5. Storage in SQLite, ChromaDB, and Neo4j (if available)

### `pipeline search` — Query Database

```bash
# Text search (semantic)
pipeline search --text "person singing" --top-k 10

# Face similarity (requires reference image - not yet implemented)
pipeline search --face reference.jpg

# Physical signature filters
pipeline search --forearm-ratio-min 0.8 --forearm-ratio-max 1.2
pipeline search --shin-ratio-min 1.0 --shin-ratio-max 1.5
pipeline search --torso-ratio-min 2.0 --torso-ratio-max 3.0

# Combined search
pipeline search --text "outdoor" --forearm-ratio-min 0.9 --top-k 20

# JSON output
pipeline search --text "dancing" --json
```

**Physical Signature Filters:**
| Filter | Description | Typical Range |
|--------|-------------|---------------|
| `--forearm-ratio-min/max` | Forearm / Upper arm | 0.8 - 1.2 |
| `--shin-ratio-min/max` | Shin / Thigh | 1.0 - 1.5 |
| `--torso-ratio-min/max` | Torso / Shoulder width | 2.0 - 3.0 |

### `pipeline watch` — Auto-Processing

```bash
# Watch default directory
pipeline watch

# Custom directory
pipeline watch --input ./incoming_videos

# Non-recursive
pipeline watch --input ./videos --no-recursive
```

Watches for new video files and automatically processes them. Press Ctrl+C to stop.

### `pipeline stats` — Statistics

```bash
pipeline stats
pipeline stats --json
```

Shows counts for videos, identities, embeddings, co-occurrences.

### `pipeline graph_*` — Relationship Graphs

```bash
# Graph statistics
pipeline graph-stats

# Co-occurrence partners for an identity
pipeline graph-partners IDENTITY_ID --min-weight 2

# Network around an identity (BFS depth)
pipeline graph-network IDENTITY_ID --depth 2

# Export full graph as GraphML
pipeline graph-export network.graphml
```

### `pipeline harvest_*` — Automatic Harvesting

```bash
# View harvest configuration
pipeline harvest-status

# Configure harvest (updates config.yaml)
pipeline harvest-configure --enabled --min-confidence 0.9
pipeline harvest-configure --api-key YOUR_SOCQ_KEY
pipeline harvest-configure --auto-scrape

# Manually trigger harvest for an identity
pipeline harvest-trigger IDENTITY_ID
```

**Harvest Configuration:**
| Setting | Description |
|---------|-------------|
| `enabled` | Master toggle |
| `socq_api_key` | SocQ API key (required for harvesting) |
| `min_confidence` | Minimum cluster confidence to trigger (default: 0.85) |
| `max_concurrent` | Max videos to download per trigger (default: 3) |
| `auto_scrape_on_new_identity` | Auto-harvest on high-confidence new identities |

### `pipeline reset` — Reset Databases

```bash
# With confirmation
pipeline reset

# Force (no prompt)
pipeline reset --yes
```

**⚠️ Dangerous:** Deletes all videos, embeddings, identities, co-occurrences, and graph data.

## Configuration

### Key Settings

```yaml
# config.yaml
paths:
  videos_dir: "./data/videos"          # Video storage
  sqlite_path: "./data/pipeline.db"    # SQLite database
  db_dir: "./data/chroma_db"           # ChromaDB directory

extraction:
  frame_sample_rate: 1.0               # FPS for spatial/identity extraction
  keyframe_rate: 0.5                   # FPS for VLM keyframes
  mediapipe_complexity: 1              # 0=lite, 1=full, 2=heavy
  insightface_model: "buffalo_l"       # buffalo_l, buffalo_s, buffalo_m
  vlm_model: "moondream2"              # moondream2, llava:7b
  vlm_host: "http://localhost:11434"   # Ollama endpoint
  
  # Adaptive frame rate
  adaptive_frame_rate: false
  base_frame_rate: 0.5
  high_confidence_frame_rate: 2.0
  identity_confidence_threshold: 0.8

harvest:
  enabled: false
  socq_api_key: ""                     # REQUIRED for harvesting
  min_confidence: 0.85
  max_concurrent: 3
  auto_scrape_on_new_identity: true
```

### Environment Overrides

```bash
# Override any setting via environment
export VFP_EXTRACTION_FRAME_SAMPLE_RATE=2.0
export VFP_HARVEST_ENABLED=true
export VFP_HARVEST_SOCQ_API_KEY=your_key_here
export VFP_LOGGING_LEVEL=DEBUG
```

## Common Workflows

### 1. Build a Performer Database

```bash
# 1. Download videos from a platform
pipeline ingest --batch performer_urls.txt --source youtube

# 2. Process all videos
pipeline process ./data/videos/youtube/ --recursive

# 3. View stats
pipeline stats

# 4. Find co-occurrences (who appears with whom)
pipeline graph-partners IDENTITY_OF_INTEREST
```

### 2. Search by Physical Type

```bash
# Find tall, lean builds (high torso/shoulder ratio)
pipeline search --torso-ratio-min 2.5 --torso-ratio-max 3.5

# Find specific limb proportions
pipeline search --forearm-ratio-min 0.95 --forearm-ratio-max 1.05 \
                --shin-ratio-min 1.1 --shin-ratio-max 1.3
```

### 3. Automatic Monitoring

```bash
# Terminal 1: Start watcher
pipeline watch --input ./data/videos

# Terminal 2: Drop videos into ./data/videos/
# They auto-process and harvest if enabled
```

### 4. Export for Analysis

```bash
# Export relationship graph for Gephi/NetworkX
pipeline graph-export performer_network.graphml

# Search results as JSON for scripts
pipeline search --text "dancing" --json > results.json
```

## Troubleshooting

### Video Download Fails
```bash
# Update yt-dlp
pip install -U yt-dlp

# Install JS runtime for YouTube
curl -fsSL https://deno.land/install.sh | sh
```

### "Neo4j not available"
- Graph features auto-disable if Neo4j unreachable
- Check `neo4j_uri`, `neo4j_user`, `neo4j_password` in config
- Or run: `docker run -d -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j`

### "Ollama connection refused"
```bash
ollama serve
ollama pull moondream2
# Or use llava:7b (larger, more accurate)
ollama pull llava:7b
```

### InsightFace Slow / OOM
```bash
# Use smaller model
# config.yaml: insightface_model: "buffalo_s"

# Quantize models (2-4x speedup)
python scripts/quantize_onnx.py --model buffalo_l --qdq --output ./models/quantized
```

### Memory Issues
```yaml
# Reduce extraction rates
extraction:
  frame_sample_rate: 0.5
  keyframe_rate: 0.25
  max_keyframes: 10
```

### Database Locked
```bash
# Only one pipeline process at a time
# Or reset: pipeline reset --yes
```

## Output Locations

| Data | Location |
|------|----------|
| Downloaded videos | `./data/videos/{source}/{video_id}.mp4` |
| Video manifests | `./data/videos/{source}/{video_id}.manifest.json` |
| SQLite database | `./data/pipeline.db` |
| ChromaDB vectors | `./data/chroma_db/` |
| Logs | `./data/pipeline.log` |
| Quantized models | `./models/quantized/` |

## Getting Help

```bash
# Command help
pipeline --help
pipeline ingest --help
pipeline search --help

# Verbose logging
pipeline -v process video.mp4

# JSON output for scripting
pipeline --json stats
pipeline --json search --text "query"
```