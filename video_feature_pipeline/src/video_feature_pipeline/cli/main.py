from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from video_feature_pipeline.core.config import get_settings
from video_feature_pipeline.core.logging import get_logger, setup_logging
from video_feature_pipeline.core.processor import VideoProcessor
from video_feature_pipeline.ingestion.downloader import VideoDownloader
from video_feature_pipeline.ingestion.watcher import FileWatcher
from video_feature_pipeline.storage.graph_db import get_graph_db
from video_feature_pipeline.storage.vector_db import get_vector_db

app = typer.Typer(
    name="pipeline",
    help="Automated Video Feature Extraction & Vector Indexing Pipeline",
    add_completion=False,
)
console = Console()
logger = get_logger(__name__)

_json_output = False


@app.callback()
def callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    global _json_output
    _json_output = json_output
    setup_logging()
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


def _print_json(data: Any) -> None:
    if _json_output:
        console.print_json(json.dumps(data, default=str))


@app.command()
def ingest(
    url: str | None = typer.Option(None, "--url", "-u", help="Single video URL to download"),
    batch_file: Path | None = typer.Option(None, "--batch", "-b", help="File with URLs (one per line)"),
    source: str | None = typer.Option(None, "--source", "-s", help="Source platform (youtube, tiktok, twitter, instagram)"),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Download video(s) from URL(s) using yt-dlp."""
    downloader = VideoDownloader(output_dir=output_dir)

    if url:
        with console.status(f"Downloading {url}..."):
            manifest = downloader.download(url, source)
        console.print(f"[green]Downloaded:[/green] {manifest.local_path}")
        _print_json(manifest.__dict__)
    elif batch_file:
        urls = batch_file.read_text().strip().splitlines()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading batch...", total=len(urls))
            for u in urls:
                if u.strip():
                    try:
                        downloader.download(u.strip(), source)
                        progress.advance(task)
                    except Exception as e:
                        console.print(f"[red]Failed:[/red] {u} - {e}")
                        progress.advance(task)
    else:
        console.print("[red]Error:[/red] Must provide --url or --batch")
        raise typer.Exit(1)


@app.command()
def process(
    input_path: Path = typer.Argument(..., help="Video file or directory to process"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Process directory recursively"),
    video_id: str | None = typer.Option(None, "--id", help="Custom video ID"),
    source: str = typer.Option("local", "--source", help="Source identifier"),
):
    """Process video file(s) through the feature extraction pipeline."""
    processor = VideoProcessor()

    try:
        if input_path.is_file():
            video_files = [input_path]
        elif input_path.is_dir():
            pattern = "**/*" if recursive else "*"
            video_files = [
                f for f in input_path.glob(pattern)
                if f.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv"}
            ]
        else:
            console.print(f"[red]Error:[/red] {input_path} not found")
            raise typer.Exit(1)

        if not video_files:
            console.print("[yellow]No video files found[/yellow]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing videos...", total=len(video_files))
            for video_file in video_files:
                vid = video_id or video_file.stem
                progress.update(task, description=f"Processing {video_file.name}")
                try:
                    result = processor.process_video(video_file, video_id=vid, source=source)
                    console.print(f"[green]✓[/green] {video_file.name}: {result}")
                except Exception as e:
                    console.print(f"[red]✗[/red] {video_file.name}: {e}")
                progress.advance(task)

    finally:
        processor.close()


@app.command()
def search(
    face: Path | None = typer.Option(None, "--face", "-f", help="Reference face image for similarity search"),
    text: str | None = typer.Option(None, "--text", "-t", help="Text query for semantic search"),
    min_height: float | None = typer.Option(None, "--min-height", help="Minimum height (cm) for spatial filter"),
    max_height: float | None = typer.Option(None, "--max-height", help="Maximum height (cm) for spatial filter"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results to return"),
    face_threshold: float = typer.Option(0.6, "--face-threshold", help="Face similarity threshold"),
    semantic_threshold: float = typer.Option(0.5, "--semantic-threshold", help="Semantic similarity threshold"),
    # Physical signature filters
    forearm_ratio_min: float | None = typer.Option(None, "--forearm-ratio-min", help="Min forearm/upper-arm ratio"),
    forearm_ratio_max: float | None = typer.Option(None, "--forearm-ratio-max", help="Max forearm/upper-arm ratio"),
    shin_ratio_min: float | None = typer.Option(None, "--shin-ratio-min", help="Min shin/thigh ratio"),
    shin_ratio_max: float | None = typer.Option(None, "--shin-ratio-max", help="Max shin/thigh ratio"),
    torso_ratio_min: float | None = typer.Option(None, "--torso-ratio-min", help="Min torso/shoulder ratio"),
    torso_ratio_max: float | None = typer.Option(None, "--torso-ratio-max", help="Max torso/shoulder ratio"),
):
    """Search the indexed database."""
    vector_db = get_vector_db()

    results: list[dict[str, Any]] = []

    if face:
        console.print("[yellow]Face search not yet implemented - requires face embedding extraction from reference image[/yellow]")

    if text:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = model.encode(text, convert_to_numpy=True).astype("float32")
        semantic_results = vector_db.search_semantic_similar(
            query_embedding,
            top_k=top_k,
            threshold=semantic_threshold,
        )
        for r in semantic_results:
            r["match_type"] = "semantic"
            results.append(r)
    # Physical signature search
    ratio_filters = {}
    if forearm_ratio_min is not None or forearm_ratio_max is not None:
        ratio_filters["forearm_to_upper_arm_L"] = (
            forearm_ratio_min or 0.0, forearm_ratio_max or 10.0
        )
        ratio_filters["forearm_to_upper_arm_R"] = (
            forearm_ratio_min or 0.0, forearm_ratio_max or 10.0
        )
    if shin_ratio_min is not None or shin_ratio_max is not None:
        ratio_filters["shin_to_thigh_L"] = (
            shin_ratio_min or 0.0, shin_ratio_max or 10.0
        )
        ratio_filters["shin_to_thigh_R"] = (
            shin_ratio_min or 0.0, shin_ratio_max or 10.0
        )
    if torso_ratio_min is not None or torso_ratio_max is not None:
        ratio_filters["torso_to_shoulder"] = (
            torso_ratio_min or 0.0, torso_ratio_max or 10.0
        )

    if ratio_filters:
        from video_feature_pipeline.core.database import get_database
        db = get_database()
        physical_results = db.search_by_physical_signature(
            ratio_filters=ratio_filters,
            limit=top_k,
        )
        for r in physical_results:
            r["match_type"] = "physical"
            results.append(r)

    if not results and not face and not text:
        console.print("[red]Error:[/red] Must provide --face or --text query")
        raise typer.Exit(1)

    if results:
        table = Table(title=f"Search Results ({len(results)})")
        table.add_column("Type", style="cyan")
        table.add_column("Video ID", style="magenta")
        table.add_column("Similarity", style="green")
        table.add_column("Details")

        for r in results:
            if r["match_type"] == "semantic":
                table.add_row(
                    "Semantic",
                    r["video_id"],
                    f"{r['similarity']:.3f}",
                    r["description"][:80] + "..." if len(r["description"]) > 80 else r["description"],
                )
            elif r["match_type"] == "face":
                table.add_row(
                    "Face",
                    r["video_id"],
                    f"{r['similarity']:.3f}",
                    f"Cluster {r['cluster_id']}, Conf: {r['confidence']:.2f}",
                )
            elif r["match_type"] == "physical":
                sig = r.get("physical_signature_json", "{}")
                import json
                try:
                    ratios = json.loads(sig)
                except Exception:
                    ratios = {}
                table.add_row(
                    "Physical",
                    r["video_id"],
                    "N/A",
                    ", ".join(f"{k}: {v:.3f}" for k, v in ratios.items()),
                )

        console.print(table)
        _print_json(results)
    else:
        console.print("[yellow]No matches found[/yellow]")


@app.command()
def watch(
    input_dir: Path | None = typer.Option(None, "--input", "-i", help="Directory to watch"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Watch recursively"),
):
    """Start file watcher to auto-process new videos."""
    processor = VideoProcessor()

    async def process_file(path: Path):
        try:
            await processor.process_video_async(path, source="watched")
        except Exception as e:
            logger.error("watch_process_failed", path=str(path), error=str(e))

    watcher = FileWatcher(input_dir=input_dir, processor=process_file, recursive=recursive)

    console.print(f"[green]Watching {watcher.input_dir}... Press Ctrl+C to stop[/green]")
    try:
        asyncio.run(watcher.run_forever())
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping watcher...[/yellow]")
    finally:
        processor.close()


@app.command()
def stats():
    """Show database statistics."""
    from video_feature_pipeline.core.database import get_database
    db = get_database()
    vector_db = get_vector_db()

    db_stats = db.get_stats()
    vec_stats = vector_db.get_stats()

    table = Table(title="Pipeline Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    for key, value in db_stats.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    for key, value in vec_stats.items():
        table.add_row(key.replace("_", " ").title(), str(value))

    console.print(table)
    _print_json({**db_stats, **vec_stats})

@app.command()
def graph_stats():
    """Show graph database statistics."""
    graph_db = get_graph_db()
    
    if not graph_db.is_available():
        console.print("[red]Neo4j not available[/red]")
        console.print("Install neo4j: pip install neo4j")
        console.print("Configure in config.yaml or environment variables")
        return
    
    stats = graph_db.get_stats()
    
    table = Table(title="Graph Database Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    for key, value in stats.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(table)
    _print_json(stats)


@app.command()
def graph_partners(
    identity_id: str = typer.Argument(..., help="Identity ID to find partners for"),
    min_weight: int = typer.Option(1, "--min-weight", help="Minimum co-occurrence weight"),
):
    """Show co-occurrence partners for an identity."""
    graph_db = get_graph_db()
    
    if not graph_db.is_available():
        console.print("[red]Neo4j not available[/red]")
        return
    
    partners = graph_db.get_partners(identity_id, min_weight=min_weight)
    
    if not partners:
        console.print(f"[yellow]No partners found for {identity_id}[/yellow]")
        return
    
    table = Table(title=f"Partners for {identity_id}")
    table.add_column("Partner ID", style="cyan")
    table.add_column("Weight", style="magenta")
    table.add_column("Videos", style="green")
    table.add_column("Frames", style="blue")
    
    for p in partners:
        table.add_row(
            p["partner_id"],
            str(p["weight"]),
            str(len(p["videos"])),
            str(len(p["frames"])),
        )
    
    console.print(table)
    _print_json(partners)


@app.command()
def graph_network(
    identity_id: str = typer.Argument(..., help="Identity ID to explore network for"),
    depth: int = typer.Option(2, "--depth", help="Network depth"),
    export: Path | None = typer.Option(None, "--export", "-e", help="Export as GraphML"),
):
    """Show or export identity network."""
    graph_db = get_graph_db()
    
    if not graph_db.is_available():
        console.print("[red]Neo4j not available[/red]")
        return
    
    network = graph_db.get_network(identity_id, depth=depth)
    
    console.print(f"[green]Network for {identity_id} (depth {depth}):[/green]")
    console.print(f"  Nodes: {len(network['nodes'])}")
    console.print(f"  Edges: {len(network['edges'])}")
    
    if export:
        graph_db.export_graphml(str(export))
        console.print(f"[green]Exported to {export}[/green]")
    
    _print_json(network)


@app.command()
def graph_export(
    path: Path = typer.Argument(..., help="Output GraphML file path"),
):
    """Export entire graph as GraphML."""
    graph_db = get_graph_db()
    
    if not graph_db.is_available():
        console.print("[red]Neo4j not available[/red]")
        return
    
    graph_db.export_graphml(str(path))
    console.print(f"[green]Graph exported to {path}[/green]")

@app.command()
def harvest_status():
    """Show harvest configuration and status."""
    settings = get_settings()
    harvest = settings.harvest
    
    table = Table(title="Harvest Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Enabled", str(harvest.enabled))
    table.add_row("SocQ API URL", harvest.socq_api_url)
    table.add_row("API Key Set", "Yes" if harvest.socq_api_key else "No")
    table.add_row("Min Confidence", str(harvest.min_confidence))
    table.add_row("Max Concurrent", str(harvest.max_concurrent))
    table.add_row("Auto Scrape on New Identity", str(harvest.auto_scrape_on_new_identity))
    
    console.print(table)
    _print_json(harvest.model_dump())


@app.command()
def harvest_configure(
    enabled: bool | None = typer.Option(None, "--enabled/--disabled", help="Enable/disable harvest"),
    api_key: str | None = typer.Option(None, "--api-key", help="SocQ API key"),
    min_confidence: float | None = typer.Option(None, "--min-confidence", help="Minimum confidence threshold"),
    auto_scrape: bool | None = typer.Option(None, "--auto-scrape/--no-auto-scrape", help="Auto scrape on new identity"),
):
    """Configure harvest settings (updates config.yaml)."""
    import yaml
    
    config_path = Path("config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if "harvest" not in config:
        config["harvest"] = {}
    
    if enabled is not None:
        config["harvest"]["enabled"] = enabled
    if api_key is not None:
        config["harvest"]["socq_api_key"] = api_key
    if min_confidence is not None:
        config["harvest"]["min_confidence"] = min_confidence
    if auto_scrape is not None:
        config["harvest"]["auto_scrape_on_new_identity"] = auto_scrape
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    console.print("[green]Harvest configuration updated[/green]")
    # Reload settings
    from video_feature_pipeline.core.config import reload_settings
    reload_settings()
    console.print("Settings reloaded")


@app.command()
def harvest_trigger(
    identity_id: str = typer.Argument(..., help="Identity ID to trigger harvest for"),
):
    """Manually trigger harvest for an identity."""
    from video_feature_pipeline.core.database import get_database
    from video_feature_pipeline.ingestion.socq_client import get_socq_client
    from video_feature_pipeline.ingestion.downloader import VideoDownloader
    
    settings = get_settings()
    db = get_database()
    
    identity = db.get_identity_by_embedding_id(identity_id)
    if not identity:
        console.print(f"[red]Identity {identity_id} not found[/red]")
        return
    
    video = db.get_video(identity["video_id"])
    source_handle = video.get("source") if video else None
    
    if not source_handle:
        console.print("[red]No source handle for this identity[/red]")
        return
    
    if not settings.harvest.socq_api_key:
        console.print("[red]SocQ API key not configured[/red]")
        return
    
    console.print(f"[green]Triggering harvest for {identity_id} from {source_handle}[/green]")
    
    async def run_harvest():
        socq = get_socq_client()
        downloader = VideoDownloader()
        
        try:
            videos = await socq.get_profile_videos(
                profile_id=source_handle,
                platform="generic",
                limit=settings.harvest.max_concurrent * 10,
            )
            
            console.print(f"Found {len(videos)} videos to harvest")
            
            for video_info in videos[:settings.harvest.max_concurrent]:
                url = video_info.get("url")
                if url:
                    try:
                        with console.status(f"Downloading {url}..."):
                            downloader.download(url, source=source_handle)
                        console.print(f"[green]✓[/green] {url}")
                    except Exception as e:
                        console.print(f"[red]✗[/red] {url}: {e}")
        
        except Exception as e:
            console.print(f"[red]Harvest failed: {e}[/red]")
        finally:
            await socq.close()
    
    asyncio.run(run_harvest())


@app.command()
def reset(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Confirm reset without prompt"),
):
    """Reset all databases (DANGEROUS - deletes all data)."""
    if not confirm:
        confirm = typer.confirm("This will delete ALL data. Are you sure?")

    if confirm:
        from video_feature_pipeline.core.database import get_database
        db = get_database()
        vector_db = get_vector_db()

        db.db["videos"].delete_where()
        db.db["spatial_features"].delete_where()
        db.db["identities"].delete_where()

        vector_db.client.delete_collection("face_embeddings")
        vector_db.client.delete_collection("semantic_embeddings")

        console.print("[green]Databases reset[/green]")
    else:
        console.print("[yellow]Cancelled[/yellow]")


if __name__ == "__main__":
    app()