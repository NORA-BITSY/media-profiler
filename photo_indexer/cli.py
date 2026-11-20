"""CLI core: click-based CLI with index/analyze/enrich/export/sidecar commands."""

from __future__ import annotations

import csv
import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import List, Optional

import click

from .database import PhotoIndexDB
from .metadata import (
    find_images,
    parse_exif,
    run_exiftool,
    sha256_file,
    write_xmp_sidecar,
)
from .metrics import (
    INDEXED_TOTAL,
    SIDECAR_WRITTEN_TOTAL,
    VLM_ANALYZED_TOTAL,
    ERRORS_TOTAL,
    index_timer,
    start_metrics_server,
    vlm_timer,
)
from .models import Asset, AssetStatus, deterministic_asset_id
from .ner import extract_entities, merge_entities
from .taxonomies import ALL_TAXONOMIES, list_taxonomies
from .vlm import VLMClient


# ---------------------------------------------------------------------------
# Helper: default paths from env
# ---------------------------------------------------------------------------
def _default_db_path() -> Path:
    import os
    return Path(os.environ.get("PHOTO_INDEX_DB", "./photo_index.db"))


def _default_staging() -> Path:
    import os
    return Path(os.environ.get("STAGING_LOCATION", "./staging"))


def _default_library() -> Path:
    import os
    return Path(os.environ.get("LIBRARY_LOCATION", "./library"))


def _default_vlm_model() -> str:
    import os
    return os.environ.get("VLM_MODEL", "llava")


def _default_vlm_url() -> str:
    import os
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------
@click.group()
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to SQLite index database (default: PHOTO_INDEX_DB or ./photo_index.db)",
)
@click.option(
    "--metrics-port",
    type=int,
    default=None,
    help="Start Prometheus metrics HTTP server on this port (default: disabled)",
)
@click.pass_context
def main(ctx: click.Context, db_path: Optional[Path], metrics_port: Optional[int]) -> None:
    """Photo Indexer — self-hosted, local-first photo indexing CLI.

    Index, enrich, and export photos with local VLM + NER + taxonomies.
    """
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db_path or _default_db_path()
    ctx.obj["db"] = PhotoIndexDB(ctx.obj["db_path"])
    if metrics_port:
        start_metrics_server(metrics_port)


# ---------------------------------------------------------------------------
# index command
# ---------------------------------------------------------------------------
@main.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    help="Recursively scan directories (default: yes)",
)
@click.option(
    "--skip-vlm",
    is_flag=True,
    default=False,
    help="Skip VLM analysis during indexing",
)
@click.option(
    "--vlm-model",
    default=None,
    help="VLM model to use (default: VLM_MODEL env or llava)",
)
@click.option(
    "--vlm-url",
    default=None,
    help="Ollama base URL (default: OLLAMA_BASE_URL or http://localhost:11434)",
)
@click.option(
    "--write-sidecar/--no-write-sidecar",
    default=False,
    help="Write XMP sidecar during indexing (default: no)",
)
@click.pass_context
def index(
    ctx: click.Context,
    paths: List[Path],
    recursive: bool,
    skip_vlm: bool,
    vlm_model: Optional[str],
    vlm_url: Optional[str],
    write_sidecar: bool,
) -> None:
    """Index image files from PATH(s): extract Exif metadata, hash, optionally run VLM + NER."""

    db: PhotoIndexDB = ctx.obj["db"]
    model = vlm_model or _default_vlm_model()
    url = vlm_url or _default_vlm_url()

    vlm: Optional[VLMClient] = None
    if not skip_vlm:
        vlm = VLMClient(base_url=url, model=model)
        if not vlm.is_available():
            click.echo(
                f"WARNING: Ollama not reachable at {url}. VLM analysis will be skipped.",
                err=True,
            )
            vlm = None

    total = 0
    indexed = 0

    with index_timer():
        for base in paths:
            base = base.resolve()
            images = find_images(base, recursive=recursive)
            click.echo(f"Found {len(images)} image(s) under {base}")

            for img_path in images:
                total += 1
                rel = str(img_path)
                click.echo(f"  [{total}] {rel}", nl=False)

                # Hash + Exif
                digest = sha256_file(img_path)
                raw_exif = run_exiftool(img_path)
                if not raw_exif and not shutil.which("exiftool"):
                    click.echo(" [exiftool not found, metadata skipped]", nl=False)
                exif = parse_exif(raw_exif) if raw_exif else None

                # Build asset
                asset = Asset(
                    id=deterministic_asset_id(rel, digest),
                    path=rel,
                    filename=img_path.name,
                    content_sha256=digest,
                    file_size=img_path.stat().st_size if img_path.exists() else None,
                    mime_type=exif.mime_type if exif else None,
                    exif=exif,
                    status=AssetStatus.INDEXED,
                )

                # VLM analysis
                if vlm is not None:
                    click.echo(" ... vlm", nl=False)
                    with vlm_timer():
                        try:
                            result = vlm.analyze_image(img_path, model=model)
                            asset.vlm = result
                            if result.caption or result.detailed_description:
                                text = (result.caption or "") + " " + (result.detailed_description or "")
                                entities = extract_entities(text)
                                asset.ner_entities = merge_entities(asset.ner_entities, entities)
                            VLM_ANALYZED_TOTAL.labels("ok").inc()
                        except Exception as exc:  # noqa: BLE001
                            click.echo(f" [vlm error: {exc}]", nl=False)
                            asset.status = AssetStatus.ERROR
                            asset.error_message = str(exc)
                            VLM_ANALYZED_TOTAL.labels("error").inc()
                            ERRORS_TOTAL.labels("index").inc()

                # Sidecar
                if write_sidecar:
                    status = write_xmp_sidecar(
                        img_path,
                        description=asset.vlm.caption if asset.vlm else None,
                        keywords=[e.label for e in asset.ner_entities if e.label],
                    )
                    asset.sidecar_status = status
                    SIDECAR_WRITTEN_TOTAL.labels(status).inc()

                asset.status = AssetStatus.ENRICHED if asset.status != AssetStatus.ERROR else AssetStatus.ERROR
                db.upsert(asset)
                indexed += 1
                INDEXED_TOTAL.labels(asset.status.value).inc()
                click.echo(" done")

    click.echo(f"\nIndexed {indexed}/{total} assets into {ctx.obj['db_path']}")


# ---------------------------------------------------------------------------
# analyze command
# ---------------------------------------------------------------------------
@main.command()
@click.option(
    "--asset-id",
    default=None,
    help="Analyze a specific asset by ID",
)
@click.option(
    "--all",
    "analyze_all",
    is_flag=True,
    default=False,
    help="Re-analyze all indexed assets",
)
@click.option(
    "--vlm-model",
    default=None,
    help="VLM model to use",
)
@click.option(
    "--vlm-url",
    default=None,
    help="Ollama base URL",
)
@click.pass_context
def analyze(
    ctx: click.Context,
    asset_id: Optional[str],
    analyze_all: bool,
    vlm_model: Optional[str],
    vlm_url: Optional[str],
) -> None:
    """Run VLM analysis on indexed assets (re-run or targeted)."""

    db: PhotoIndexDB = ctx.obj["db"]
    model = vlm_model or _default_vlm_model()
    url = vlm_url or _default_vlm_url()

    vlm = VLMClient(base_url=url, model=model)
    if not vlm.is_available():
        click.echo(f"ERROR: Ollama not reachable at {url}", err=True)
        sys.exit(1)

    if analyze_all:
        assets = db.list_all(limit=10000)
        click.echo(f"Re-analyzing {len(assets)} assets...")
        for a in assets:
            img_path = Path(a.path)
            if not img_path.exists():
                click.echo(f"  SKIP (missing): {a.path}")
                continue
            result = vlm.analyze_image(img_path, model=model)
            a.vlm = result
            if result.caption or result.detailed_description:
                text = (result.caption or "") + " " + (result.detailed_description or "")
                entities = extract_entities(text)
                a.ner_entities = merge_entities(a.ner_entities, entities)
            a.status = AssetStatus.ANALYZED
            db.upsert(a)
            click.echo(f"  OK: {a.filename}")
    elif asset_id:
        a = db.get(asset_id)
        if not a:
            click.echo(f"Asset not found: {asset_id}", err=True)
            sys.exit(1)
        img_path = Path(a.path)
        if not img_path.exists():
            click.echo(f"Image not found: {img_path}", err=True)
            sys.exit(1)
        result = vlm.analyze_image(img_path, model=model)
        a.vlm = result
        if result.caption or result.detailed_description:
            text = (result.caption or "") + " " + (result.detailed_description or "")
            entities = extract_entities(text)
            a.ner_entities = merge_entities(a.ner_entities, entities)
        a.status = AssetStatus.ANALYZED
        db.upsert(a)
        click.echo(f"OK: {a.filename}")
    else:
        click.echo("Specify --asset-id or --all", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# enrich command
# ---------------------------------------------------------------------------
@main.command()
@click.option(
    "--asset-id",
    default=None,
    help="Enrich a specific asset by ID",
)
@click.option(
    "--all",
    "enrich_all",
    is_flag=True,
    default=False,
    help="Run NER enrichment on all indexed assets",
)
@click.option(
    "--taxonomy",
    "taxonomy_names",
    multiple=True,
    help="Taxonomy names to use (default: all)",
)
@click.pass_context
def enrich(
    ctx: click.Context,
    asset_id: Optional[str],
    enrich_all: bool,
    taxonomy_names: List[str],
) -> None:
    """Run NER entity extraction against VLM text using taxonomies."""

    db: PhotoIndexDB = ctx.obj["db"]
    names = list(taxonomy_names) if taxonomy_names else None

    if enrich_all:
        assets = db.list_all(limit=10000)
        for a in assets:
            text_parts = []
            if a.vlm and a.vlm.caption:
                text_parts.append(a.vlm.caption)
            if a.vlm and a.vlm.detailed_description:
                text_parts.append(a.vlm.detailed_description)
            text = " ".join(text_parts)
            if text:
                entities = extract_entities(text, taxonomy_names=names)
                a.ner_entities = merge_entities(a.ner_entities, entities)
                db.upsert(a)
                click.echo(f"  OK: {a.filename} ({len(entities)} entities)")
    elif asset_id:
        a = db.get(asset_id)
        if not a:
            click.echo(f"Asset not found: {asset_id}", err=True)
            sys.exit(1)
        text_parts = []
        if a.vlm and a.vlm.caption:
            text_parts.append(a.vlm.caption)
        if a.vlm and a.vlm.detailed_description:
            text_parts.append(a.vlm.detailed_description)
        text = " ".join(text_parts)
        if text:
            entities = extract_entities(text, taxonomy_names=names)
            a.ner_entities = merge_entities(a.ner_entities, entities)
            db.upsert(a)
            click.echo(f"OK: {a.filename} ({len(entities)} entities)")
    else:
        click.echo("Specify --asset-id or --all", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# sidecar command
# ---------------------------------------------------------------------------
@main.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing sidecar files",
)
@click.option(
    "--description",
    default=None,
    help="Description to write into sidecar",
)
@click.option(
    "--keywords",
    default=None,
    help="Comma-separated keywords",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Report what would be written without writing",
)
@click.pass_context
def sidecar(
    ctx: click.Context,
    paths: List[Path],
    force: bool,
    description: Optional[str],
    keywords: Optional[str],
    dry_run: bool,
) -> None:
    """Write XMP sidecar files for PATH(s)."""
    kw = [k.strip() for k in keywords.split(",")] if keywords else []
    total = 0
    written = 0

    for p in paths:
        p = p.resolve()
        if dry_run:
            click.echo(f"  would-write: {p.name}")
            total += 1
            continue
        status = write_xmp_sidecar(
            p,
            description=description,
            keywords=kw,
            force=force,
        )
        total += 1
        if status == "written":
            written += 1
        click.echo(f"  {status}: {p.name}")

    click.echo(f"\nSidecars: {written}/{total} written")


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------
@main.command(name="list")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "csv", "table"]),
    default="table",
    help="Output format (default: table)",
)
@click.option(
    "--limit",
    type=int,
    default=100,
    show_default=True,
    help="Maximum number of assets to show",
)
@click.option(
    "--offset",
    type=int,
    default=0,
    show_default=True,
    help="Number of assets to skip",
)
@click.pass_context
def list_assets(ctx: click.Context, fmt: str, limit: int, offset: int) -> None:
    """List indexed assets."""

    db: PhotoIndexDB = ctx.obj["db"]
    assets = db.list_all(limit=limit, offset=offset)

    if fmt == "json":
        click.echo(json.dumps([
            {
                "id": str(a.id),
                "path": a.path,
                "filename": a.filename,
                "content_sha256": a.content_sha256,
                "file_size": a.file_size,
                "mime_type": a.mime_type,
                "status": str(a.status),
                "indexed_at": a.indexed_at.isoformat() if a.indexed_at else None,
                "enriched_at": a.enriched_at.isoformat() if a.enriched_at else None,
            }
            for a in assets
        ], indent=2))
    elif fmt == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(["id", "path", "filename", "content_sha256", "file_size", "mime_type", "status"])
        for a in assets:
            writer.writerow([
                str(a.id), a.path, a.filename, a.content_sha256,
                str(a.file_size or ""), a.mime_type or "", str(a.status),
            ])
    else:
        click.echo(f"{'ID':<38} {'Filename':<40} {'Status':<12}")
        click.echo("-" * 92)
        for a in assets:
            click.echo(f"{str(a.id):<38} {a.filename:<40} {str(a.status):<12}")


# ---------------------------------------------------------------------------
# taxonomies command
# ---------------------------------------------------------------------------
@main.command()
def taxonomies() -> None:
    """List available taxonomies and their entries."""
    for name in list_taxonomies():
        t = ALL_TAXONOMIES[name]
        click.echo(f"## {name} ({t.description})")
        for entry in t.entries:
            syn = f" (syn: {', '.join(entry.synonyms)})" if entry.synonyms else ""
            click.echo(f"  - {entry.label}{syn}")


# ---------------------------------------------------------------------------
# config command
# ---------------------------------------------------------------------------
@main.command()
@click.option(
    "--check-ollama",
    is_flag=True,
    default=False,
    help="Check if Ollama is reachable",
)
@click.pass_context
def config(ctx: click.Context, check_ollama: bool) -> None:
    """Show configuration and environment."""
    count = ctx.obj["db"].count()
    click.echo(f"DB path: {ctx.obj['db_path']}")
    click.echo(f"DB records:     {count}")
    click.echo(f"VLM model: {_default_vlm_model()}")
    click.echo(f"VLM URL: {_default_vlm_url()}")
    click.echo(f"Staging: {_default_staging()}")
    click.echo(f"Library: {_default_library()}")
    click.echo(f"Taxonomies: {', '.join(list_taxonomies())}")

    if check_ollama:
        vlm = VLMClient(base_url=_default_vlm_url(), model=_default_vlm_model())
        if vlm.is_available():
            click.echo(f"Ollama: reachable at {_default_vlm_url()}")
        else:
            click.echo(f"Ollama: NOT reachable at {_default_vlm_url()}", err=True)


# ---------------------------------------------------------------------------
# tag command
# ---------------------------------------------------------------------------
@main.command()
@click.option(
    "--asset-id",
    required=True,
    help="Asset ID to tag",
)
@click.option(
    "--add",
    "add_tag",
    default=None,
    help="Tag to add",
)
@click.option(
    "--remove",
    "remove_tag",
    default=None,
    help="Tag to remove",
)
@click.option(
    "--list",
    "list_tags",
    is_flag=True,
    default=False,
    help="List all tags for the asset",
)
@click.pass_context
def tag(ctx: click.Context, asset_id: str, add_tag: Optional[str], remove_tag: Optional[str], list_tags: bool) -> None:
    """Add, remove, or list tags for an asset."""
    db: PhotoIndexDB = ctx.obj["db"]
    asset = db.get(asset_id)
    if not asset:
        click.echo(f"Asset not found: {asset_id}", err=True)
        sys.exit(1)
    
    if list_tags or (add_tag is None and remove_tag is None):
        tags = db.get_tags(asset_id)
        if tags:
            click.echo(f"Tags for {asset.filename}:")
            for tag in tags:
                click.echo(f"  {tag}")
        else:
            click.echo(f"No tags for {asset.filename}")
    
    if add_tag:
        if db.add_tag(asset_id, add_tag):
            click.echo(f"Added tag '{add_tag}' to {asset.filename}")
        else:
            click.echo(f"Tag '{add_tag}' already exists or asset not found")
    
    if remove_tag:
        if db.remove_tag(asset_id, remove_tag):
            click.echo(f"Removed tag '{remove_tag}' from {asset.filename}")
        else:
            click.echo(f"Tag '{remove_tag}' not found")


# ---------------------------------------------------------------------------
# tags command (find assets by tag)
# ---------------------------------------------------------------------------
@main.command()
@click.option(
    "--tag",
    required=True,
    help="Tag name to search for",
)
@click.option(
    "--limit",
    type=int,
    default=100,
    show_default=True,
    help="Maximum number of assets to show",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "csv", "table"]),
    default="table",
    help="Output format (default: table)",
)
@click.pass_context
def tags(ctx: click.Context, tag: str, limit: int, fmt: str) -> None:
    """Find assets by tag."""
    db: PhotoIndexDB = ctx.obj["db"]
    assets = db.find_assets_by_tag(tag, limit=limit)
    
    if fmt == "json":
        click.echo(json.dumps([
            {"id": str(a.id), "filename": a.filename, "path": a.path}
            for a in assets
        ], indent=2))
    elif fmt == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(["id", "filename", "path"])
        for a in assets:
            writer.writerow([str(a.id), a.filename, a.path])
    else:
        click.echo(f"Assets with tag '{tag}':")
        for a in assets:
            click.echo(f"  {str(a.id):<38} {a.filename:<40} {a.path}")


# ---------------------------------------------------------------------------
# tag-import command
# ---------------------------------------------------------------------------
@main.command()
@click.option(
    "--file",
    "file_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to tags file (TSV/CSV: asset_id\\ttag1,tag2)",
)
@click.pass_context
def tag_import(ctx: click.Context, file_path: Path) -> None:
    """Bulk import tags from a file."""
    db: PhotoIndexDB = ctx.obj["db"]
    count = db.import_tags_from_file(str(file_path))
    click.echo(f"Imported {count} tags from {file_path}")


# ---------------------------------------------------------------------------
# person command (face recognition)
# ---------------------------------------------------------------------------
@main.command()
@click.option(
    "--create",
    default=None,
    help="Create a new known person with this name",
)
@click.option(
    "--list",
    "list_persons",
    is_flag=True,
    default=False,
    help="List all known persons",
)
@click.option(
    "--get",
    default=None,
    help="Get person details by name",
)
@click.option(
    "--embedding-file",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="JSON file with face embedding (required for --create)",
)
@click.pass_context
def person(ctx: click.Context, create: Optional[str], list_persons: bool, get: Optional[str], embedding_file: Optional[Path]) -> None:
    """Manage known persons for facial recognition."""
    db: PhotoIndexDB = ctx.obj["db"]
    
    if create:
        if not embedding_file:
            click.echo("--embedding-file required with --create", err=True)
            sys.exit(1)
        import json
        with open(embedding_file) as f:
            embedding = json.load(f)
        if not isinstance(embedding, list):
            click.echo("Embedding file must contain a JSON array of floats", err=True)
            sys.exit(1)
        person_id = db.create_person(create, embedding)
        click.echo(f"Created person '{create}' with ID: {person_id}")
    
    elif list_persons:
        persons = db.list_persons()
        if persons:
            click.echo("Known persons:")
            for p in persons:
                click.echo(f"  {p['name']} (ID: {p['id']}, created: {p['created_at']})")
        else:
            click.echo("No known persons")
    
    elif get:
        person = db.get_person_by_name(get)
        if person:
            click.echo(f"Person: {person['name']}")
            click.echo(f"  ID: {person['id']}")
            click.echo(f"  Created: {person['created_at']}")
        else:
            click.echo(f"Person not found: {get}", err=True)


# ---------------------------------------------------------------------------
# match-faces command
# ---------------------------------------------------------------------------
@main.command()
@click.option(
    "--asset-id",
    default=None,
    help="Match faces in a specific asset",
)
@click.option(
    "--all",
    "match_all",
    is_flag=True,
    default=False,
    help="Match faces in all assets",
)
@click.option(
    "--threshold",
    type=float,
    default=0.6,
    show_default=True,
    help="Similarity threshold (0-1)",
)
@click.pass_context
def match_faces(ctx: click.Context, asset_id: Optional[str], match_all: bool, threshold: float) -> None:
    """Match detected faces to known persons."""
    db: PhotoIndexDB = ctx.obj["db"]
    
    if asset_id:
        matches = db.match_faces_in_asset(asset_id, threshold=threshold)
        if matches:
            for m in matches:
                click.echo(f"Face {m['face_id'][:8]} -> {m['person_name']} (similarity: {m['similarity']:.3f})")
        else:
            click.echo("No matches found")
    
    elif match_all:
        # Get all assets with faces
        assets = db.list_all(limit=10000)
        total_matches = 0
        for a in assets:
            matches = db.match_faces_in_asset(str(a.id), threshold=threshold)
            for m in matches:
                db.link_face_to_person(m['face_id'], m['person_id'], m['similarity'])
                total_matches += 1
        click.echo(f"Matched {total_matches} faces to known persons")


# ---------------------------------------------------------------------------
# person-faces command
# ---------------------------------------------------------------------------
@main.command()
@click.option(
    "--name",
    required=True,
    help="Person name to find faces for",
)
@click.pass_context
def person_faces(ctx: click.Context, name: str) -> None:
    """Show all faces matched to a specific person."""
    db: PhotoIndexDB = ctx.obj["db"]
    person = db.get_person_by_name(name)
    if not person:
        click.echo(f"Person not found: {name}", err=True)
        sys.exit(1)
    
    faces = db.get_faces_by_person(person["id"])
    if faces:
        click.echo(f"Faces for {name}:")
        for f in faces:
            click.echo(f"  Face {f['id'][:8]} in asset {f['asset_id'][:8]} (similarity: {f['similarity']:.3f}, matched: {f['matched_at']})")
    else:
        click.echo(f"No faces matched to {name}")

# ---------------------------------------------------------------------------
# body-pose command
# ---------------------------------------------------------------------------
@main.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--model-complexity",
    type=click.IntRange(0, 2),
    default=1,
    help="MediaPipe Pose model complexity (0=light, 1=full, 2=heavy)",
)
@click.option(
    "--min-confidence",
    type=float,
    default=0.5,
    help="Minimum detection confidence",
)
@click.option(
    "--reference-height",
    type=float,
    default=None,
    help="Known height in cm for absolute measurement scaling",
)
@click.option(
    "--store/--no-store",
    default=True,
    help="Store results in database",
)
@click.pass_context
def body_pose(ctx: click.Context, paths: List[Path], model_complexity: int, min_confidence: float,
             reference_height: Optional[float], store: bool) -> None:
    """Detect body poses and compute measurements from images."""
    from .body import BodyDetector, BodyDetectorConfig, compute_measurements_from_pose
    from .database import PhotoIndexDB
    from .models import deterministic_asset_id
    import uuid
    
    db = ctx.obj["db"] if store else None
    config = BodyDetectorConfig(
        model_complexity=model_complexity,
        min_detection_confidence=min_confidence,
    )
    detector = BodyDetector(config)
    
    total_poses = 0
    total_measurements = 0
    
    for path in paths:
        click.echo(f"Processing {path}...")
        pose = detector.detect_pose(path)
        if pose:
            total_poses += 1
            click.echo(f"  Pose detected (score: {pose.det_score:.3f})")
            
            # Get or compute deterministic asset_id
            asset_id = None
            if store and db:
                # Try to find existing asset by path
                with db._connect() as conn:
                    row = conn.execute(
                        "SELECT asset_id FROM enrich_assets WHERE file_path = ?",
                        (str(path),)
                    ).fetchone()
                    if row:
                        asset_id = row["asset_id"]
                    else:
                        asset_id = str(deterministic_asset_id(str(path)))
                pose.asset_id = asset_id
            measurements = compute_measurements_from_pose(pose, reference_height)
            if measurements:
                click.echo(f"  Computed {len(measurements)} measurements:")
                for m in measurements:
                    click.echo(f"    {m.measurement_type}: {m.value} {m.unit} (conf: {m.confidence:.2f})")
                    if store and db:
                        db.upsert_body_measurement({
                            "asset_id": asset_id,
                            "measurement_type": m.measurement_type,
                            "value": m.value,
                            "unit": m.unit,
                            "confidence": m.confidence,
                            "method": m.method,
                            "reference_scale": m.reference_scale,
                            "landmarks_used": m.landmarks_used,
                            "notes": m.notes,
                        })
                        total_measurements += 1
            
            if store and db:
                db.upsert_body_pose(asset_id, {
                    "landmark_set": pose.landmark_set,
                    "landmarks_2d": pose.landmarks_2d,
                    "landmarks_3d": pose.landmarks_3d,
                    "bbox": pose.bbox,
                    "det_score": pose.det_score,
                    "model_name": pose.model_name,
                })
        else:
            click.echo(f"  No pose detected")
    
    click.echo(f"\nProcessed {len(paths)} images: {total_poses} poses, {total_measurements} measurements stored")


# ---------------------------------------------------------------------------
# measurements command
# ---------------------------------------------------------------------------
@main.command()
@click.option(
    "--asset-id",
    default=None,
    help="Show measurements for a specific asset",
)
@click.option(
    "--performer-id",
    default=None,
    help="Show measurements for a specific performer",
)
@click.option(
    "--type",
    "measurement_type",
    default=None,
    help="Filter by measurement type",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format",
)
@click.pass_context
def measurements(ctx: click.Context, asset_id: Optional[str], performer_id: Optional[str],
                 measurement_type: Optional[str], fmt: str) -> None:
    """List body measurements."""
    db: PhotoIndexDB = ctx.obj["db"]
    
    results = db.get_body_measurements(
        asset_id=asset_id,
        performer_id=performer_id,
        measurement_type=measurement_type,
    )
    
    if not results:
        click.echo("No measurements found")
        return
    
    if fmt == "table":
        click.echo(f"{'ID':<38} {'Asset':<38} {'Performer':<38} {'Type':<25} {'Value':>10} {'Unit':<6} {'Conf':>6}")
        click.echo("-" * 170)
        for r in results:
            aid = r["asset_id"][:36] if r["asset_id"] else "N/A"
            pid = r["performer_id"][:36] if r["performer_id"] else "N/A"
            click.echo(f"{r['id'][:36]:<38} {aid:<38} {pid:<38} {r['measurement_type']:<25} {r['value']:>10.1f} {r['unit']:<6} {r['confidence']:>6.2f}")
    elif fmt == "json":
        import json
        click.echo(json.dumps(results, indent=2, default=str))
    elif fmt == "csv":
        click.echo("id,asset_id,performer_id,measurement_type,value,unit,confidence,method,reference_scale,computed_at")
        for r in results:
            click.echo(f"{r['id']},{r['asset_id'] or ''},{r['performer_id'] or ''},{r['measurement_type']},{r['value']},{r['unit']},{r['confidence']},{r['method']},{r['reference_scale'] or ''},{r['computed_at'] or ''}")


# ---------------------------------------------------------------------------
# performer command (extended from person)
# ---------------------------------------------------------------------------
@main.command()
@click.option(
    "--create",
    default=None,
    help="Create a new performer with this name",
)
@click.option(
    "--list",
    "list_performers",
    is_flag=True,
    default=False,
    help="List all performers",
)
@click.option(
    "--get",
    default=None,
    help="Get performer details by name",
)
@click.option(
    "--update",
    default=None,
    help="Update performer (provide performer ID)",
)
@click.option(
    "--face-embedding",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="JSON file with face embedding (for --create)",
)
@click.option(
    "--aliases",
    default=None,
    help="Comma-separated aliases (for --create)",
)
@click.option(
    "--reference-height",
    type=float,
    default=None,
    help="Known height in cm (for --create/--update)",
)
@click.option(
    "--body-type",
    default=None,
    type=click.Choice(["ectomorph", "mesomorph", "endomorph", "unknown"]),
    help="Body type classification",
)
@click.option(
    "--notes",
    default=None,
    help="Notes about performer",
)
@click.pass_context
def performer(ctx: click.Context, create: Optional[str], list_performers: bool, get: Optional[str],
              update: Optional[str], face_embedding: Optional[Path], aliases: Optional[str],
              reference_height: Optional[float], body_type: Optional[str], notes: Optional[str]) -> None:
    """Manage performer profiles."""
    db: PhotoIndexDB = ctx.obj["db"]
    
    if create:
        embedding = None
        if face_embedding:
            import json
            with open(face_embedding) as f:
                embedding = json.load(f)
        
        alias_list = [a.strip() for a in aliases.split(",")] if aliases else []
        
        performer_id = db.create_performer(
            name=create,
            aliases=alias_list,
            face_embedding=embedding,
            notes=notes,
        )
        
        if reference_height:
            db.update_performer(performer_id, {"height_cm": reference_height})
        if body_type:
            db.update_performer(performer_id, {"body_type": body_type})
        
        click.echo(f"Created performer '{create}' with ID: {performer_id}")
    
    elif list_performers:
        performers = db.list_performers()
        if performers:
            click.echo("Performers:")
            for p in performers:
                click.echo(f"  {p['name']} (ID: {p['id'][:8]}, assets: {p['asset_count']}, "
                          f"height: {p['height_cm'] or 'N/A'}cm, body_type: {p['body_type'] or 'N/A'})")
        else:
            click.echo("No performers")
    
    elif get:
        performer = db.get_performer_by_name(get)
        if performer:
            click.echo(f"Performer: {performer['name']}")
            click.echo(f"  ID: {performer['id']}")
            click.echo(f"  Aliases: {performer['aliases_json']}")
            click.echo(f"  Face embedding quality: {performer['face_embedding_quality']:.3f}")
            click.echo(f"  Height: {performer['height_cm'] or 'N/A'}cm (conf: {performer['height_confidence']:.2f})")
            click.echo(f"  Body type: {performer['body_type'] or 'N/A'}")
            click.echo(f"  Asset count: {performer['asset_count']}")
            click.echo(f"  First seen: {performer['first_seen'] or 'N/A'}")
            click.echo(f"  Last seen: {performer['last_seen'] or 'N/A'}")
            click.echo(f"  Source tags: {performer['source_tags_json']}")
            click.echo(f"  Notes: {performer['notes'] or 'N/A'}")
            
            # Show aggregated measurements
            if performer['measurements_json']:
                import json
                meas = json.loads(performer['measurements_json'])
                if meas:
                    click.echo("\n  Aggregated measurements:")
                    for mtype, stats in meas.items():
                        click.echo(f"    {mtype}: mean={stats.get('mean',0):.1f} std={stats.get('std',0):.1f} "
                                  f"[{stats.get('min',0):.1f}-{stats.get('max',0):.1f}] count={stats.get('count',0)}")
        else:
            click.echo(f"Performer not found: {get}", err=True)
    
    elif update:
        updates = {}
        if reference_height:
            updates["height_cm"] = reference_height
        if body_type:
            updates["body_type"] = body_type
        if notes:
            updates["notes"] = notes
        if aliases:
            updates["aliases_json"] = json.dumps([a.strip() for a in aliases.split(",")])
        
        if updates:
            if db.update_performer(update, updates):
                click.echo(f"Updated performer {update}")
            else:
                click.echo(f"Performer not found: {update}", err=True)
        else:
            click.echo("No updates specified", err=True)


# ---------------------------------------------------------------------------
# build-profile command
# ---------------------------------------------------------------------------
@main.command()
@click.option(
    "--name",
    required=True,
    help="Performer name",
)
@click.option(
    "--assets",
    required=True,
    help="Comma-separated asset IDs to build profile from",
)
@click.option(
    "--aliases",
    default=None,
    help="Comma-separated aliases",
)
@click.option(
    "--reference-height",
    type=float,
    default=None,
    help="Known height in cm for measurement scaling",
)
@click.option(
    "--store/--no-store",
    default=True,
    help="Store profile in database",
)
@click.pass_context
def build_profile(ctx: click.Context, name: str, assets: str, aliases: Optional[str],
                  reference_height: Optional[float], store: bool) -> None:
    """Build a performer profile from multiple assets."""
    from .performers import PerformerAggregator, AggregationConfig
    from uuid import UUID
    
    db: PhotoIndexDB = ctx.obj["db"]
    asset_ids = [a.strip() for a in assets.split(",")]
    alias_list = [a.strip() for a in aliases.split(",")] if aliases else []
    
    config = AggregationConfig(
        min_images_for_profile=1,
        require_face_match=False,
    )
    aggregator = PerformerAggregator(db, config)
    
    click.echo(f"Building profile for '{name}' from {len(asset_ids)} assets...")
    
    try:
        profile = aggregator.build_performer_profile(
            name=name,
            asset_ids=asset_ids,
            aliases=alias_list,
            reference_height_cm=reference_height,
        )
        
        click.echo(f"Profile built successfully!")
        click.echo(f"  ID: {profile.id}")
        click.echo(f"  Face embedding quality: {profile.face_embedding_quality:.3f}")
        height_str = f"{profile.height_cm:.1f}cm" if profile.height_cm is not None else "N/A"
        height_conf_str = f"{profile.height_confidence:.2f}" if profile.height_confidence is not None else "N/A"
        click.echo(f"  Height: {height_str} (conf: {height_conf_str})")
        click.echo(f"  Body type: {profile.body_type}")
        click.echo(f"  Measurements: {len(profile.measurements)} types")
        
        for mtype, stats in profile.measurements.items():
            click.echo(f"    {mtype}: {stats['mean']:.1f}±{stats['std']:.1f} {stats.get('unit','cm')}")
        
        if store:
            # Store performer
            perf_id = db.create_performer(
                name=profile.name,
                aliases=profile.aliases,
                face_embedding=profile.face_embedding,
                face_embedding_quality=profile.face_embedding_quality,
                notes=profile.notes,
            )
            # Update with aggregated data
            import json
            db.update_performer(perf_id, {
                "measurements_json": json.dumps(profile.measurements),
                "measurement_confidence_json": json.dumps(profile.measurement_confidence),
                "body_type": profile.body_type,
                "height_cm": profile.height_cm,
                "height_confidence": profile.height_confidence,
                "asset_count": profile.asset_count,
                "first_seen": profile.first_seen.isoformat() if profile.first_seen else None,
                "last_seen": profile.last_seen.isoformat() if profile.last_seen else None,
            })
            
            # Link assets
            for asset_id in asset_ids:
                db.link_performer_to_asset(perf_id, asset_id, match_method="manual", match_score=1.0)
            
            click.echo(f"\nStored as performer ID: {perf_id}")
    
    except Exception as e:
        click.echo(f"Error building profile: {e}", err=True)


# ---------------------------------------------------------------------------
# facial-biometrics command
# ---------------------------------------------------------------------------
@main.command()
@click.option(
    "--performer",
    required=True,
    help="Performer name or ID",
)
@click.option(
    "--assets",
    required=True,
    help="Comma-separated asset IDs for biometrics",
)
@click.option(
    "--store/--no-store",
    default=True,
    help="Store biometrics in database",
)
@click.pass_context
def facial_biometrics(ctx: click.Context, performer: str, assets: str, store: bool) -> None:
    """Build facial biometrics for a performer from multiple assets."""
    from .performers import PerformerAggregator, AggregationConfig
    
    db: PhotoIndexDB = ctx.obj["db"]
    
    # Resolve performer
    perf = db.get_performer_by_name(performer)
    if not perf:
        perf = db.get_performer(performer)
    if not perf:
        click.echo(f"Performer not found: {performer}", err=True)
        return
    
    performer_id = perf["id"]
    asset_ids = [a.strip() for a in assets.split(",")]
    
    config = AggregationConfig()
    aggregator = PerformerAggregator(db, config)
    
    click.echo(f"Building facial biometrics for '{perf['name']}' from {len(asset_ids)} assets...")
    
    try:
        biometrics = aggregator.build_facial_biometrics(performer_id, asset_ids)
        click.echo(f"Facial biometrics built successfully!")
        click.echo(f"  ID: {biometrics.id}")
        click.echo(f"  Source images: {biometrics.source_image_count}")
        click.echo(f"  Identity quality: {biometrics.identity_quality_mean:.3f}")
        click.echo(f"  Canonical landmarks 68: {'Yes' if biometrics.canonical_landmarks_68 else 'No'}")
        click.echo(f"  Canonical landmarks 468: {'Yes' if biometrics.canonical_landmarks_468 else 'No'}")
        click.echo(f"  Facial measurements: {len(biometrics.facial_measurements)} types")
        click.echo(f"  Pose coverage: {biometrics.pose_coverage}")
        
        if store:
            bio_id = db.upsert_facial_biometrics({
                "performer_id": performer_id,
                "identity_embedding": biometrics.identity_embedding,
                "identity_embedding_std": biometrics.identity_embedding_std,
                "identity_quality_mean": biometrics.identity_quality_mean,
                "canonical_landmarks_68": biometrics.canonical_landmarks_68,
                "canonical_landmarks_68_cov": biometrics.canonical_landmarks_68_cov,
                "canonical_landmarks_468": biometrics.canonical_landmarks_468,
                "facial_measurements": biometrics.facial_measurements,
                "source_image_count": biometrics.source_image_count,
                "pose_coverage": biometrics.pose_coverage,
                "expression_coverage": biometrics.expression_coverage,
                "model_versions": biometrics.model_versions,
            })
            
            # Link to performer
            db.update_performer(performer_id, {"facial_biometrics_id": bio_id})
            
            click.echo(f"\nStored as facial biometrics ID: {bio_id}")
    
    except Exception as e:
        click.echo(f"Error building facial biometrics: {e}", err=True)


# ---------------------------------------------------------------------------
# search-performers command
# ---------------------------------------------------------------------------
@main.command()
@click.option(
    "--face-embedding",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="JSON file with face embedding to search for",
)
@click.option(
    "--threshold",
    type=float,
    default=0.6,
    help="Similarity threshold",
)
@click.option(
    "--limit",
    type=int,
    default=10,
    help="Max results",
)
@click.pass_context
def search_performers(ctx: click.Context, face_embedding: Path, threshold: float, limit: int) -> None:
    """Search performers by face embedding similarity."""
    import json
    import math
    
    db: PhotoIndexDB = ctx.obj["db"]
    
    with open(face_embedding) as f:
        query_embedding = json.load(f)
    
    performers = db.list_performers()
    results = []
    
    for p in performers:
        if p.get("face_embedding_json"):
            p_embedding = json.loads(p["face_embedding_json"])
            # Cosine similarity
            dot = sum(a * b for a, b in zip(query_embedding, p_embedding))
            norm1 = math.sqrt(sum(a * a for a in query_embedding))
            norm2 = math.sqrt(sum(b * b for b in p_embedding))
            if norm1 > 0 and norm2 > 0:
                sim = dot / (norm1 * norm2)
                if sim >= threshold:
                    results.append((p, sim))
    
    results.sort(key=lambda x: x[1], reverse=True)
    results = results[:limit]
    
    if results:
        click.echo(f"Found {len(results)} matches:")
        for p, sim in results:
            click.echo(f"  {p['name']} (ID: {p['id'][:8]}) - similarity: {sim:.4f}")
    else:
        click.echo("No matches found")


# ---------------------------------------------------------------------------
# link-performer command
# ---------------------------------------------------------------------------
@main.command()
@click.option(
    "--performer",
    required=True,
    help="Performer name or ID",
)
@click.option(
    "--asset",
    required=True,
    help="Asset ID to link",
)
@click.option(
    "--method",
    default="face",
    type=click.Choice(["face", "metadata", "manual"]),
    help="Match method",
)
@click.option(
    "--score",
    type=float,
    default=None,
    help="Match score",
)
@click.pass_context
def link_performer(ctx: click.Context, performer: str, asset: str, method: str, score: Optional[float]) -> None:
    """Link a performer to an asset."""
    db: PhotoIndexDB = ctx.obj["db"]
    
    # Resolve performer
    perf = db.get_performer_by_name(performer)
    if not perf:
        perf = db.get_performer(performer)
    if not perf:
        click.echo(f"Performer not found: {performer}", err=True)
        return
    
    db.link_performer_to_asset(perf["id"], asset, match_method=method, match_score=score)
    click.echo(f"Linked performer '{perf['name']}' to asset {asset}")


# ---------------------------------------------------------------------------
# performer-assets command
# ---------------------------------------------------------------------------
@main.command()
@click.option(
    "--performer",
    required=True,
    help="Performer name or ID",
)
@click.pass_context
def performer_assets(ctx: click.Context, performer: str) -> None:
    """List all assets linked to a performer."""
    db: PhotoIndexDB = ctx.obj["db"]
    
    perf = db.get_performer_by_name(performer)
    if not perf:
        perf = db.get_performer(performer)
    if not perf:
        click.echo(f"Performer not found: {performer}", err=True)
        return
    
    assets = db.get_performer_assets(perf["id"])
    if assets:
        click.echo(f"Assets for '{perf['name']}':")
        for a in assets:
            click.echo(f"  {a['asset_id'][:36]} - {a['file_name']} (matched: {a['matched_at']}, "
                      f"method: {a['match_method']}, score: {a['match_score']})")
    else:
        click.echo(f"No assets linked to '{perf['name']}'")

if __name__ == "__main__":
    main()
