#!/usr/bin/env python3
"""
Batch Embedding Optimization

Optimizes ChromaDB batch operations for large-scale embedding inserts.
Demonstrates efficient batching strategies for face and semantic embeddings.

Usage:
    python scripts/batch_embed.py --benchmark
    python scripts/batch_embed.py --batch-size 100 --count 10000
"""

from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from video_feature_pipeline.core.config import get_settings
from video_feature_pipeline.storage.vector_db import get_vector_db


def generate_random_embedding(dim: int = 512) -> np.ndarray:
    """Generate a random normalized embedding."""
    emb = np.random.randn(dim).astype(np.float32)
    return emb / np.linalg.norm(emb)


def generate_batch_embeddings(count: int, dim: int = 512) -> list[np.ndarray]:
    """Generate a batch of random embeddings."""
    embeddings = np.random.randn(count, dim).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / norms).tolist()


def benchmark_batch_sizes(vector_db, embedding_dim: int = 512, 
                          counts: list[int] = None, 
                          batch_sizes: list[int] = None) -> dict[str, Any]:
    """Benchmark different batch sizes for embedding insertion."""
    if counts is None:
        counts = [100, 500, 1000, 5000, 10000]
    if batch_sizes is None:
        batch_sizes = [10, 50, 100, 200, 500, 1000]
    
    results = {}
    
    for count in counts:
        print(f"\nBenchmarking {count} embeddings...")
        embeddings = generate_batch_embeddings(count, embedding_dim)
        metadatas = [{"video_id": f"test_{i}", "cluster_id": 0, "confidence": 0.9} for i in range(count)]
        
        results[count] = {}
        
        for batch_size in batch_sizes:
            if batch_size > count:
                continue
                
            # Clear collection for fair test
            vector_db.face_collection.delete(where={})
            
            start = time.perf_counter()
            
            # Batch insert
            for i in range(0, count, batch_size):
                batch_emb = embeddings[i:i+batch_size]
                batch_meta = metadatas[i:i+batch_size]
                batch_ids = [str(uuid.uuid4())[:16] for _ in range(len(batch_emb))]
                
                vector_db.face_collection.add(
                    ids=batch_ids,
                    embeddings=batch_emb,
                    metadatas=batch_meta,
                )
            
            elapsed = time.perf_counter() - start
            throughput = count / elapsed
            
            results[count][batch_size] = {
                "time_seconds": elapsed,
                "throughput_emb_per_sec": throughput,
            }
            
            print(f"  Batch size {batch_size:4d}: {elapsed:.3f}s ({throughput:,.0f} emb/s)")
    
    return results


def benchmark_single_vs_batch(vector_db, embedding_dim: int = 512, count: int = 5000) -> dict[str, Any]:
    """Compare single inserts vs batch inserts."""
    embeddings = generate_batch_embeddings(count, embedding_dim)
    metadatas = [{"video_id": f"test_{i}", "cluster_id": 0, "confidence": 0.9} for i in range(count)]
    
    # Single inserts
    vector_db.face_collection.delete(where={})
    start = time.perf_counter()
    for i in range(count):
        vector_db.face_collection.add(
            ids=[str(uuid.uuid4())[:16]],
            embeddings=[embeddings[i]],
            metadatas=[metadatas[i]],
        )
    single_time = time.perf_counter() - start
    
    # Batch inserts (optimal batch size)
    batch_size = 500
    vector_db.face_collection.delete(where={})
    start = time.perf_counter()
    for i in range(0, count, batch_size):
        batch_emb = embeddings[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids = [str(uuid.uuid4())[:16] for _ in range(len(batch_emb))]
        vector_db.face_collection.add(
            ids=batch_ids,
            embeddings=batch_emb,
            metadatas=batch_meta,
        )
    batch_time = time.perf_counter() - start
    
    return {
        "single_insert_time": single_time,
        "batch_insert_time": batch_time,
        "speedup": single_time / batch_time,
        "single_throughput": count / single_time,
        "batch_throughput": count / batch_time,
    }


def test_search_performance(vector_db, embedding_dim: int = 512, 
                           count: int = 10000, queries: int = 100) -> dict[str, Any]:
    """Test search performance with varying index sizes."""
    # Populate with embeddings
    vector_db.face_collection.delete(where={})
    embeddings = generate_batch_embeddings(count, embedding_dim)
    metadatas = [{"video_id": f"test_{i}", "cluster_id": i % 100, "confidence": 0.9} for i in range(count)]
    
    batch_size = 1000
    for i in range(0, count, batch_size):
        batch_emb = embeddings[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids = [str(uuid.uuid4())[:16] for _ in range(len(batch_emb))]
        vector_db.face_collection.add(
            ids=batch_ids,
            embeddings=batch_emb,
            metadatas=batch_meta,
        )
    
    # Search benchmark
    query_embeddings = [generate_random_embedding(embedding_dim) for _ in range(queries)]
    
    start = time.perf_counter()
    for q_emb in query_embeddings:
        vector_db.search_face_similar(q_emb, top_k=10, threshold=0.0)
    search_time = time.perf_counter() - start
    
    return {
        "index_size": count,
        "queries": queries,
        "total_search_time": search_time,
        "avg_query_time_ms": (search_time / queries) * 1000,
        "queries_per_second": queries / search_time,
    }


def optimize_batch_size(vector_db, embedding_dim: int = 512) -> dict[str, Any]:
    """Find optimal batch size for current system."""
    test_counts = [1000, 5000, 10000]
    batch_sizes = [10, 25, 50, 100, 200, 500, 1000]
    
    optimal = {}
    
    for count in test_counts:
        embeddings = generate_batch_embeddings(count, embedding_dim)
        metadatas = [{"video_id": f"test_{i}", "cluster_id": 0} for i in range(count)]
        
        best_time = float('inf')
        best_batch = None
        
        for batch_size in batch_sizes:
            if batch_size > count:
                continue
            
            vector_db.face_collection.delete(where={})
            
            start = time.perf_counter()
            for i in range(0, count, batch_size):
                batch_emb = embeddings[i:i+batch_size]
                batch_meta = metadatas[i:i+batch_size]
                batch_ids = [str(uuid.uuid4())[:16] for _ in range(len(batch_emb))]
                vector_db.face_collection.add(
                    ids=batch_ids,
                    embeddings=batch_emb,
                    metadatas=batch_meta,
                )
            elapsed = time.perf_counter() - start
            
            if elapsed < best_time:
                best_time = elapsed
                best_batch = batch_size
        
        optimal[count] = {
            "optimal_batch_size": best_batch,
            "best_time": best_time,
            "throughput": count / best_time,
        }
        print(f"  Count {count:6d}: Optimal batch size = {best_batch:4d} ({best_time:.3f}s, {count/best_time:,.0f} emb/s)")
    
    return optimal


def main():
    parser = argparse.ArgumentParser(description="Batch Embedding Optimization")
    parser.add_argument("--benchmark", action="store_true", help="Run full benchmark")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for testing")
    parser.add_argument("--count", type=int, default=5000, help="Number of embeddings to test")
    parser.add_argument("--queries", type=int, default=100, help="Number of search queries")
    parser.add_argument("--dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--output", type=Path, help="Output JSON file for results")
    
    args = parser.parse_args()
    
    print("Initializing Vector Database...")
    vector_db = get_vector_db()
    
    results = {}
    
    if args.benchmark:
        print("=" * 60)
        print("BATCH EMBEDDING OPTIMIZATION BENCHMARK")
        print("=" * 60)
        
        # 1. Batch size optimization
        print("\n1. Optimal Batch Size Search:")
        optimal = optimize_batch_size(vector_db, args.dim)
        results["optimal_batch_sizes"] = optimal
        
        # 2. Single vs Batch comparison
        print("\n2. Single vs Batch Insert Comparison:")
        comparison = benchmark_single_vs_batch(vector_db, args.dim, args.count)
        results["single_vs_batch"] = comparison
        print(f"  Single inserts: {comparison['single_insert_time']:.3f}s ({comparison['single_throughput']:,.0f} emb/s)")
        print(f"  Batch inserts:  {comparison['batch_insert_time']:.3f}s ({comparison['batch_throughput']:,.0f} emb/s)")
        print(f"  Speedup: {comparison['speedup']:.1f}x")
        
        # 3. Search performance
        print("\n3. Search Performance:")
        search_perf = test_search_performance(vector_db, args.dim, count=10000, queries=args.queries)
        results["search_performance"] = search_perf
        print(f"  Index size: {search_perf['index_size']:,}")
        print(f"  Queries: {search_perf['queries']}")
        print(f"  Avg query time: {search_perf['avg_query_time_ms']:.2f}ms")
        print(f"  QPS: {search_perf['queries_per_second']:.1f}")
        
        # 4. Detailed batch size benchmark
        print("\n4. Detailed Batch Size Benchmark:")
        benchmark_results = benchmark_batch_sizes(vector_db, args.dim)
        results["batch_size_benchmark"] = benchmark_results
    else:
        # Quick test
        print(f"Quick test: {args.count} embeddings, batch size {args.batch_size}")
        comparison = benchmark_single_vs_batch(vector_db, args.dim, args.count)
        results["single_vs_batch"] = comparison
        print(f"  Single: {comparison['single_insert_time']:.3f}s")
        print(f"  Batch:  {comparison['batch_insert_time']:.3f}s")
        print(f"  Speedup: {comparison['speedup']:.1f}x")
    
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()