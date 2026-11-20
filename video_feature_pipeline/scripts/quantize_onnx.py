#!/usr/bin/env python3
"""
ONNX Model Quantization Script

Quantizes InsightFace ONNX models to INT8 for faster inference.
Supports dynamic quantization (weights only) and static quantization (weights + activations).

Usage:
    python scripts/quantize_onnx.py --model buffalo_l --output ./models/quantized
    python scripts/quantize_onnx.py --model buffalo_s --output ./models/quantized --static --calibration-dir ./calibration_data
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnxruntime.quantization import (
    QuantType,
    quantize_dynamic,
    quantize_static,
    CalibrationDataReader,
)
from onnxruntime.quantization.onnx_model import ONNXModel

try:
    import insightface
    from insightface.model_zoo import get_model
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False


class ImageCalibrationReader(CalibrationDataReader):
    """Calibration data reader for static quantization using image files."""
    
    def __init__(self, calibration_dir: Path, input_name: str, input_shape: tuple[int, ...]):
        self.calibration_dir = calibration_dir
        self.input_name = input_name
        self.input_shape = input_shape
        self.image_files = list(calibration_dir.glob("*.jpg")) + list(calibration_dir.glob("*.png"))
        self.idx = 0
        self.batch_size = 1
        
    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.idx >= len(self.image_files):
            return None
        
        import cv2
        img_path = self.image_files[self.idx]
        img = cv2.imread(str(img_path))
        if img is None:
            self.idx += 1
            return self.get_next()
        
        # Resize to expected input shape
        h, w = self.input_shape[2], self.input_shape[3]
        img = cv2.resize(img, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        self.idx += 1
        return {self.input_name: img}


def find_insightface_models(model_pack: str = "buffalo_l") -> dict[str, Path]:
    """Find InsightFace ONNX model files."""
    if not INSIGHTFACE_AVAILABLE:
        raise RuntimeError("insightface not installed")
    
    # InsightFace models are stored in ~/.insightface/models/{pack_name}/
    home = Path.home()
    model_dir = home / ".insightface" / "models" / model_pack
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model pack {model_pack} not found at {model_dir}")
    
    models = {}
    for onnx_file in model_dir.glob("*.onnx"):
        models[onnx_file.stem] = onnx_file
    
    return models


def quantize_model_dynamic(
    input_path: Path,
    output_path: Path,
    weight_type: QuantType = QuantType.QInt8,
    per_channel: bool = True,
    reduce_range: bool = True,
) -> None:
    """Apply dynamic quantization to an ONNX model."""
    print(f"Dynamic quantizing {input_path.name} -> {output_path}")
    
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=weight_type,
        per_channel=per_channel,
        reduce_range=reduce_range,
    )
    print(f"  Done: {output_path}")


def quantize_model_static(
    input_path: Path,
    output_path: Path,
    calibration_reader: CalibrationDataReader,
    weight_type: QuantType = QuantType.QInt8,
    activation_type: QuantType = QuantType.QInt8,
    per_channel: bool = True,
    reduce_range: bool = True,
) -> None:
    """Apply static quantization to an ONNX model."""
    print(f"Static quantizing {input_path.name} -> {output_path}")
    
    quantize_static(
        model_input=str(input_path),
        model_output=str(output_path),
        calibration_data_reader=calibration_reader,
        weight_type=weight_type,
        activation_type=activation_type,
        per_channel=per_channel,
        reduce_range=reduce_range,
    )
    print(f"  Done: {output_path}")


def quantize_model_qdq(
    input_path: Path,
    output_path: Path,
    weight_type: QuantType = QuantType.QInt8,
    activation_type: QuantType = QuantType.QInt8,
) -> None:
    """Apply QDQ (Quantize-Dequantize) format quantization - recommended for ONNX Runtime."""
    from onnxruntime.quantization import quantize_qdq
    
    print(f"QDQ quantizing {input_path.name} -> {output_path}")
    
    quantize_qdq(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=weight_type,
        activation_type=activation_type,
    )
    print(f"  Done: {output_path}")


def verify_model(model_path: Path) -> bool:
    """Verify an ONNX model is valid."""
    try:
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        print(f"  ✓ Model verified: {model_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ Model verification failed: {e}")
        return False


def get_model_size(model_path: Path) -> int:
    """Get model file size in MB."""
    return model_path.stat().st_size / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Quantize InsightFace ONNX models")
    parser.add_argument(
        "--model", 
        default="buffalo_l", 
        choices=["buffalo_l", "buffalo_s", "buffalo_m"],
        help="InsightFace model pack to quantize"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=Path("./models/quantized"),
        help="Output directory for quantized models"
    )
    parser.add_argument(
        "--static", 
        action="store_true",
        help="Use static quantization (requires calibration data)"
    )
    parser.add_argument(
        "--qdq", 
        action="store_true",
        help="Use QDQ format (recommended for ORT)"
    )
    parser.add_argument(
        "--calibration-dir", 
        type=Path,
        help="Directory with calibration images (for static quantization)"
    )
    parser.add_argument(
        "--weight-type", 
        default="qint8",
        choices=["qint8", "quint8"],
        help="Weight quantization type"
    )
    parser.add_argument(
        "--verify", 
        action="store_true",
        help="Verify models after quantization"
    )
    parser.add_argument(
        "--models", 
        nargs="+",
        help="Specific models to quantize (default: all in pack)"
    )
    
    args = parser.parse_args()
    
    if not INSIGHTFACE_AVAILABLE:
        print("Error: insightface not installed. Run: pip install insightface")
        return 1
    
    # Find models
    try:
        models = find_insightface_models(args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run: python -c 'import insightface; insightface.app.FaceAnalysis(name=\"buffalo_l\").prepare(ctx_id=0)' to download")
        return 1
    
    print(f"Found models in {args.model}:")
    for name, path in models.items():
        print(f"  {name}: {path} ({get_model_size(path):.1f} MB)")
    
    # Filter models if specified
    if args.models:
        models = {k: v for k, v in models.items() if k in args.models}
        if not models:
            print(f"Error: None of the specified models found: {args.models}")
            return 1
    
    # Setup output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Determine quantization method
    weight_type = QuantType.QInt8 if args.weight_type == "qint8" else QuantType.QUInt8
    
    # For static quantization, need calibration data
    if args.static and not args.calibration_dir:
        print("Error: Static quantization requires --calibration-dir")
        return 1
    
    calibration_reader = None
    if args.static:
        calibration_reader = ImageCalibrationReader(
            args.calibration_dir,
            input_name="input",  # Common input name for InsightFace models
            input_shape=(1, 3, 640, 640),  # NCHW format
        )
    
    # Quantize each model
    results = {}
    for name, model_path in models.items():
        output_path = args.output / f"{name}_quantized.onnx"
        
        print(f"\n{'='*50}")
        print(f"Quantizing: {name}")
        print(f"{'='*50}")
        
        original_size = get_model_size(model_path)
        
        try:
            if args.qdq:
                quantize_model_qdq(model_path, output_path, weight_type=weight_type)
            elif args.static:
                quantize_model_static(
                    model_path, output_path, calibration_reader,
                    weight_type=weight_type
                )
            else:
                quantize_model_dynamic(model_path, output_path, weight_type=weight_type)
            
            quantized_size = get_model_size(output_path)
            reduction = (1 - quantized_size / original_size) * 100
            
            results[name] = {
                "original_mb": original_size,
                "quantized_mb": quantized_size,
                "reduction_pct": reduction,
                "path": output_path,
            }
            
            print(f"  Size: {original_size:.1f} MB -> {quantized_size:.1f} MB ({reduction:.1f}% reduction)")
            
            if args.verify:
                verify_model(output_path)
                
        except Exception as e:
            print(f"  Error quantizing {name}: {e}")
            results[name] = {"error": str(e)}
    
    # Summary
    print(f"\n{'='*50}")
    print("QUANTIZATION SUMMARY")
    print(f"{'='*50}")
    for name, result in results.items():
        if "error" in result:
            print(f"  {name}: FAILED - {result['error']}")
        else:
            print(f"  {name}: {result['original_mb']:.1f} MB -> {result['quantized_mb']:.1f} MB ({result['reduction_pct']:.1f}% reduction)")
    
    return 0


if __name__ == "__main__":
    exit(main())