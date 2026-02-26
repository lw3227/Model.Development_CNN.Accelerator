#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import tensorflow as tf


def _to_cell(items: list[object]) -> np.ndarray:
    arr = np.empty((len(items), 1), dtype=object)
    for i, v in enumerate(items):
        arr[i, 0] = v
    return arr


def collect_param_tensor_indices(interpreter: tf.lite.Interpreter) -> list[int]:
    details = interpreter.get_tensor_details()
    all_indices = {d["index"] for d in details}
    model_inputs = {d["index"] for d in interpreter.get_input_details()}

    op_inputs: set[int] = set()
    op_outputs: set[int] = set()
    for op in interpreter._get_ops_details():
        op_inputs.update(i for i in op["inputs"] if i >= 0)
        op_outputs.update(i for i in op["outputs"] if i >= 0)

    candidates = (op_inputs - op_outputs - model_inputs) & all_indices
    return sorted(candidates)


def export_params_to_mat(model_path: Path, out_path: Path) -> None:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    tensor_details = {d["index"]: d for d in interpreter.get_tensor_details()}
    param_indices = collect_param_tensor_indices(interpreter)

    names: list[str] = []
    tensor_indices: list[int] = []
    dtypes: list[str] = []
    shapes: list[np.ndarray] = []
    shape_signatures: list[np.ndarray] = []
    quantized_dims: list[int] = []
    values: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    zero_points: list[np.ndarray] = []

    for idx in param_indices:
        d = tensor_details[idx]
        q = d["quantization_parameters"]
        v = interpreter.get_tensor(idx)

        names.append(d["name"])
        tensor_indices.append(int(idx))
        dtypes.append(str(v.dtype))
        shapes.append(np.asarray(d["shape"], dtype=np.int64))
        shape_signatures.append(np.asarray(d["shape_signature"], dtype=np.int64))
        quantized_dims.append(int(q["quantized_dimension"]))
        values.append(v)
        scales.append(np.asarray(q["scales"], dtype=np.float32))
        zero_points.append(np.asarray(q["zero_points"], dtype=np.int32))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(
        str(out_path),
        {
            "model_path": str(model_path),
            "num_params": np.int32(len(param_indices)),
            "names": _to_cell(names),
            "tensor_indices": np.asarray(tensor_indices, dtype=np.int32).reshape(-1, 1),
            "dtypes": _to_cell(dtypes),
            "shapes": _to_cell(shapes),
            "shape_signatures": _to_cell(shape_signatures),
            "quantized_dimensions": np.asarray(quantized_dims, dtype=np.int32).reshape(-1, 1),
            "values": _to_cell(values),
            "scales": _to_cell(scales),
            "zero_points": _to_cell(zero_points),
        },
        do_compression=True,
    )

    print(f"[OK] Exported {len(param_indices)} parameter tensors")
    print(f"[OK] Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export int8 TFLite parameter tensors to a MATLAB .mat file."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/v2.int8.tflite"),
        help="Path to .tflite model",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/v2.int8.params.mat"),
        help="Output .mat path",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    export_params_to_mat(args.model.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
