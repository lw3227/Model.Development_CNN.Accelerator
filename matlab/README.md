# MATLAB INT8 Golden Flow

This directory reproduces TFLite INT8 inference layer by layer and aligns outputs against golden tensors.

## Main Entry Points

- `rps_conv2_2.m`: main pipeline (Conv/ReLU/Pool/Flatten/FC)
- `compare_tflite_matlab_layers.m`: compares MATLAB outputs with TFLite intermediates
- `dump_tflite_conv_acts.py`: exports TFLite intermediate activations

## Recommended Flow

1. Export TFLite intermediate activations from repository root:

```bash
.venv_tflite/bin/python matlab/dump_tflite_conv_acts.py \
  --model models/v2.int8.tflite \
  --image matlab/scissors_200_v1_test_1644.png \
  --outdir matlab/debug
```

2. Run in MATLAB:

```matlab
run('matlab/rps_conv2_2.m')
run('matlab/compare_tflite_matlab_layers.m')
```

## Notes

- `models/` stays at repository root and is shared across Python and MATLAB flows.
- `npy-matlab*` folders provide NPY read/write support.
