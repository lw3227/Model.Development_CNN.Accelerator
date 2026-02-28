# CNN_int8_v1

这是一个 Rock / Paper / Scissors 手势分类项目，核心目标是：
- 训练并导出全整型 `int8` TFLite 模型
- 在 Python 侧验证精度
- 在 MATLAB 侧逐层复现并与 TFLite 做 golden 对齐（已到 Dense 对齐）

## 目录结构
- `wlx_train_cnn.ipynb`: 训练、量化导出、张量检查
- `wlx_test_cnn.ipynb`: int8 模型验证（accuracy / per-class / confusion matrix）
- `export_tflite_params_mat.py`: 导出 TFLite 参数到 `.mat`
- `models/`: `.tflite`、`.mat` 和模型说明
- `matlab/`: MATLAB int8 推理与逐层对齐脚本

## 环境
建议使用两个环境：
- Python 训练/推理：`.venv_tflite`
- MATLAB：R2024+（需能运行 `readNPY/writeNPY`）

Python 依赖安装：
```bash
python3 -m venv .venv_tflite
source .venv_tflite/bin/activate
pip install -r requirements.txt
```

## Python 工作流
### 1) 训练并导出 int8 模型
使用 `wlx_train_cnn.ipynb`：
- 训练 CNN
- 导出 `models/*.int8.tflite`
- 可导出参数到 `.mat`

### 2) 验证 int8 推理
使用 `wlx_test_cnn.ipynb`：
- 读取验证集
- 按模型输入量化参数做 `int8` 输入
- 统计整体准确率、分类别准确率、混淆矩阵

## MATLAB 工作流（golden 对齐）
当前脚本已实现：
- Conv1/2/3 + ReLU + MaxPool
- Flatten（NHWC 顺序）
- Fully Connected（int8->int32->requant int8）

关键文件：
- `matlab/rps_conv2_2.m`: 主流程
- `matlab/conv2D_int8.m`: 卷积
- `matlab/requant_int32_to_int8.m`: 量化重标定
- `matlab/flatten_nhwc_int8.m`: Flatten
- `matlab/fully_connected_int8.m`: FC

### 逐层对齐步骤
1. 先导出 TFLite 参考中间层：
```bash
.venv_tflite/bin/python matlab/dump_tflite_conv_acts.py \
  --model models/v2.int8.tflite \
  --image matlab/scissors_200_v1_test_1644.png \
  --outdir matlab/debug
```

2. 在 MATLAB 跑主流程：
```matlab
run('matlab/rps_conv2_2.m')
```

3. 在 MATLAB 对比误差：
```matlab
run('matlab/compare_tflite_matlab_layers.m')
```

对比项包含：
- `input_q`
- `conv1_relu`
- `conv2_relu`
- `conv3_relu`
- `dense_i8`

目前状态：
- `dense_i8` 可做到 `0 diff`
- conv 层存在极少量 `±1` rounding 边界差，属于正常量化误差

## 分类标签映射
类别顺序（Python 0-based）：
- `0 -> paper`
- `1 -> rock`
- `2 -> scissors`

MATLAB `max` 返回 1-based：
- `1 -> paper`
- `2 -> rock`
- `3 -> scissors`

示例：
```matlab
[~, idx] = max(double(out_fc_i8));
class_names = ["paper", "rock", "scissors"];
pred_label = class_names(idx);
```

## 备注
- `matlab/load_tflite_params_mat.m` 与根目录同名文件功能一致，优先使用 `matlab/` 下版本进行 MATLAB 流程。
- 若迁移到新仓库，建议保留 `models/models.txt` 记录模型版本与说明。
