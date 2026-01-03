## Neural Network Training Script (`neural_net.py`)


### Purpose

Central training pipeline that consolidates raw CSV data from multiple bag segments into unified datasets, trains MLP models, and exports them to ONNX format for deployment.

### Key Capabilities

- ✅ **Multi-segment aggregation** - Combines data from multiple `seg_*/` directories
- ✅ **Multi-dataset fusion** - Merges data from different collection runs
- ✅ **Automatic preprocessing** - Handles feature extraction, scaling, and train/val splitting
- ✅ **ONNX export** - Converts trained models for C++ runtime inference
- ✅ **Metadata tracking** - Records dataset provenance and training parameters
- ✅ **Adaptive local goals** - Supports both fixed-spacing and curvature-aware local goals

---

## Usage

### Basic Training (Single Dataset)

Train on all segments within one bag directory:
```bash
python3 neural_net.py ~/ros_ws/ros_bag/readme_example --large --single_dkr
```

**Arguments:**
- `--large` - Use multi-segment aggregation mode
- `--single_dkr` - All training data is within one parent directory

**What it does:**
1. Scans for `seg_0/`, `seg_1/`, ... subdirectories
2. Loads `lidar_data.csv`, `odom_data.csv`, `local_goals.csv`, `cmd_vel_output.csv` from each
3. Combines into unified `combined_features.csv` and `combined_labels.csv`
4. Trains MLP model with MinMax scaling
5. Exports `.keras` and `.onnx` models with scaler files

---

### Multi-Dataset Training

Train on data from multiple collection runs:
```bash
python3 neural_net.py \
    ~/ros_ws/ros_bag/dataset1 \
    ~/ros_ws/ros_bag/dataset2 \
    ~/ros_ws/ros_bag/dataset3 \
    --large
```

**What it does:**
- Aggregates data across all specified directories
- Ensures feature dimensions match before combining
- Creates timestamped output directory with metadata

---

### Adaptive Local Goals

Use curvature-aware local goals instead of fixed-spacing:
```bash
python3 neural_net.py ~/ros_ws/ros_bag/readme_example --large --single_dkr --adaptive
```

**Effect:**
- Reads from `adaptive_local_goals.csv` instead of `local_goals.csv`
- Provides variable-spacing goal representations based on path curvature

---

## Model Architecture

**Network structure:**
```
Input Layer (1085 features)
    ↓
Dense(256, ReLU)
    ↓
Dense(256, ReLU)
    ↓
Dense(256, ReLU)
    ↓
Output Layer (2) → [linear_velocity, angular_velocity]
```

**Features (1085 total):**
- **Odometry:** `odom_v`, `odom_w` (2)
- **Local Goals:** `goal_x`, `goal_y`, `goal_yaw` (3)
- **LiDAR:** `lidar_0` ... `lidar_1079` (1080)

**Labels (2):**
- `cmd_v` - Linear velocity command (m/s)
- `cmd_w` - Angular velocity command (rad/s)

---

## Training Configuration

**Hyperparameters:**
```python
EPOCHS: 150
BATCH_SIZE: 256
OPTIMIZER: Adam with exponential decay
  - Initial learning rate: 1e-3
  - Decay steps: 10,000
  - Decay rate: 0.9
LOSS: Mean Squared Error (MSE)
VALIDATION_SPLIT: 20%
EARLY_STOPPING: Patience=5, min_delta=0.001
```

**Data preprocessing:**
- Scaler: `MinMaxScaler` (scales features to [0, 1] range)
- Missing rows: Automatically removed (LiDAR has 1 fewer row than commands)
- Minimum segment size: 200 samples (smaller segments skipped)

---

## Output Files

**Training produces these files in `data_set/<timestamp>/`:**
```
data_set/2026_01_02_14_30/
├── 2026_01_02_14_30.keras              # Keras model checkpoint
├── 2026_01_02_14_30.onnx               # ONNX runtime model
├── 2026_01_02_14_30_scaler_mins.txt    # MinMax scaler minimum values
├── 2026_01_02_14_30_scaler_maxs.txt    # MinMax scaler maximum values
├── graphs.png                          # Training/validation loss curves
└── metadata.yaml                       # Dataset provenance and shapes
```

**Metadata YAML structure:**
```yaml
run:
  timestamp: "2026_01_02_14_30"
  output_dir: "data_set/2026_01_02_14_30"

datasets:
  - name: "2026-01-01_12-40-05_gaus"
    path: "/home/mobrob/ros_ws/ros_bag/readme_example/2026-01-01_12-40-05_gaus"
    features_shape: {rows: 7542, cols: 1085}
    labels_shape: {rows: 7542, cols: 2}
    seg_directories_processed: 4
  
  - name: "2026-01-01_13-06-59_gaus"
    ...

combined:
  features_shape: {rows: 45128, cols: 1085}
  labels_shape: {rows: 45128, cols: 2}
  num_datasets: 6
  total_rows: 45128
```

---

## Workflow Examples

### Example 1: Quick Test (Single Small Dataset)
```bash
python3 neural_net.py ~/ros_ws/ros_bag/test_run --large --single_dkr
```

**Expected output:**
```
Found subdirectories: ['seg_0', 'seg_1', 'seg_2']
Shape of odom (1200, 2) lidar (1200, 1080) local goals (1200, 3)
Features shape for seg_0: (1200, 1085)
Combined features (3600, 1085) and labels (3600, 2)

Training MLP model...
Epoch 1/150: loss: 0.0456 - val_loss: 0.0312
Epoch 2/150: loss: 0.0234 - val_loss: 0.0198
...
Model saved to: data_set/2026_01_02_14_30/2026_01_02_14_30.keras
Converted data_set/2026_01_02_14_30/2026_01_02_14_30.onnx
```

---

### Example 2: Production Training (Multiple Datasets)
```bash
python3 neural_net.py \
    ~/ros_ws/ros_bag/gauss_2_part1 \
    ~/ros_ws/ros_bag/gauss_2_part2 \
    --large
```

**Expected output:**
```
Processing 2 directories
Files already exist in ~/ros_ws/ros_bag/gauss_2_part1
Combined features (75000, 1085) and labels (75000, 2)
Files already exist in ~/ros_ws/ros_bag/gauss_2_part2
Combined features (150000, 1085) and labels (150000, 2)

Final combined dataset: features (150000, 1085), labels (150000, 2)
Training MLP model...
Epoch 1/150: loss: 0.0198 - val_loss: 0.0156
...
```

---

### Example 3: Adaptive Goals (Curvature-Aware)
```bash
python3 neural_net.py ~/ros_ws/ros_bag/asymmetric_data --large --single_dkr --adaptive
```

**Effect:**
- Uses `adaptive_local_goals.csv` with variable spacing
- Better representation of curved paths
- Improved performance in tight corridors

---

## Data Processing Pipeline

**Step-by-step breakdown:**

1. **Discovery:**
   - Scan for `seg_*/input_data/` directories
   - Check for existing `combined_features.csv` (reuse if present)

2. **Per-segment loading:**
```python
   lidar_data.csv       → 1080 columns (ranges)
   odom_data.csv        → 2 columns (v, w)
   local_goals.csv      → 3 columns (x, y, yaw)
   cmd_vel_output.csv   → 2 columns (v_cmd, w_cmd)
```

3. **Preprocessing:**
   - Drop last LiDAR row (sync with commands)
   - Rename columns with prefixes: `odom_*`, `lidar_*`, `goal_*`
   - Filter segments with <200 samples

4. **Aggregation:**
   - Concatenate segments within each bag
   - Save `combined_features.csv` and `combined_labels.csv`
   - Merge across multiple bags (if specified)

5. **Training:**
   - 80/20 train/val split
   - MinMax scaling fitted on training set only
   - Adam optimizer with exponential learning rate decay

6. **Export:**
   - Save Keras `.h5` model
   - Convert to ONNX with `tf2onnx`
   - Export scaler min/max for C++ inference

---

## Scaler Files Explained

**Purpose:** C++ controller needs exact normalization parameters used during training.

**Format:**
```
# _scaler_mins.txt (1085 values, one per feature)
-0.234    # odom_v min
-1.456    # odom_w min
2.345     # goal_x min
...
0.164     # lidar_0 min (sensor minimum range)
...

# _scaler_maxs.txt (1085 values)
0.456     # odom_v max
1.234     # odom_w max
5.678     # goal_x max
...
12.0      # lidar_0 max (sensor maximum range)
...
```

**Usage in C++:**
```cpp
// Normalize input features at runtime
for (int i = 0; i < 1085; i++) {
    normalized[i] = (raw[i] - scaler_mins[i]) / (scaler_maxs[i] - scaler_mins[i]);
}
```

---

## Troubleshooting

**Problem:** "Feature dimension mismatch"
- **Cause:** Different segments have inconsistent LiDAR configurations
- **Solution:** Verify all bags use same sensor (1080 ranges), check for corrupted CSVs

**Problem:** "No seg_* directories found"
- **Cause:** Dataset generation step (Step 5) was skipped or failed
- **Solution:** Run `./training_script.sh` to generate segment data first

**Problem:** NaN loss during training
- **Cause:** Invalid values in features (inf, nan) or labels
- **Solution:** 
```bash
  # Check for problems in combined data
  python3 -c "import pandas as pd; import numpy as np; 
  df = pd.read_csv('combined_features.csv'); 
  print('NaNs:', df.isna().sum().sum()); 
  print('Infs:', np.isinf(df.select_dtypes(include=[np.number])).sum().sum())"
```

**Problem:** "Segment too small (X rows), skipping"
- **Cause:** Very short random walk produced <200 samples
- **Solution:** Normal behavior - segment is excluded from training

**Problem:** ONNX conversion fails
- **Cause:** TensorFlow/ONNX version mismatch
- **Solution:** Update dependencies:
```bash
  pip install --upgrade tf2onnx onnx tensorflow
```

---

## Advanced Options

**Command-line flags (full list):**

| Flag | Description |
|------|-------------|
| `--large` | Multi-segment/multi-dataset aggregation mode |
| `--single_dkr` | All data in one parent directory (scan for subdirs) |
| `--adaptive` | Use adaptive local goals (curvature-aware) |
| `--train` | *(deprecated)* Single-bag training mode |
| `--combine` | *(deprecated)* Combine datasets without training |
| `--train_combine` | *(deprecated)* Train on pre-combined dataset |

**Recommended workflow:**
- Always use `--large --single_dkr` for new datasets
- Add `--adaptive` if you generated adaptive local goals in dataset creation

---

## Integration with Pipeline

**Position in workflow:**
```
Dataset Generation (Step 5)
    ↓
→ neural_net.py (Step 6) ← YOU ARE HERE
    ↓
Model Deployment (Step 7)
    ↓
Testing (Steps 8-9)
```

**Inputs required:**
- `seg_*/input_data/*.csv` files from dataset generation
- At least 50k total samples recommended for good performance

**Outputs consumed by:**
- `multiple_scripts.sh` (reads `.onnx` and scaler files)
- ONNX controller C++ node (runtime inference)
