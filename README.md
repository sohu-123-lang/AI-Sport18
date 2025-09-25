# Learning to Anticipate: A Conditional Representation Fusion Network for Pre-Stroke Prediction 

## Demo Video
![Demo GIF](./TrackNetV2_together3-3-good3.gif)


## 🌟 Introduction
Our Conditional Gate-Based Cross-Fusion Network (ConFu) is a novel multimodal framework that predicts badminton shuttlecock landing positions **300ms before stroke execution** by integrating:
- 3D shuttlecock trajectory reconstruction
- Player dynamic localization 
- Keypoint-based arm gestures
- Stroke type classification

The model achieves **92.6% accuracy** (within 0.3m) and **97ms inference time**, enabling real-time tactical feedback.

## 🚀 Key Features
| Feature | Technical Innovation | Benefit |
|---------|----------------------|---------|
| **Multimodal Fusion** | Combines 4 data streams with cross-attention | 12.6% more accurate than unimodal baselines |
| **Conditional Gating** | Dynamic feature recalibration via LSTM | 23% faster inference by suppressing noise |
| **Spatio-Temporal Encoding** | Dual-branch transformer architecture | Captures both trajectory dynamics and court positioning |
| **Early Prediction** | Forecasts at stroke moment (not post-trajectory) | Provides 1.25s decision time advantage |

## 🔧 Data Processing

### Step 1: Raw Video to Features
Results are stored in this folder 1-PrecessedResultFromTrackNetV2

### Dataset Specifications
| Feature | Shape | Description |
|---------|-------|-------------|
| X1_3d | [N, 21, 3] | 3D shuttlecock positions (21 frames before stroke) |
| X2_team | [N, 4] | (x,y) coordinates of both players |
| X3_gesture | [N, 12, 20] | Arm keypoint differentials (6 keypoints × 2D) |
| X4_stroke | [N] | Stroke type (0:smash, 1:drive, 2:lift, 3:defensive) |
| y_labels | [N, 2] | Ground truth landing coordinates (x,y) |

## 🏋️ Training & Evaluation

### Training Configuration (`configs/train.yaml`)
```yaml
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 4000
  early_stopping:
    patience: 100
    delta: 0.001

model:
  hidden_dim: 128
  fusion_dim: 256
  num_heads: 4
  aux_loss_weight: 0.3
```

### Start Training
```bash
python 3-SourceCode-Train&Test/train.py \
  --data_dir 2-Data \
  --config configs/train.yaml \
  --log_dir runs/experiment_1
```

### Evaluation Metrics
```python
# MAE (Mean Absolute Error)
mae = torch.mean(torch.abs(predictions - targets))

# Accuracy@0.3m
accuracy = (torch.norm(predictions - targets, dim=1) < 0.3).float().mean()
```

## 📊 Results

### Performance Comparison
| Model | Accuracy@0.3m | MAE (m) | Inference Time (ms) |
|-------|--------------|---------|---------------------|
| DyMF | 83.2% | 0.28 | 623 |
| MonoTrack | 84.8% | 0.21 | 127 |
| FCST | 82.1% | 0.29 | 184 |
| **ConFu (Ours)** | **92.6%** | **0.20** | **97** |

## 📜 Citation
If you use this work in your research, please cite:
```bibtex
@article{confu202x,
  year={202x}
}
```

## 📄 License
This project is licensed under the MIT License 
```


