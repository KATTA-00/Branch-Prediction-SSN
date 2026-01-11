# Branch Prediction with SNN and ANN

This project implements and compares **Spiking Neural Networks (SNN)** and **Artificial Neural Networks (ANN)** for CPU branch prediction.

## 📁 Project Structure

```
Branch-Prediction-SSN/
│
├── 📓 snn_branch_prediction.ipynb    # Main SNN implementation notebook
├── 📓 ann_branch_prediction.ipynb    # Main ANN implementation notebook
│
├── 📂 data/                          # All datasets
│   ├── branch_data_processed.csv     # Main processed dataset
│   ├── I04.csv                       # Raw trace files
│   ├── INT03.csv
│   ├── MM03.csv
│   ├── MM05.csv
│   ├── S02.csv
│   └── S04.csv
│
├── 📂 scripts/                       # Python scripts and analysis notebooks
│   ├── dataset.py                    # Dataset processing utilities
│   ├── process_branch_data.ipynb     # Data preprocessing notebook
│   └── branch_prediction_analysis.ipynb
│
├── 📂 models/                        # Saved model files (.pth)
│   ├── snn_sequential_pc_only.pth    # Sequential SNN (PC-only)
│   ├── ann_branch_predictor.pth      # Full ANN model
│   ├── ann_pc_only.pth               # ANN PC-only model
│   └── ann_bh_only.pth               # ANN Branch History-only model
│
├── 📂 results/                       # Generated visualizations and outputs
│   ├── snn_training_history.png
│   ├── snn_confusion_matrix.png
│   ├── snn_feature_ablation_comparison.png
│   ├── snn_sequential_vs_independent_pc.png
│   ├── ann_training_history.png
│   ├── ann_confusion_matrix.png
│   ├── ann_roc_curve.png
│   └── ann_feature_ablation_comparison.png
│
└── 📂 outputs/                       # Training logs (timestamped)
    ├── snn_full_model_YYYYMMDD_HHMMSS.log
    ├── snn_pc_only_YYYYMMDD_HHMMSS.log
    ├── snn_bh_only_YYYYMMDD_HHMMSS.log
    ├── snn_sequential_pc_YYYYMMDD_HHMMSS.log
    ├── ann_full_model_YYYYMMDD_HHMMSS.log
    ├── ann_pc_only_YYYYMMDD_HHMMSS.log
    └── ann_bh_only_YYYYMMDD_HHMMSS.log
```

## 🚀 Quick Start

### 1. Run SNN Branch Prediction
```bash
# Open and run all cells in order
jupyter notebook snn_branch_prediction.ipynb
```

### 2. Run ANN Branch Prediction
```bash
# Open and run all cells in order
jupyter notebook ann_branch_prediction.ipynb
```

## 📊 Model Architectures

### Spiking Neural Network (SNN)
- **Input**: 64 neurons (32 PC bits + 32 Branch History bits)
- **Hidden**: 16 LIF (Leaky Integrate-and-Fire) neurons
- **Output**: 2 neurons (Taken / Not Taken)
- **Special Features**: 
  - Temporal dynamics with membrane potentials
  - Sequential processing with state maintenance
  - Event-driven spike-based computation

### Artificial Neural Network (ANN)
- **Input**: 64 neurons
- **Hidden**: 16 neurons (BatchNorm + ReLU + Dropout)
- **Output**: 2 neurons (Softmax)
- **Features**: Standard feedforward with regularization

## 🔬 Experiments Included

### 1. Full Model Training
- Uses both PC (Program Counter) and Branch History features
- Trains for 30 epochs with optimized hyperparameters
- Generates accuracy and loss curves

### 2. Feature Ablation Studies
- **PC-Only Model**: Tests predictive power of Program Counter alone
- **Branch History-Only Model**: Tests predictive power of history alone
- Comparison analysis to understand feature importance

### 3. Sequential SNN (SNN Only)
- Leverages SNN's inherent memory through state maintenance
- Processes sequences of branches to learn temporal patterns
- Compares sequential vs independent processing

## ⚙️ Configuration

Both models use **identical hyperparameters** for fair comparison:
- **Epochs**: 30
- **Batch Size**: 256
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Data Split**: Sequential 80/20 (maintains temporal order)
- **No Shuffling**: Preserves execution sequence

## 📈 Results

All results are saved in organized folders:

### 📂 results/
- Training history plots (accuracy & loss curves)
- Confusion matrices
- ROC curves (ANN)
- Feature ablation comparisons
- Sequential vs independent processing comparisons (SNN)

### 📂 outputs/
- **Complete training logs** for each model run
- Timestamped files (YYYYMMDD_HHMMSS format)
- Logs are written in real-time to both **console and file**
- Separate log files for each model variant:
  - Full models (SNN & ANN)
  - PC-only models
  - Branch History-only models
  - Sequential models (SNN)
- Each log contains:
  - Training progress (epoch-by-epoch)
  - Loss and accuracy values
  - Final inference results
  - Model architecture details
  - Hyperparameter configurations

## 💾 Saved Models

All trained models are saved in `models/` folder:
- Full models (SNN & ANN)
- Feature ablation models (PC-only, BH-only)
- Sequential processing models (SNN)

## 🔧 Requirements

```bash
pip install torch snntorch pandas numpy scikit-learn matplotlib seaborn tqdm
```

## 🎨 Features

### 📝 Automatic Logging
- **Dual Output**: All print statements are written to BOTH console and log files
- **Timestamped Files**: Each training run creates unique log files with timestamps
- **Organized Storage**: All logs saved in `outputs/` folder
- **Separate Logs**: Different log files for each model variant
- **Complete Record**: Captures all training details, accuracies, and configurations
- **No Data Loss**: Real-time flushing ensures logs are saved even if training is interrupted

## 📝 Key Findings

1. **Sequential Data Handling**: Branch predictions are temporal - sequential splitting (no shuffling) is crucial
2. **Feature Importance**: Feature ablation reveals which features (PC vs BH) contribute most
3. **SNN Memory**: Sequential SNN can leverage neuron state to learn temporal patterns
4. **Fair Comparison**: Identical configurations ensure valid SNN vs ANN comparison

## 🎯 Future Work

- Extend sequential processing to full models
- Test different sequence lengths
- Explore neuromorphic hardware deployment
- Advanced SNN architectures (multi-layer, different neuron types)

---

**Note**: Make sure to run cells sequentially in each notebook. The notebooks are self-contained and will generate all necessary visualizations and save models automatically.
