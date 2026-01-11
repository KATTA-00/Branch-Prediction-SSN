# Training Logs Guide

## 📝 Automatic Logging System

Both `snn_branch_prediction.ipynb` and `ann_branch_prediction.ipynb` now feature **automatic dual logging** that writes all output to both the console and timestamped log files.

## 🎯 Features

### ✅ What Gets Logged
- **Training Progress**: Epoch-by-epoch loss and accuracy
- **Model Architecture**: Layer details and parameter counts
- **Dataset Information**: Sizes, shapes, and distributions
- **Final Results**: Test accuracy, confusion matrices, classification reports
- **Hyperparameters**: Learning rates, batch sizes, epochs, etc.
- **Timestamps**: Start and end times for each training session

### 📁 Log File Naming Convention

```
outputs/
├── snn_full_model_20260111_090530.log      # Full SNN model
├── snn_pc_only_20260111_090545.log          # SNN with PC features only
├── snn_bh_only_20260111_090600.log          # SNN with Branch History only
├── snn_sequential_pc_20260111_090615.log    # Sequential SNN (PC-only)
├── ann_full_model_20260111_090630.log       # Full ANN model
├── ann_pc_only_20260111_090645.log          # ANN with PC features only
└── ann_bh_only_20260111_090700.log          # ANN with Branch History only
```

**Format**: `{model_type}_{variant}_{YYYYMMDD_HHMMSS}.log`

## 🔄 How It Works

### TeeLogger Class
```python
class TeeLogger:
    """Writes output to both console and file simultaneously"""
    - Captures all print() statements
    - Writes to console (normal behavior)
    - Writes to log file (persistent record)
    - Real-time flushing (no data loss)
```

### Automatic Log Switching
- Each major model training section creates a NEW log file
- Previous log is automatically closed
- Timestamp ensures unique filenames for each run
- No manual intervention required

## 📊 Log File Contents

### Example Structure
```
======================================================================
SNN Training Session Started: 2026-01-11 09:05:30
Log file: outputs/snn_full_model_20260111_090530.log
======================================================================

Using device: cuda:1
✓ Libraries imported successfully!

Loading dataset: data/branch_data_processed.csv
✓ Dataset loaded successfully!
Shape: (350783, 65)

======================================================================
                 TRAINING SPIKING NEURAL NETWORK
======================================================================

Epochs: 30
Batch size: 256
Timesteps: 25

----------------------------------------------------------------------
Epoch [ 1/30] | Train Loss: 0.6821 | Train Acc: 55.23% | Test Loss: 0.6754 | Test Acc: 56.12%
Epoch [ 2/30] | Train Loss: 0.6598 | Train Acc: 58.45% | Test Loss: 0.6512 | Test Acc: 59.67%
...
----------------------------------------------------------------------
✓ Training complete!
Best Test Accuracy: 78.45%

🎯 Test Accuracy: 78.45%
📉 Test Loss: 0.4321

📊 Classification Report:
...

======================================================================
Training Session Ended: 2026-01-11 09:35:45
======================================================================
```

## 🎨 Benefits

### 1. **Reproducibility**
- Complete record of every training run
- Easy to compare different runs
- Track hyperparameter changes over time

### 2. **Analysis**
- Review training progress offline
- Share results with collaborators
- Generate reports from logs

### 3. **Debugging**
- Identify where training failed
- Compare successful vs failed runs
- Track down accuracy degradation

### 4. **Documentation**
- Automatic timestamp for each run
- No need to manually copy-paste results
- Permanent record of all experiments

## 🔍 Viewing Logs

### In Real-Time
- Watch console output as normal during training
- Everything displayed is also being written to log

### After Training
```bash
# View specific log
cat outputs/snn_full_model_20260111_090530.log

# View most recent SNN log
Get-ChildItem outputs/snn_*.log | Sort-Object LastWriteTime | Select-Object -Last 1 | Get-Content

# Search for accuracy in all logs
Select-String -Path "outputs/*.log" -Pattern "Test Accuracy"

# Compare accuracies across runs
Select-String -Path "outputs/snn_full_*.log" -Pattern "Best Test Accuracy"
```

## 📈 Log Analysis Examples

### Extract Final Accuracies
```powershell
# Get all final test accuracies from SNN runs
Select-String -Path "outputs/snn_*.log" -Pattern "🎯 Test Accuracy:" | ForEach-Object {
    [PSCustomObject]@{
        File = Split-Path $_.Path -Leaf
        Accuracy = ($_ -replace '.*: ', '').Trim()
    }
}
```

### Compare Training Times
```powershell
# Extract start and end times
$logs = Get-ChildItem outputs/*.log
foreach ($log in $logs) {
    $start = Select-String -Path $log -Pattern "Training Session Started:" | Select-Object -First 1
    $end = Select-String -Path $log -Pattern "Training Session Ended:" | Select-Object -First 1
    Write-Host "$($log.Name): $start -> $end"
}
```

### Find Best Performing Model
```powershell
# Find log with highest accuracy
$best = Select-String -Path "outputs/*.log" -Pattern "Best Test Accuracy:" | 
    ForEach-Object { 
        [PSCustomObject]@{
            File = Split-Path $_.Path -Leaf
            Accuracy = [float]($_.Line -replace '.*: |%', '')
        }
    } | 
    Sort-Object Accuracy -Descending | 
    Select-Object -First 1

Write-Host "Best Model: $($best.File) with $($best.Accuracy)%"
```

## 🛠️ Maintenance

### Cleaning Old Logs
```bash
# Delete logs older than 30 days
Get-ChildItem outputs/*.log | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-30)} | Remove-Item

# Keep only 5 most recent logs per model type
Get-ChildItem outputs/snn_full_*.log | Sort-Object LastWriteTime -Descending | Select-Object -Skip 5 | Remove-Item
```

### Archiving Logs
```bash
# Compress old logs
Compress-Archive -Path outputs/*.log -DestinationPath "logs_archive_$(Get-Date -Format 'yyyyMMdd').zip"
```

## 🚀 Quick Start

Just run the notebooks as usual:
```bash
jupyter notebook snn_branch_prediction.ipynb
```

**That's it!** Logging happens automatically:
1. ✅ Console shows all output (normal behavior)
2. ✅ Files are created in `outputs/` folder
3. ✅ Each model gets its own timestamped log
4. ✅ Complete training record saved automatically

## 💡 Tips

1. **Run notebooks completely** to ensure all models are logged
2. **Check outputs/ folder** after training to verify logs were created
3. **Keep important logs** - rename or move them to prevent accidental deletion
4. **Compare timestamps** to track which runs correspond to which model versions
5. **Use logs for reports** - they contain all necessary metrics and results

---

**All output is automatically logged - no configuration needed!** 🎉
