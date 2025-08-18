# How to Run the Credit Card Fraud Detection Project

A step-by-step guide to get started with the machine learning fraud detection system.

## Quick Start Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Check Your Dataset
```bash
python download_dataset.py
```

### Step 3: Run the Analysis
```bash
python CreditCardFraud.py
```

## What You'll See

### With Current Sample Dataset:
- ✅ **Works immediately** - No errors
- 📊 Shows data structure and preprocessing steps
- ⚠️ **No ML training** - Only normal transactions in sample
- 📋 Provides instructions for getting full dataset

### With Full Dataset (Recommended):
- ✅ **Complete ML analysis** with 4 algorithms
- 📈 Performance metrics and ROC curves
- 🎯 Real fraud detection results
- 📊 Model comparison and evaluation

## Getting the Full Dataset

1. **Visit Kaggle**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. **Download**: Click the download button
3. **Replace**: Put the new `creditcard.csv` in this folder
4. **Run**: `python CreditCardFraud.py`

## Expected Results

### Sample Dataset (Current):
```
Dataset shape: (199, 30)
Fraudulent transactions: 0
Normal transactions: 199
[Data structure demonstration]
```

### Full Dataset:
```
Dataset shape: (284807, 30)
Fraudulent transactions: 492
Normal transactions: 284315
[Complete ML analysis with 4 models]
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `FileNotFoundError` | Make sure `creditcard.csv` is in the folder |
| Single-class error | Download the full dataset from Kaggle |
| Memory error | Ensure you have 4GB+ RAM for full dataset |

## Files in Project

- `CreditCardFraud.py` - Main analysis script
- `download_dataset.py` - Dataset checker and guide
- `requirements.txt` - Python packages needed
- `creditcard.csv` - **SAMPLE** dataset (replace with full)
- `README.md` - Detailed documentation

## Next Steps

1. **Try the sample**: Run with current dataset to see the structure
2. **Get full dataset**: Download from Kaggle for real analysis
3. **Experiment**: Modify the code to try different algorithms
4. **Learn**: Study the ROC curves and performance metrics

---

**Ready to start?** Run `python download_dataset.py` to check your setup! 