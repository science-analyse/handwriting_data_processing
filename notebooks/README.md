# Notebooks 📓

Jupyter notebooks for training, experimentation, and analysis.

## Available Notebooks

### 🏆 SOCAR_Hackathon_Training.ipynb

**Complete training pipeline for Google Colab**

This notebook implements the full hybrid OCR architecture on real handwriting data.

#### What it does:

1. ✅ Downloads Kaggle handwriting dataset
2. ✅ Implements preprocessing pipeline
3. ✅ Fine-tunes TrOCR model
4. ✅ Creates ensemble system
5. ✅ Evaluates performance (CER/WER)
6. ✅ Saves trained model

#### How to use:

**Option 1: Google Colab (Recommended)**

1. Upload to Colab: https://colab.research.google.com/
2. Click "File" → "Upload notebook"
3. Select `SOCAR_Hackathon_Training.ipynb`
4. Enable GPU: Runtime → Change runtime type → GPU
5. Run all cells!

**Option 2: Local Jupyter**

```bash
# Install Jupyter
pip install jupyter

# Launch
jupyter notebook

# Open SOCAR_Hackathon_Training.ipynb
```

#### Features:

- 📥 **Auto dataset download** using kagglehub
- 🔧 **Complete preprocessing** (deskew, denoise, enhance)
- 🤖 **TrOCR fine-tuning** with Hugging Face Transformers
- 📊 **Visualization** of training progress
- 🎯 **Ensemble** implementation
- 📈 **Evaluation** metrics (CER, WER)
- 💾 **Model export** for deployment
- 🎮 **Interactive demo** to test on your images

#### Requirements:

- GPU recommended (Colab free tier works!)
- ~12GB RAM
- ~5GB disk space

#### Expected Results:

After training (3 epochs, ~1-2 hours on Colab GPU):
- **CER**: 3-5%
- **WER**: 8-12%
- Ready-to-deploy model

#### Dataset:

Uses: [Handwritten2Text Training Dataset](https://www.kaggle.com/datasets/chaimaourgani/handwritten2text-training-dataset)

Contains handwritten text images with ground truth labels.

---

## Coming Soon

- [ ] `Donut_Training.ipynb` - OCR-free model training
- [ ] `LayoutLMv3_Training.ipynb` - Multimodal model training
- [ ] `Ensemble_Comparison.ipynb` - Compare all models
- [ ] `Data_Augmentation.ipynb` - Advanced augmentation techniques
- [ ] `Model_Optimization.ipynb` - Quantization and optimization

---

## Tips

### For Best Results:

1. **Use GPU** - Training is much faster
2. **Start small** - Test with subset first
3. **Monitor metrics** - Watch CER/WER during training
4. **Save checkpoints** - Don't lose progress!
5. **Experiment** - Try different hyperparameters

### Common Issues:

**Out of Memory**
```python
# Reduce batch size
training_args.per_device_train_batch_size = 4
```

**Slow training**
```python
# Reduce dataset size for testing
train_pairs = train_pairs[:1000]
```

**Poor accuracy**
```python
# Train longer
training_args.num_train_epochs = 5
```

---

## Notebook Structure

Each notebook follows this template:

```
1. Setup & Installation
2. Dataset Loading
3. Data Exploration
4. Preprocessing
5. Model Training
6. Evaluation
7. Visualization
8. Export
```

---

## Resources

- 📚 [Model Design Guide](../docs/model_design.md)
- 🏗️ [Architecture](../docs/ARCHITECTURE.md)
- ⚡ [Quick Start](../docs/QUICKSTART.md)

---

**Made for SOCAR Hackathon 2025** 🏆
