# MSENSIS-MLE Task: Cat vs Dog Image Classifier

A deep learning web application for classifying images of cats and dogs using Vision Transformers.

---

## 📋 Project Overview

This project implements a complete ML pipeline for binary image classification (cats vs dogs) with:

- **Pre-trained Model**: ViT Base fine-tuned on cats vs dogs dataset ([nateraw/vit-base-cats-vs-dogs](https://huggingface.co/nateraw/vit-base-cats-vs-dogs))
- **Custom Model**: MobileViTv3-S (placeholder for custom training)
- **Web Interface**: Streamlit-based UI with elegant design
- **Real-time Inference**: Instant classification with confidence scores

---

## 🛠️ Installation

### Prerequisites

- Python 3.9+ 
- pip or conda package manager
- Git

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MSENSIS-MLE-task
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/MacOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Application

### Start the Streamlit Web App

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

---

## 📁 Project Structure

```
MSENSIS-MLE-task/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── models/
│   ├── __init__.py          # Package initialization
│   ├── inference.py         # Model loading and inference
│   └── network_viz.py       # Neural network visualization
│
└── dataset/                  # (To be added) Training dataset
    ├── train/
    │   ├── cats/
    │   └── dogs/
    └── test/
        ├── cats/
        └── dogs/
```

---

## 🎯 Features

### 1. Model Selection
- Toggle between pre-trained ViT and custom MobileViTv3 models
- Real-time model switching with architecture visualization update

### 2. Neural Network Visualization
- Interactive SVG-based architecture diagram
- Shows layer structure and neuron counts
- Updates output neurons with prediction scores

### 3. Image Upload
- Drag-and-drop functionality
- File browser support
- Image preview before classification

### 4. Classification Results
- Confidence scores for both classes
- Visual progress bars
- Clear prediction badge

---

## 🔧 Technical Details

### Pre-trained Model: ViT Base

- **Source**: [nateraw/vit-base-cats-vs-dogs](https://huggingface.co/nateraw/vit-base-cats-vs-dogs)
- **Architecture**: Vision Transformer (ViT-Base)
- **Input Size**: 224×224 pixels
- **Parameters**: ~86M
- **Fine-tuned on**: Cats vs Dogs dataset

### Custom Model: MobileViTv3-S (Placeholder)

- **Source**: [MobileViTv3](https://github.com/micronDLA/MobileViTv3)
- **Variant**: Small (S)
- **Input Size**: 256×256 pixels
- **Parameters**: ~5.8M
- **Status**: To be trained on provided dataset

---

## 📝 Design Decisions

### Why Streamlit?
- Rapid prototyping for ML applications
- Built-in support for image handling
- Easy deployment options
- Clean integration with PyTorch/Transformers

### Why Vision Transformers?
- State-of-the-art performance on image classification
- Pre-trained weights available from HuggingFace
- Good balance of accuracy and inference speed

### UI Design Philosophy
- Dark theme for reduced eye strain
- Clear visual hierarchy
- Responsive layout
- Intuitive drag-and-drop interaction

---

## 🔮 Future Enhancements

1. **FastAPI Backend**: REST API for programmatic access
2. **Custom Model Training**: Train MobileViTv3 on provided dataset
3. **Batch Processing**: Classify multiple images at once
4. **Model Comparison**: Side-by-side inference with both models
5. **Explainability**: Attention visualization and GradCAM

---

## 📜 License

This project was created as part of an ML Engineering interview task for MSENSIS.

---

## 👤 Author

Interview Candidate  
January 2026

