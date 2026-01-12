# MSENSIS-MLE Task: Cat vs Dog Image Classifier

A deep learning Web application for classifying images of cats and dogs using Vision Transformers.

---

## 📋 Project Overview

This project implements a complete ML pipeline for binary image classification (cats vs dogs) with:

- **Pre-trained Model**: ViT Base fine-tuned on cats vs dogs dataset ([nateraw/vit-base-cats-vs-dogs](https://huggingface.co/nateraw/vit-base-cats-vs-dogs))
- **Custom Model**: MobileViTv3-S (https://github.com/micronDLA/MobileViTv3)
- **Web Interface**: Streamlit-based UI
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
   git clone https://github.com/kalpazidis/MSENSIS-MLE-task.git
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

## 🛠️Activating the enviroment

    ```bash
    # Windows
    .\venv\Scripts\activate
    ```


## 🚀 Running the Application

### Start the Streamlit Web App

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

---

## 🎯 Features

### 1. Model Selection
- Toggle between pre-trained ViT and custom MobileViTv3 models
- Real-time model switching with architecture visualization update

### 2. Neural Network Visualization
- Interactive SVG-based architecture diagram
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

### Custom Model: MobileViTv3-S

- **Source**: [MobileViTv3](https://github.com/micronDLA/MobileViTv3)
- **Variant**: Small (S)
- **Input Size**: 256×256 pixels
- **Parameters**: ~5.8M
- **Status**: Trained on provided dataset

---

## 📝 Design Decisions

### Why Streamlit?
- Rapid prototyping for ML applications
- Built-in support for image handling
- Easy deployment options
- Clean integration with PyTorch/Transformers


### UI Design Philosophy
- Dark theme for reduced eye strain
- Clear visual hierarchy
- Responsive layout
- Intuitive drag-and-drop interaction

---

## 🔮 Future Enhancements


1. **Custom Model Training**: Expand training set and train MobileViTv3 for more epochs
2. **Batch Processing**: Classify multiple images at once

---

## 📜 License

This project was created as part of an ML Engineering interview task for MSENSIS.

---

## 👤 Author

Kalpazidis Alexandros  
January 2026
