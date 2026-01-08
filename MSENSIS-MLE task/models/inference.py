# =============================================================================
# MSENSIS ML Engineering Task - Model Inference Module
# =============================================================================
"""
This module handles model loading and inference for image classification.

Supported Models:
    - ViT Base (nateraw/vit-base-cats-vs-dogs): Pre-trained Vision Transformer
    - MobileViTv3-S: Custom trained model (placeholder for future implementation)

Author: Interview Candidate
Date: January 2026
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Optional
from transformers import ViTImageProcessor, ViTForImageClassification
import streamlit as st


class ModelInference:
    """
    Handles loading and inference for cat/dog classification models.
    
    This class provides a unified interface for different model architectures,
    allowing seamless switching between pre-trained and custom models.
    
    Attributes:
        model_name (str): Currently loaded model identifier
        model: The loaded PyTorch model
        processor: Image preprocessor for the model
        device (str): Computing device ('cuda' or 'cpu')
        class_labels (list): Classification labels ['cat', 'dog']
    """
    
    # =========================================================================
    # Class Constants
    # =========================================================================
    
    # Available model configurations
    AVAILABLE_MODELS = {
        'vit': {
            'name': 'ViT Base (Pre-trained)',
            'hub_id': 'nateraw/vit-base-cats-vs-dogs',
            'description': 'Vision Transformer fine-tuned on cats vs dogs dataset',
            'architecture': 'Vision Transformer (ViT-Base)',
            'input_size': 224
        },
        'mobilevit': {
            'name': 'MobileViTv3-S (Custom)',
            'hub_id': None,  # To be loaded from local checkpoint
            'description': 'Lightweight MobileViT variant trained on custom dataset',
            'architecture': 'MobileViTv3 Small',
            'input_size': 256
        }
    }
    
    # Classification labels
    CLASS_LABELS = ['Cat', 'Dog']
    
    def __init__(self, model_type: str = 'vit'):
        """
        Initialize the inference engine with specified model.
        
        Args:
            model_type (str): Model identifier ('vit' or 'mobilevit')
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.processor = None
        self.model_type = model_type
        self._load_model(model_type)
    
    # =========================================================================
    # Model Loading Methods
    # =========================================================================
    
    def _load_model(self, model_type: str) -> None:
        """
        Load the specified model into memory.
        
        Args:
            model_type (str): Model identifier ('vit' or 'mobilevit')
            
        Raises:
            ValueError: If model_type is not recognized
        """
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available: {list(self.AVAILABLE_MODELS.keys())}")
        
        if model_type == 'vit':
            self._load_vit_model()
        elif model_type == 'mobilevit':
            self._load_mobilevit_model()
    
    @st.cache_resource
    def _load_vit_model(_self) -> None:
        """
        Load the pre-trained ViT model from HuggingFace.
        
        The model is cached using Streamlit's cache_resource decorator
        to avoid reloading on every interaction.
        """
        hub_id = _self.AVAILABLE_MODELS['vit']['hub_id']
        
        # Load processor (handles image preprocessing)
        _self.processor = ViTImageProcessor.from_pretrained(hub_id)
        
        # Load the model
        _self.model = ViTForImageClassification.from_pretrained(hub_id)
        _self.model.to(_self.device)
        _self.model.eval()  # Set to evaluation mode
        
    def _load_mobilevit_model(self) -> None:
        """
        Load the custom MobileViTv3 model.
        
        Note: This is a placeholder for when the custom trained model
        is added. Currently returns a message indicating the model
        is not yet available.
        """
        # Placeholder - will be implemented when custom model is provided
        self.model = None
        self.processor = None
        
    # =========================================================================
    # Inference Methods
    # =========================================================================
    
    def predict(self, image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        """
        Perform classification on the input image.
        
        Args:
            image (PIL.Image): Input image to classify
            
        Returns:
            Tuple containing:
                - predicted_class (str): 'Cat' or 'Dog'
                - confidence (float): Confidence score (0-1)
                - all_scores (dict): Scores for all classes
        """
        if self.model is None:
            return self._placeholder_prediction()
        
        if self.model_type == 'vit':
            return self._predict_vit(image)
        else:
            return self._predict_mobilevit(image)
    
    def _predict_vit(self, image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        """
        Run inference using the ViT model.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            Tuple of (predicted_class, confidence, all_scores)
        """
        # Preprocess the image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference (no gradient computation needed)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Convert logits to probabilities using softmax
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        probs = probabilities.cpu().numpy()[0]
        
        # Get prediction
        predicted_idx = np.argmax(probs)
        predicted_class = self.CLASS_LABELS[predicted_idx]
        confidence = float(probs[predicted_idx])
        
        # Build scores dictionary
        all_scores = {
            label: float(prob) 
            for label, prob in zip(self.CLASS_LABELS, probs)
        }
        
        return predicted_class, confidence, all_scores
    
    def _predict_mobilevit(self, image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        """
        Run inference using the MobileViTv3 model.
        
        Note: Placeholder implementation until custom model is added.
        """
        return self._placeholder_prediction()
    
    def _placeholder_prediction(self) -> Tuple[str, float, Dict[str, float]]:
        """
        Return placeholder prediction for unavailable models.
        
        Returns:
            Tuple with placeholder values indicating model not loaded
        """
        return (
            "Model Not Loaded",
            0.0,
            {"Cat": 0.0, "Dog": 0.0}
        )
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_model_info(self) -> Dict:
        """
        Get information about the currently loaded model.
        
        Returns:
            Dictionary containing model metadata
        """
        return self.AVAILABLE_MODELS.get(self.model_type, {})
    
    def is_model_loaded(self) -> bool:
        """
        Check if a model is currently loaded and ready for inference.
        
        Returns:
            bool: True if model is ready, False otherwise
        """
        return self.model is not None
    
    @classmethod
    def get_available_models(cls) -> Dict:
        """
        Get all available model configurations.
        
        Returns:
            Dictionary of available models and their configurations
        """
        return cls.AVAILABLE_MODELS

