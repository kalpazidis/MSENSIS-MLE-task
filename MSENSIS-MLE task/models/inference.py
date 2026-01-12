# =============================================================================
# MSENSIS ML Engineering Task - Model Inference Module
# =============================================================================
"""
This module handles model loading and inference for image classification.

Supported Models:
    - ViT Base (nateraw/vit-base-cats-vs-dogs): Pre-trained Vision Transformer
    - MobileViTv3-S: Custom trained model on Cats vs Dogs dataset

Author: Kalpazidis Alexandros
Date: January 2026
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Optional
from pathlib import Path
import torchvision.transforms as transforms

# HuggingFace imports for ViT
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
        class_labels (list): Classification labels ['Cat', 'Dog']
    """
    
    # =========================================================================
    # Class Constants
    # =========================================================================
    
    # Project root directory
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # Path to trained MobileViTv3 checkpoint
    MOBILEVIT_CHECKPOINT = PROJECT_ROOT / "results_mobilevitv3_catdog" / "mobilevitv3_catdog" / "checkpoint_best.pt"
    MOBILEVIT_EMA_CHECKPOINT = PROJECT_ROOT / "results_mobilevitv3_catdog" / "mobilevitv3_catdog" / "checkpoint_ema_best.pt"
    
    # CVNets path for model architecture
    CVNETS_PATH = PROJECT_ROOT / "MobileViT3S"
    
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
            'hub_id': None,
            'description': 'Lightweight MobileViT variant trained on custom dataset',
            'architecture': 'MobileViTv3 Small',
            'input_size': 256
        }
    }
    
    # Classification labels (alphabetical order as per ImageNet folder structure)
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
        self.transform = None
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
        Load the custom trained MobileViTv3-S model.
        
        Attempts to load from the trained checkpoint. If not available,
        returns None to indicate model not ready.
        """
        # Check if checkpoint exists
        checkpoint_path = None
        if self.MOBILEVIT_EMA_CHECKPOINT.exists():
            checkpoint_path = self.MOBILEVIT_EMA_CHECKPOINT
        elif self.MOBILEVIT_CHECKPOINT.exists():
            checkpoint_path = self.MOBILEVIT_CHECKPOINT
        
        if checkpoint_path is None:
            # Model not yet trained
            self.model = None
            self.processor = None
            print("[INFO] MobileViTv3 checkpoint not found. Model not available.")
            return
        
        try:
            # Add CVNets to path for model imports
            if str(self.CVNETS_PATH) not in sys.path:
                sys.path.insert(0, str(self.CVNETS_PATH))
            
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Get the state dict from checkpoint
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (from DDP training)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Build model using cvnets architecture
            from cvnets.models.classification import build_classification_model
            from argparse import Namespace
            
            # Create opts matching the training config
            opts = self._create_mobilevit_opts()
            
            # Build the model
            self.model = build_classification_model(opts)
            
            # Load weights
            self.model.load_state_dict(state_dict, strict=True)
            self.model.to(self.device)
            self.model.eval()
            
            # Setup image transforms for MobileViT (matching training transforms)
            # NOTE: Training used NumpyToTensor which only divides by 255, NO normalization!
            self.transform = transforms.Compose([
                transforms.Resize(288),  # Resize shorter side to 288 (256 + 32)
                transforms.CenterCrop(256),  # Center crop to 256x256
                transforms.ToTensor(),  # This converts to [0,1] range (divides by 255)
                # NO normalization - training didn't use it
            ])
            
            print(f"[INFO] MobileViTv3 model loaded successfully from {checkpoint_path}")
            
        except Exception as e:
            import traceback
            print(f"[WARNING] Failed to load MobileViTv3 model: {e}")
            traceback.print_exc()
            self.model = None
            self.processor = None
    
    def _create_mobilevit_opts(self):
        """
        Create a minimal options object for MobileViTv3 model construction.
        Uses argparse.Namespace with getattr for compatibility with cvnets.
        
        Returns:
            Namespace object with model configuration
        """
        from argparse import Namespace
        
        # Create a dictionary with all required options
        opts_dict = {
            # Model classification settings
            'model.classification.name': 'mobilevit_v3',
            'model.classification.n_classes': 2,
            'model.classification.classifier_dropout': 0.2,
            'model.classification.pretrained': None,
            'model.classification.freeze_batch_norm': False,
            
            # MobileViT specific settings
            'model.classification.mit.mode': 'small_v3',
            'model.classification.mit.ffn_dropout': 0.0,
            'model.classification.mit.attn_dropout': 0.0,
            'model.classification.mit.dropout': 0.1,
            'model.classification.mit.number_heads': 4,
            'model.classification.mit.no_fuse_local_global_features': False,
            'model.classification.mit.conv_kernel_size': 3,
            'model.classification.mit.head_dim': None,
            'model.classification.mit.transformer_norm_layer': 'layer_norm',
            
            # Activation settings
            'model.activation.name': 'swish',
            'model.activation.inplace': False,
            'model.activation.neg_slope': 0.1,
            'model.classification.activation.name': 'swish',
            'model.classification.activation.inplace': False,
            'model.classification.activation.neg_slope': 0.1,
            
            # Normalization settings
            'model.normalization.name': 'batch_norm_2d',
            'model.normalization.momentum': 0.1,
            'model.normalization.groups': 32,
            
            # Layer settings
            'model.layer.global_pool': 'mean',
            'model.layer.conv_init': 'kaiming_normal',
            'model.layer.conv_init_std_dev': None,
            'model.layer.linear_init': 'trunc_normal',
            'model.layer.linear_init_std_dev': 0.02,
            'model.layer.group_linear_init': 'xavier_uniform',
            'model.layer.group_linear_init_std_dev': 0.01,
        }
        
        # Create a custom Namespace that handles dotted attribute access
        class DottedNamespace(Namespace):
            def __init__(self, **kwargs):
                super().__init__()
                self._opts = kwargs
            
            def __getattr__(self, name):
                # First check in _opts with dotted notation
                if '_opts' in self.__dict__ and name in self._opts:
                    return self._opts[name]
                # Check with dotted version
                for key, value in self._opts.items():
                    if key == name or key.replace('.', '_') == name:
                        return value
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # For cvnets, we need to use getattr with dotted strings
        # Create a Namespace and add a custom getattr handler
        opts = Namespace()
        for key, value in opts_dict.items():
            setattr(opts, key, value)
        
        return opts
    
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
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            Tuple of (predicted_class, confidence, all_scores)
        """
        if self.model is None or self.transform is None:
            return self._placeholder_prediction()
        
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess the image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('out', None))
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
        
        # Convert logits to probabilities
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
    
    @classmethod
    def is_mobilevit_trained(cls) -> bool:
        """
        Check if the MobileViTv3 model has been trained.
        
        Returns:
            bool: True if checkpoint exists, False otherwise
        """
        return cls.MOBILEVIT_CHECKPOINT.exists() or cls.MOBILEVIT_EMA_CHECKPOINT.exists()
