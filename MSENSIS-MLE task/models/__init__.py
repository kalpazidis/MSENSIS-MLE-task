# =============================================================================
# MSENSIS ML Engineering Task - Models Package
# =============================================================================
"""
This package contains all model-related utilities for the Cat vs Dog classifier.

Available Models:
    1. ViT (Vision Transformer) - Pre-trained from HuggingFace
       Source: nateraw/vit-base-cats-vs-dogs
       
    2. MobileViTv3 - Custom trained model
       Source: Trained on provided dataset
"""

from .inference import ModelInference
from .network_viz import NetworkVisualizer

__all__ = ['ModelInference', 'NetworkVisualizer']
