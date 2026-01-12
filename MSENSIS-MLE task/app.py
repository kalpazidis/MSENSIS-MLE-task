# =============================================================================
# MSENSIS ML Engineering Task - Main Application
# =============================================================================
"""
Cat vs Dog Image Classifier - Streamlit Web Application

This application provides a web interface for classifying images of cats and dogs
using pre-trained deep learning models. It features an elegant UI with neural
network visualization and real-time inference.

Author: Kalpazidis Alexandros
Date: January 2026
"""

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import base64
from io import BytesIO
from models.inference import ModelInference

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Cat vs Dog Classifier | MSENSIS",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# Custom CSS Styling
# =============================================================================

st.markdown("""
<style>
    /* ===================================================================== */
    /* Global Theme - Pure Black Background                                  */
    /* ===================================================================== */
    
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp, .main, [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"], section[data-testid="stSidebar"],
    .block-container, [data-testid="stVerticalBlock"] {
        background-color: #000000 !important;
    }
    
    .stApp > header {
        background-color: transparent !important;
    }
    
    #MainMenu, footer, header, [data-testid="stDecoration"] {
        visibility: hidden !important;
        display: none !important;
    }
    
    .block-container {
        padding: 1rem 2rem 2rem 2rem !important;
        max-width: 1600px !important;
    }
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* ===================================================================== */
    /* Radio Button Styling - Horizontal and Centered                        */
    /* ===================================================================== */
    
    .stRadio {
        display: flex !important;
        justify-content: center !important;
    }
    
    .stRadio > div {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 1.5rem !important;
        width: auto !important;
    }
    
    .stRadio > div > label {
        background: #111 !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        color: #fff !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
        min-width: 180px !important;
        text-align: center !important;
        display: inline-flex !important;
        justify-content: center !important;
        align-items: center !important;
        flex-shrink: 0 !important;
    }
    
    .stRadio > div > label:hover {
        border-color: #6366f1 !important;
        background: #1a1a1a !important;
    }
    
    /* Ensure the column containing radio is wide enough for horizontal layout */
    [data-testid="column"]:has(.stRadio) {
        min-width: 450px !important;
    }
    
    /* ===================================================================== */
    /* File Uploader Styling - Centered content                              */
    /* ===================================================================== */
    
    [data-testid="stFileUploader"] {
        background: transparent !important;
    }
    
    [data-testid="stFileUploader"] section {
        padding: 0 !important;
    }
    
    [data-testid="stFileUploader"] section > div {
        background: transparent !important;
        padding: 0 !important;
    }
    
    [data-testid="stFileUploaderDropzone"] {
        background: #0a0a0a !important;
        border: 2px dashed #333 !important;
        border-radius: 12px !important;
        padding: 2rem 1rem !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
    }
    
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #6366f1 !important;
    }
    
    [data-testid="stFileUploaderDropzone"] > div {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        width: 100% !important;
    }
    
    [data-testid="stFileUploaderDropzone"] span {
        text-align: center !important;
    }
    
    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        margin: 0 auto !important;
    }
    
    [data-testid="stFileUploaderDropzone"] small {
        color: #555 !important;
        font-size: 11px !important;
        text-align: center !important;
        display: block !important;
        width: 100% !important;
    }
    
    /* Hide the uploaded file info completely */
    [data-testid="stFileUploader"] > div > div:nth-child(2) {
        display: none !important;
    }
    
    /* ===================================================================== */
    /* Image Styling                                                         */
    /* ===================================================================== */
    
    [data-testid="stImage"] {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 2px solid #22c55e !important;
        margin-top: 0.75rem !important;
    }
    
    [data-testid="stImage"] img {
        border-radius: 10px !important;
    }
    
    /* ===================================================================== */
    /* Section Title Styling - Centered                                      */
    /* ===================================================================== */
    
    .section-title {
        font-size: 11px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.75rem;
        text-align: center;
        width: 100%;
        display: block;
    }
    
    /* Ensure the radio container itself is centered in the page flow */
    div:has(> .stRadio) {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }
    
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Helper: Convert PIL Image to Base64 for SVG embedding
# =============================================================================

def image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL image to base64 data URI for embedding in SVG.
    
    Args:
        image: PIL Image object
        
    Returns:
        Base64 encoded data URI string
    """
    # Create a copy and resize for the SVG preview
    img = image.copy()
    img.thumbnail((150, 150), Image.Resampling.LANCZOS)
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"


# =============================================================================
# Network Visualization - ViT Architecture (Based on Original Paper)
# =============================================================================

def generate_vit_architecture_html(predictions: dict = None, image_b64: str = None) -> str:
    """
    Generate ViT architecture visualization based on the original paper diagram.
    
    Args:
        predictions: Classification scores {'Cat': x, 'Dog': y}
        image_b64: Base64 encoded image for input preview (optional)
    
    The architecture shows:
    - Input image split into patches (or actual uploaded image)
    - Linear projection of flattened patches
    - Patch + Position embeddings
    - Transformer Encoder
    - MLP Head
    - Class output (Cat/Dog)
    """
    
    width = 700
    height = 420
    
    # Colors
    c_patch = "#8B5CF6"       # Purple for patches
    c_embed = "#EC4899"       # Pink for linear projection
    c_position = "#F59E0B"    # Orange/yellow for embeddings
    c_transformer = "#6B7280" # Gray for transformer
    c_mlp = "#F97316"         # Orange for MLP
    c_cat = "#F97316"         # Orange for cat
    c_dog = "#3B82F6"         # Blue for dog
    
    # Get prediction values
    cat_score = predictions.get('Cat', 0) if predictions else 0
    dog_score = predictions.get('Dog', 0) if predictions else 0
    
    # Determine input image section based on whether image is provided
    if image_b64:
        # Show actual uploaded image with green border
        input_image_section = f'''
            <!-- Actual uploaded image preview -->
            <defs>
                <clipPath id="imgClip">
                    <rect x="32" y="142" width="76" height="76" rx="4"/>
                </clipPath>
            </defs>
            <rect x="30" y="140" width="80" height="80" fill="#1a1a1a" stroke="#22c55e" stroke-width="2" rx="6"/>
            <image x="32" y="142" width="76" height="76" 
                   href="{image_b64}" 
                   preserveAspectRatio="xMidYMid slice"
                   clip-path="url(#imgClip)"/>
        '''
    else:
        # Show default patch grid placeholder
        input_image_section = f'''
            <!-- Default patch grid placeholder -->
            <rect x="30" y="140" width="80" height="80" fill="#1a1a1a" stroke="#333" rx="6"/>
            <g opacity="0.7">
                <rect x="32" y="142" width="25" height="25" fill="{c_patch}" opacity="0.7" rx="2"/>
                <rect x="58" y="142" width="25" height="25" fill="{c_patch}" opacity="0.5" rx="2"/>
                <rect x="84" y="142" width="24" height="25" fill="{c_patch}" opacity="0.6" rx="2"/>
                <rect x="32" y="168" width="25" height="25" fill="{c_patch}" opacity="0.5" rx="2"/>
                <rect x="58" y="168" width="25" height="25" fill="{c_patch}" opacity="0.8" rx="2"/>
                <rect x="84" y="168" width="24" height="25" fill="{c_patch}" opacity="0.4" rx="2"/>
                <rect x="32" y="194" width="25" height="24" fill="{c_patch}" opacity="0.6" rx="2"/>
                <rect x="58" y="194" width="25" height="24" fill="{c_patch}" opacity="0.4" rx="2"/>
                <rect x="84" y="194" width="24" height="24" fill="{c_patch}" opacity="0.7" rx="2"/>
            </g>
        '''
    
    svg = f'''
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
        <defs>
            <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.3"/>
            </filter>
            <linearGradient id="transformerGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style="stop-color:#4B5563"/>
                <stop offset="100%" style="stop-color:#374151"/>
            </linearGradient>
            <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#444"/>
            </marker>
        </defs>
        
        <!-- Background -->
        <rect width="{width}" height="{height}" fill="#0a0a0a" rx="16"/>
        <rect x="1" y="1" width="{width-2}" height="{height-2}" fill="none" stroke="#222" rx="15"/>
        
        <!-- Title -->
        <text x="{width/2}" y="28" fill="#fff" font-size="15" font-family="Inter, sans-serif" 
              text-anchor="middle" font-weight="600">Vision Transformer (ViT) Architecture</text>
        <text x="{width/2}" y="46" fill="#666" font-size="11" font-family="Inter, sans-serif" 
              text-anchor="middle">Pre-trained on ImageNet-21k, fine-tuned for Cat vs Dog</text>
        
        <!-- ============================================================ -->
        <!-- INPUT IMAGE                                                   -->
        <!-- ============================================================ -->
        
        {input_image_section}
        
        <text x="70" y="240" fill="#888" font-size="10" font-family="Inter" text-anchor="middle">Input Image</text>
        <text x="70" y="253" fill="#666" font-size="9" font-family="Inter" text-anchor="middle">224 × 224</text>
        
        <!-- Arrow from image to linear projection -->
        <line x1="115" y1="180" x2="145" y2="180" stroke="#444" stroke-width="1.5" marker-end="url(#arrow)"/>
        
        <!-- ============================================================ -->
        <!-- LINEAR PROJECTION (Patch Embedding)                           -->
        <!-- ============================================================ -->
        
        <rect x="150" y="130" width="30" height="100" fill="{c_embed}" rx="4" filter="url(#shadow)"/>
        <text x="165" y="185" fill="#fff" font-size="9" font-family="Inter" text-anchor="middle" 
              transform="rotate(-90, 165, 185)">Linear Projection</text>
        
        <!-- Arrow -->
        <line x1="185" y1="180" x2="205" y2="180" stroke="#444" stroke-width="1.5" marker-end="url(#arrow)"/>
        
        <!-- ============================================================ -->
        <!-- PATCH + POSITION EMBEDDINGS                                   -->
        <!-- ============================================================ -->
        
        <g transform="translate(210, 140)">
            <!-- Class token (special) -->
            <rect x="0" y="20" width="22" height="22" fill="{c_position}" rx="3" filter="url(#shadow)"/>
            <text x="11" y="35" fill="#000" font-size="8" font-family="Inter" text-anchor="middle" font-weight="600">CLS</text>
            
            <!-- Patch embeddings -->
            <rect x="26" y="20" width="22" height="22" fill="{c_position}" opacity="0.8" rx="3"/>
            <text x="37" y="35" fill="#000" font-size="9" font-family="Inter" text-anchor="middle">1</text>
            
            <rect x="52" y="20" width="22" height="22" fill="{c_position}" opacity="0.8" rx="3"/>
            <text x="63" y="35" fill="#000" font-size="9" font-family="Inter" text-anchor="middle">2</text>
            
            <rect x="78" y="20" width="22" height="22" fill="{c_position}" opacity="0.8" rx="3"/>
            <text x="89" y="35" fill="#000" font-size="9" font-family="Inter" text-anchor="middle">3</text>
            
            <text x="104" y="35" fill="#666" font-size="12" font-family="Inter">...</text>
            
            <rect x="115" y="20" width="22" height="22" fill="{c_position}" opacity="0.8" rx="3"/>
            <text x="126" y="35" fill="#000" font-size="9" font-family="Inter" text-anchor="middle">N</text>
            
            <!-- Label -->
            <text x="70" y="65" fill="#888" font-size="9" font-family="Inter" text-anchor="middle">Patch + Position</text>
            <text x="70" y="77" fill="#666" font-size="8" font-family="Inter" text-anchor="middle">Embeddings</text>
        </g>
        
        <!-- Arrow down to transformer -->
        <line x1="280" y1="225" x2="280" y2="260" stroke="#444" stroke-width="1.5" marker-end="url(#arrow)"/>
        
        <!-- ============================================================ -->
        <!-- TRANSFORMER ENCODER                                           -->
        <!-- ============================================================ -->
        
        <rect x="170" y="265" width="220" height="70" fill="url(#transformerGrad)" rx="8" filter="url(#shadow)"/>
        <rect x="172" y="267" width="216" height="66" fill="none" stroke="#555" rx="6" stroke-dasharray="4,2"/>
        
        <text x="280" y="295" fill="#fff" font-size="13" font-family="Inter" text-anchor="middle" font-weight="600">Transformer Encoder</text>
        <text x="280" y="312" fill="#9CA3AF" font-size="10" font-family="Inter" text-anchor="middle">12 Layers × 12 Heads × 768 Dim</text>
        <text x="280" y="326" fill="#6B7280" font-size="9" font-family="Inter" text-anchor="middle">Self-Attention + MLP</text>
        
        <!-- Arrow from transformer to MLP -->
        <line x1="395" y1="300" x2="430" y2="300" stroke="#444" stroke-width="1.5" marker-end="url(#arrow)"/>
        
        <!-- ============================================================ -->
        <!-- MLP HEAD                                                      -->
        <!-- ============================================================ -->
        
        <rect x="435" y="275" width="70" height="50" fill="{c_mlp}" rx="6" filter="url(#shadow)"/>
        <text x="470" y="297" fill="#fff" font-size="11" font-family="Inter" text-anchor="middle" font-weight="600">MLP</text>
        <text x="470" y="312" fill="#fff" font-size="9" font-family="Inter" text-anchor="middle" opacity="0.8">Head</text>
        
        <!-- Arrow to output -->
        <line x1="510" y1="300" x2="545" y2="300" stroke="#444" stroke-width="1.5" marker-end="url(#arrow)"/>
        
        <!-- ============================================================ -->
        <!-- OUTPUT CLASSES (Cat / Dog)                                    -->
        <!-- ============================================================ -->
        
        <g transform="translate(550, 260)">
            <!-- Class box -->
            <rect x="0" y="0" width="120" height="80" fill="#1a1a1a" stroke="#333" rx="8"/>
            <text x="60" y="18" fill="#888" font-size="9" font-family="Inter" text-anchor="middle" 
                  letter-spacing="1">Classification</text>
            
            <!-- Cat output -->
            <rect x="10" y="28" width="100" height="22" fill="#111" rx="4"/>
            <text x="20" y="43" fill="{c_cat}" font-size="11" font-family="Inter" font-weight="500">Cat</text>
            <text x="100" y="43" fill="{c_cat}" font-size="11" font-family="monospace" 
                  text-anchor="end" font-weight="600">{cat_score:.1%}</text>
            
            <!-- Dog output -->
            <rect x="10" y="54" width="100" height="22" fill="#111" rx="4"/>
            <text x="20" y="69" fill="{c_dog}" font-size="11" font-family="Inter" font-weight="500">Dog</text>
            <text x="100" y="69" fill="{c_dog}" font-size="11" font-family="monospace" 
                  text-anchor="end" font-weight="600">{dog_score:.1%}</text>
        </g>
        
        <!-- ============================================================ -->
        <!-- Data Flow Label                                               -->
        <!-- ============================================================ -->
        
        <text x="{width/2}" y="{height - 20}" fill="#444" font-size="10" font-family="Inter" 
              text-anchor="middle">← Data Flow →</text>
        
        <!-- ============================================================ -->
        <!-- Architecture Info                                             -->
        <!-- ============================================================ -->
        
        <text x="70" y="380" fill="#555" font-size="9" font-family="Inter" text-anchor="middle">Patch Size: 16×16</text>
        <text x="280" y="380" fill="#555" font-size="9" font-family="Inter" text-anchor="middle">Hidden Dim: 768</text>
        <text x="470" y="380" fill="#555" font-size="9" font-family="Inter" text-anchor="middle">Parameters: 86M</text>
        
    </svg>
    '''
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; padding: 0; background: #000; display: flex; justify-content: center; }}
        </style>
    </head>
    <body>{svg}</body>
    </html>
    '''


def generate_mobilevit_architecture_html(predictions: dict = None, image_b64: str = None) -> str:
    """
    Generate MobileViTv3-S architecture visualization based on the paper diagram.
    
    Architecture flow:
    Input → Conv3×3↓2 → MV2 → MV2↓2 → 2×MV2 → MV2↓2+MobileViT(L=2) → 
    MV2↓2+MobileViT(L=4) → MV2↓2+MobileViT(L=3) → Conv1×1 → GlobalPool → Linear → Classes
    
    Args:
        predictions: Classification scores {'Cat': x, 'Dog': y}
        image_b64: Base64 encoded image for input preview
    """
    
    width = 850
    height = 420
    
    cat_score = predictions.get('Cat', 0) if predictions else 0
    dog_score = predictions.get('Dog', 0) if predictions else 0
    
    # Colors matching the diagram
    c_conv = "#F97316"       # Orange/Coral for Conv layers
    c_mv2 = "#84CC16"        # Light green for MV2 (InvertedResidual)
    c_mvit = "#EC4899"       # Pink/Magenta for MobileViT blocks
    c_pool = "#FDE047"       # Yellow for Global pool + Linear
    c_cat = "#F97316"
    c_dog = "#3B82F6"
    
    # Determine input image section
    if image_b64:
        input_image_section = f'''
            <defs>
                <clipPath id="imgClipMV">
                    <rect x="17" y="167" width="56" height="56" rx="4"/>
                </clipPath>
            </defs>
            <rect x="15" y="165" width="60" height="60" fill="#1a1a1a" stroke="#22c55e" stroke-width="2" rx="6"/>
            <image x="17" y="167" width="56" height="56" 
                   href="{image_b64}" 
                   preserveAspectRatio="xMidYMid slice"
                   clip-path="url(#imgClipMV)"/>
        '''
    else:
        input_image_section = '''
            <rect x="15" y="165" width="60" height="60" fill="#1a1a1a" stroke="#333" rx="6"/>
            <text x="45" y="200" fill="#444" font-size="18" font-family="Inter" text-anchor="middle">?</text>
        '''
    
    svg = f'''
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
        <defs>
            <filter id="shadowMV" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="2" stdDeviation="2" flood-opacity="0.3"/>
            </filter>
            <marker id="arrowMV" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                <polygon points="0 0, 8 3, 0 6" fill="#444"/>
            </marker>
        </defs>
        
        <!-- Background -->
        <rect width="{width}" height="{height}" fill="#0a0a0a" rx="16"/>
        <rect x="1" y="1" width="{width-2}" height="{height-2}" fill="none" stroke="#222" rx="15"/>
        
        <!-- Title -->
        <text x="{width/2}" y="28" fill="#fff" font-size="15" font-family="Inter, sans-serif" 
              text-anchor="middle" font-weight="600">MobileViTv3-S Architecture</text>
        <text x="{width/2}" y="46" fill="#666" font-size="11" font-family="Inter, sans-serif" 
              text-anchor="middle">Lightweight Hybrid CNN-Transformer for Mobile Devices</text>
        
        <!-- ============================================================ -->
        <!-- INPUT IMAGE                                                   -->
        <!-- ============================================================ -->
        
        {input_image_section}
        <text x="45" y="245" fill="#888" font-size="9" font-family="Inter" text-anchor="middle">Input</text>
        
        <!-- Arrow to Conv -->
        <line x1="80" y1="195" x2="98" y2="195" stroke="#444" stroke-width="1.5" marker-end="url(#arrowMV)"/>
        
        <!-- ============================================================ -->
        <!-- CONV 3×3 ↓2                                                   -->
        <!-- ============================================================ -->
        
        <rect x="102" y="170" width="38" height="50" fill="{c_conv}" rx="4" filter="url(#shadowMV)"/>
        <text x="121" y="190" fill="#fff" font-size="8" font-family="Inter" text-anchor="middle" font-weight="600">Conv</text>
        <text x="121" y="200" fill="#fff" font-size="7" font-family="Inter" text-anchor="middle">3×3</text>
        <text x="121" y="212" fill="#fff" font-size="7" font-family="Inter" text-anchor="middle" opacity="0.8">↓2</text>
        <text x="121" y="235" fill="#666" font-size="7" font-family="Inter" text-anchor="middle">128×128</text>
        
        <!-- Arrow -->
        <line x1="145" y1="195" x2="163" y2="195" stroke="#444" stroke-width="1.5" marker-end="url(#arrowMV)"/>
        
        <!-- ============================================================ -->
        <!-- MV2 Block                                                     -->
        <!-- ============================================================ -->
        
        <rect x="167" y="175" width="32" height="40" fill="{c_mv2}" rx="4" filter="url(#shadowMV)"/>
        <text x="183" y="192" fill="#000" font-size="8" font-family="Inter" text-anchor="middle" font-weight="600">MV2</text>
        <text x="183" y="205" fill="#000" font-size="7" font-family="Inter" text-anchor="middle" opacity="0.7"></text>
        <text x="183" y="235" fill="#666" font-size="7" font-family="Inter" text-anchor="middle">64×64</text>
        
        <!-- Arrow -->
        <line x1="204" y1="195" x2="218" y2="195" stroke="#444" stroke-width="1.5" marker-end="url(#arrowMV)"/>
        
        <!-- ============================================================ -->
        <!-- MV2 ↓2 Block                                                  -->
        <!-- ============================================================ -->
        
        <rect x="222" y="175" width="32" height="40" fill="{c_mv2}" rx="4" filter="url(#shadowMV)"/>
        <text x="238" y="190" fill="#000" font-size="8" font-family="Inter" text-anchor="middle" font-weight="600">MV2</text>
        <text x="238" y="202" fill="#000" font-size="7" font-family="Inter" text-anchor="middle" opacity="0.8">↓2</text>
        
        <!-- Arrow -->
        <line x1="259" y1="195" x2="273" y2="195" stroke="#444" stroke-width="1.5" marker-end="url(#arrowMV)"/>
        
        <!-- ============================================================ -->
        <!-- 2× MV2 Block                                                  -->
        <!-- ============================================================ -->
        
        <rect x="277" y="175" width="38" height="40" fill="{c_mv2}" rx="4" filter="url(#shadowMV)"/>
        <text x="296" y="185" fill="#000" font-size="7" font-family="Inter" text-anchor="middle">2×</text>
        <text x="296" y="198" fill="#000" font-size="8" font-family="Inter" text-anchor="middle" font-weight="600">MV2</text>
        <text x="296" y="235" fill="#666" font-size="7" font-family="Inter" text-anchor="middle">32×32</text>
        
        <!-- Arrow -->
        <line x1="320" y1="195" x2="334" y2="195" stroke="#444" stroke-width="1.5" marker-end="url(#arrowMV)"/>
        
        <!-- ============================================================ -->
        <!-- MV2↓2 + MobileViT Block (L=2)                                 -->
        <!-- ============================================================ -->
        
        <g transform="translate(338, 155)">
            <!-- MV2 ↓2 -->
            <rect x="0" y="25" width="28" height="30" fill="{c_mv2}" rx="3"/>
            <text x="14" y="40" fill="#000" font-size="7" font-family="Inter" text-anchor="middle" font-weight="600">MV2</text>
            <text x="14" y="50" fill="#000" font-size="6" font-family="Inter" text-anchor="middle">↓2</text>
            
            <!-- MobileViT Block -->
            <rect x="32" y="15" width="38" height="50" fill="{c_mvit}" rx="4" filter="url(#shadowMV)"/>
            <text x="51" y="35" fill="#fff" font-size="7" font-family="Inter" text-anchor="middle" font-weight="600">MobileViT</text>
            <text x="51" y="47" fill="#fff" font-size="7" font-family="Inter" text-anchor="middle">block</text>
            
            <!-- L label -->
            <text x="51" y="8" fill="#888" font-size="8" font-family="Inter" text-anchor="middle" font-style="italic">L = 2</text>
            <text x="51" y="90" fill="#666" font-size="7" font-family="Inter" text-anchor="middle">16×16</text>
            <text x="51" y="78" fill="#666" font-size="6" font-family="Inter" text-anchor="middle">h=w=2</text>
        </g>
        
        <!-- Arrow -->
        <line x1="413" y1="195" x2="427" y2="195" stroke="#444" stroke-width="1.5" marker-end="url(#arrowMV)"/>
        
        <!-- ============================================================ -->
        <!-- MV2↓2 + MobileViT Block (L=4)                                 -->
        <!-- ============================================================ -->
        
        <g transform="translate(431, 155)">
            <!-- MV2 ↓2 -->
            <rect x="0" y="25" width="28" height="30" fill="{c_mv2}" rx="3"/>
            <text x="14" y="40" fill="#000" font-size="7" font-family="Inter" text-anchor="middle" font-weight="600">MV2</text>
            <text x="14" y="50" fill="#000" font-size="6" font-family="Inter" text-anchor="middle">↓2</text>
            
            <!-- MobileViT Block -->
            <rect x="32" y="15" width="38" height="50" fill="{c_mvit}" rx="4" filter="url(#shadowMV)"/>
            <text x="51" y="35" fill="#fff" font-size="7" font-family="Inter" text-anchor="middle" font-weight="600">MobileViT</text>
            <text x="51" y="47" fill="#fff" font-size="7" font-family="Inter" text-anchor="middle">block</text>
            
            <!-- L label -->
            <text x="51" y="8" fill="#888" font-size="8" font-family="Inter" text-anchor="middle" font-style="italic">L = 4</text>
            <text x="51" y="90" fill="#666" font-size="7" font-family="Inter" text-anchor="middle">8×8</text>
            <text x="51" y="78" fill="#666" font-size="6" font-family="Inter" text-anchor="middle">h=w=2</text>
        </g>
        
        <!-- Arrow -->
        <line x1="506" y1="195" x2="520" y2="195" stroke="#444" stroke-width="1.5" marker-end="url(#arrowMV)"/>
        
        <!-- ============================================================ -->
        <!-- MV2↓2 + MobileViT Block (L=3)                                 -->
        <!-- ============================================================ -->
        
        <g transform="translate(524, 155)">
            <!-- MV2 ↓2 -->
            <rect x="0" y="25" width="28" height="30" fill="{c_mv2}" rx="3"/>
            <text x="14" y="40" fill="#000" font-size="7" font-family="Inter" text-anchor="middle" font-weight="600">MV2</text>
            <text x="14" y="50" fill="#000" font-size="6" font-family="Inter" text-anchor="middle">↓2</text>
            
            <!-- MobileViT Block -->
            <rect x="32" y="15" width="38" height="50" fill="{c_mvit}" rx="4" filter="url(#shadowMV)"/>
            <text x="51" y="35" fill="#fff" font-size="7" font-family="Inter" text-anchor="middle" font-weight="600">MobileViT</text>
            <text x="51" y="47" fill="#fff" font-size="7" font-family="Inter" text-anchor="middle">block</text>
            
            <!-- L label -->
            <text x="51" y="8" fill="#888" font-size="8" font-family="Inter" text-anchor="middle" font-style="italic">L = 3</text>
            <text x="51" y="90" fill="#666" font-size="7" font-family="Inter" text-anchor="middle">8×8</text>
            <text x="51" y="78" fill="#666" font-size="6" font-family="Inter" text-anchor="middle">h=w=2</text>
        </g>
        
        <!-- Arrow -->
        <line x1="599" y1="195" x2="613" y2="195" stroke="#444" stroke-width="1.5" marker-end="url(#arrowMV)"/>
        
        <!-- ============================================================ -->
        <!-- Conv 1×1                                                      -->
        <!-- ============================================================ -->
        
        <rect x="617" y="175" width="32" height="40" fill="{c_conv}" rx="4" filter="url(#shadowMV)"/>
        <text x="633" y="190" fill="#fff" font-size="7" font-family="Inter" text-anchor="middle" font-weight="600">Conv</text>
        <text x="633" y="202" fill="#fff" font-size="7" font-family="Inter" text-anchor="middle">1×1</text>
        
        <!-- Arrow -->
        <line x1="654" y1="195" x2="668" y2="195" stroke="#444" stroke-width="1.5" marker-end="url(#arrowMV)"/>
        
        <!-- ============================================================ -->
        <!-- Global Pool + Linear                                          -->
        <!-- ============================================================ -->
        
        <rect x="672" y="165" width="55" height="60" fill="{c_pool}" rx="4" filter="url(#shadowMV)"/>
        <text x="700" y="185" fill="#000" font-size="7" font-family="Inter" text-anchor="middle" font-weight="600">Global pool</text>
        <text x="700" y="198" fill="#000" font-size="8" font-family="Inter" text-anchor="middle">→</text>
        <text x="700" y="212" fill="#000" font-size="7" font-family="Inter" text-anchor="middle" font-weight="600">Linear</text>
        <text x="700" y="245" fill="#666" font-size="7" font-family="Inter" text-anchor="middle">1×1</text>
        
        <!-- Arrow -->
        <line x1="732" y1="195" x2="746" y2="195" stroke="#444" stroke-width="1.5" marker-end="url(#arrowMV)"/>
        
        <!-- ============================================================ -->
        <!-- OUTPUT CLASSES                                                -->
        <!-- ============================================================ -->
        
        <g transform="translate(750, 160)">
            <rect x="0" y="0" width="80" height="70" fill="#1a1a1a" stroke="#333" rx="6"/>
            <text x="40" y="17" fill="#888" font-size="8" font-family="Inter" text-anchor="middle" 
                  letter-spacing="0.5" font-weight="500">Classes</text>
            
            <!-- Cat output -->
            <rect x="8" y="24" width="64" height="18" fill="#111" rx="3"/>
            <text x="16" y="37" fill="{c_cat}" font-size="9" font-family="Inter" font-weight="500">Cat</text>
            <text x="64" y="37" fill="{c_cat}" font-size="9" font-family="monospace" 
                  text-anchor="end" font-weight="600">{cat_score:.1%}</text>
            
            <!-- Dog output -->
            <rect x="8" y="46" width="64" height="18" fill="#111" rx="3"/>
            <text x="16" y="59" fill="{c_dog}" font-size="9" font-family="Inter" font-weight="500">Dog</text>
            <text x="64" y="59" fill="{c_dog}" font-size="9" font-family="monospace" 
                  text-anchor="end" font-weight="600">{dog_score:.1%}</text>
        </g>
        
        <!-- ============================================================ -->
        <!-- LEGEND                                                        -->
        <!-- ============================================================ -->
        
        <g transform="translate(60, 290)">
            <text x="0" y="0" fill="#666" font-size="9" font-family="Inter" font-weight="500">Legend:</text>
            
            <!-- Conv -->
            <rect x="55" y="-10" width="16" height="12" fill="{c_conv}" rx="2"/>
            <text x="76" y="0" fill="#888" font-size="8" font-family="Inter">Conv</text>
            
            <!-- MV2 -->
            <rect x="120" y="-10" width="16" height="12" fill="{c_mv2}" rx="2"/>
            <text x="141" y="0" fill="#888" font-size="8" font-family="Inter">MV2 (InvertedResidual)</text>
            
            <!-- MobileViT -->
            <rect x="265" y="-10" width="16" height="12" fill="{c_mvit}" rx="2"/>
            <text x="286" y="0" fill="#888" font-size="8" font-family="Inter">MobileViT Block</text>
            
            <!-- Pool -->
            <rect x="390" y="-10" width="16" height="12" fill="{c_pool}" rx="2"/>
            <text x="411" y="0" fill="#888" font-size="8" font-family="Inter">Global Pool + Linear</text>
        </g>
        
        <!-- ============================================================ -->
        <!-- Output Spatial Dimensions Label                               -->
        <!-- ============================================================ -->
        
        <text x="45" y="260" fill="#555" font-size="8" font-family="Inter" text-anchor="middle">256×256</text>
        
        <!-- ============================================================ -->
        <!-- Architecture Info                                             -->
        <!-- ============================================================ -->
        
        <text x="200" y="380" fill="#555" font-size="9" font-family="Inter" text-anchor="middle">Parameters: 5.8M</text>
        <text x="425" y="380" fill="#555" font-size="9" font-family="Inter" text-anchor="middle">MACs: 1.84G</text>
        <text x="650" y="380" fill="#555" font-size="9" font-family="Inter" text-anchor="middle">Input: 256×256</text>
        
        <!-- ============================================================ -->
        <!-- Data Flow Label                                               -->
        <!-- ============================================================ -->
        
        <text x="{width/2}" y="{height - 15}" fill="#444" font-size="9" font-family="Inter" 
              text-anchor="middle">← Data Flow: Image → Features → Classification →</text>
        
    </svg>
    '''
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; padding: 0; background: #000; display: flex; justify-content: center; }}
        </style>
    </head>
    <body>{svg}</body>
    </html>
    '''


def generate_results_html(predictions: dict = None, predicted_class: str = None) -> str:
    """
    Generate HTML for results display.
    
    Args:
        predictions: Classification scores {'Cat': x, 'Dog': y}
        predicted_class: The predicted class label
    """
    
    if predictions is None:
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
                * { margin: 0; padding: 0; box-sizing: border-box; }
                html { height: 100%; background: transparent; }
                body {
                    height: 100%;
                    background: #0a0a0a;
                    font-family: 'Inter', sans-serif;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    color: #555;
                    border: 1px solid #222;
                    border-radius: 16px;
                }
                .text { font-size: 14px; text-align: center; line-height: 1.6; }
            </style>
        </head>
        <body>
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#444" stroke-width="1.5" style="margin-bottom: 16px;">
                <circle cx="11" cy="11" r="8"/>
                <line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
            <div class="text">Upload an image<br/>to see results</div>
        </body>
        </html>
        '''
    
    cat_score = predictions.get('Cat', 0)
    dog_score = predictions.get('Dog', 0)
    
    winner_label = 'Cat' if cat_score > dog_score else 'Dog'
    winner_color = '#f97316' if cat_score > dog_score else '#3b82f6'
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            html {{ height: 100%; background: transparent; }}
            body {{
                height: 100%;
                padding: 24px;
                background: #0a0a0a;
                font-family: 'Inter', sans-serif;
                border: 1px solid #222;
                border-radius: 16px;
            }}
            .title {{
                font-size: 11px;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                text-align: center;
                margin-bottom: 24px;
                font-weight: 600;
            }}
            .row {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }}
            .label {{ color: #ddd; font-size: 15px; font-weight: 500; }}
            .score {{ font-size: 18px; font-weight: 700; }}
            .score.cat {{ color: #f97316; }}
            .score.dog {{ color: #3b82f6; }}
            .bar {{
                height: 8px;
                background: #1a1a1a;
                border-radius: 4px;
                margin-bottom: 20px;
                overflow: hidden;
            }}
            .fill {{
                height: 100%;
                border-radius: 4px;
                transition: width 0.4s ease;
            }}
            .fill.cat {{ background: linear-gradient(90deg, #f97316, #fb923c); }}
            .fill.dog {{ background: linear-gradient(90deg, #3b82f6, #60a5fa); }}
            .badge {{
                display: block;
                margin: 16px auto 0;
                padding: 10px 28px;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 600;
                color: white;
                text-align: center;
                background: {winner_color};
            }}
        </style>
    </head>
    <body>
        <div class="title">Prediction Scores</div>
        
        <div class="row">
            <span class="label">Cat</span>
            <span class="score cat">{cat_score:.1%}</span>
        </div>
        <div class="bar"><div class="fill cat" style="width: {cat_score * 100}%"></div></div>
        
        <div class="row">
            <span class="label">Dog</span>
            <span class="score dog">{dog_score:.1%}</span>
        </div>
        <div class="bar"><div class="fill dog" style="width: {dog_score * 100}%"></div></div>
        
        <div class="badge">{winner_label}</div>
    </body>
    </html>
    '''


# =============================================================================
# Model Loading
# =============================================================================

@st.cache_resource
def load_model(model_type: str) -> ModelInference:
    """Load and cache the specified model."""
    return ModelInference(model_type)


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""

    # =========================================================================
    # Startup warm-up (one-time, silent)
    # =========================================================================
    # Streamlit can briefly re-render widgets on the first interaction (e.g., first upload),
    # especially when cache_resource objects are initialized for the first time.
    #
    # There was brief “duplicate uploader” flicker that disappears after switching
    # model once and switching back. We reproduce that behavior automatically ONCE on
    # first app open, before rendering any UI, so the user never notices.
    # This is UI bug that i didnt have time to fix so that's just a UX workaround.

    def _silent_startup_warmup() -> None:
        if st.session_state.get("_warmup_done", False):
            return

        # Step 0 -> set selector to the other option, warm cache, rerun
        step = int(st.session_state.get("_warmup_step", 0))
        if step == 0:
            st.session_state["_warmup_step"] = 1
            # Must match the exact option label used in st.radio(...)
            st.session_state["model_selector"] = "MobileViTv3-S"
            # Warm up the model cache for the non-default branch (even if it's a placeholder).
            _ = load_model("mobilevit")
            st.rerun()

        # Step 1 -> set selector back to default, warm cache, mark done, rerun
        if step == 1:
            st.session_state["_warmup_step"] = 2
            st.session_state["model_selector"] = "Pre-trained ViT"
            _ = load_model("vit")
            st.session_state["_warmup_done"] = True
            st.session_state.pop("_warmup_step", None)
            st.rerun()

    _silent_startup_warmup()
    
    # =========================================================================
    # Header
    # =========================================================================
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0 1.5rem 0; border-bottom: 1px solid #222; margin-bottom: 1.5rem;">
            <h1 style="font-size: 2rem; font-weight: 700; margin: 0 0 0.3rem 0; color: #fff !important;">
                Cat vs Dog Classifier
            </h1>
            <p style="font-size: 0.9rem; color: #666 !important; margin: 0;">
                Deep Learning Image Classification with Vision Transformers
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # =========================================================================
    # Model Selector - Centered
    # =========================================================================
    st.markdown('<p class="section-title" style="margin-bottom: 0.75rem;">Select Classification Model</p>', unsafe_allow_html=True)
    
    # Render radio button in full width - CSS will handle centering
    model_choice = st.radio(
        "model",
        options=["Pre-trained ViT", "MobileViTv3-S"],
        horizontal=True,
        label_visibility="collapsed",
        key="model_selector"
    )
    
    selected_model = 'vit' if model_choice == "Pre-trained ViT" else 'mobilevit'
    
    # Load model
    model = load_model(selected_model)
    
    # State variables
    predictions = None
    predicted_class = None
    image_b64 = None
    
    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
    
    # =========================================================================
    # Main Three-Column Layout
    # =========================================================================
    # Wider center column to accommodate MobileViT architecture (850px)
    col_left, col_center, col_right = st.columns([1, 3.5, 1])
    
    # -------------------------------------------------------------------------
    # LEFT: Input Image
    # -------------------------------------------------------------------------
    with col_left:
        st.markdown('<p class="section-title">Input Image</p>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload",
            type=['jpg', 'jpeg', 'png', 'webp'],
            label_visibility="collapsed",
            key="uploader"
        )
        
        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            
            # Convert to base64 for network visualization preview
            image_b64 = image_to_base64(image)
            
            # Run inference
            if model.is_model_loaded():
                predicted_class, confidence, predictions = model.predict(image)
            
            # Show image
            st.image(image, use_container_width=True)
    
    # -------------------------------------------------------------------------
    # CENTER: Network Visualization
    # -------------------------------------------------------------------------
    with col_center:
        st.markdown('<p class="section-title">Network Architecture</p>', unsafe_allow_html=True)
        
        # Generate architecture visualization based on selected model
        if selected_model == 'vit':
            network_html = generate_vit_architecture_html(predictions, image_b64)
            network_height = 440
        else:
            network_html = generate_mobilevit_architecture_html(predictions, image_b64)
            network_height = 440
        
        components.html(network_html, height=network_height)
        
        # Status badge
        if selected_model == 'vit':
            status = "Ready"
            status_color = "#22c55e"
        else:
            status = "Training in Progress" if not model.is_model_loaded() else "Ready"
            status_color = "#f59e0b" if not model.is_model_loaded() else "#22c55e"
        st.markdown(f'''
            <div style="text-align: center; margin-top: 0.5rem;">
                <span style="display: inline-flex; align-items: center; gap: 8px; 
                            background: #111; padding: 8px 16px; border-radius: 20px; 
                            font-size: 12px; color: #888; border: 1px solid #222;">
                    <span style="width: 8px; height: 8px; border-radius: 50%; background: {status_color};"></span>
                    {status}
                </span>
            </div>
        ''', unsafe_allow_html=True)
    
    # -------------------------------------------------------------------------
    # RIGHT: Results
    # -------------------------------------------------------------------------
    with col_right:
        # Add vertical spacing to center results with network (network ~440px, results ~230px)
        # Offset = (440 - 230) / 2 ≈ 105px, plus ~25px for section title = ~80px
        st.markdown('<div style="height: 80px;"></div>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Classification Results</p>', unsafe_allow_html=True)
        
        results_html = generate_results_html(predictions, predicted_class)
        components.html(results_html, height=230)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
