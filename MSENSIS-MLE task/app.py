# =============================================================================
# MSENSIS ML Engineering Task - Main Application
# =============================================================================
"""
Cat vs Dog Image Classifier - Streamlit Web Application

This application provides a web interface for classifying images of cats and dogs
using pre-trained deep learning models. It features an elegant UI with neural
network visualization and real-time inference.

Author: Interview Candidate
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
        max-width: 1400px !important;
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
    Generate MobileViTv3 architecture visualization.
    
    Args:
        predictions: Classification scores
        image_b64: Base64 encoded image for input preview
    """
    
    width = 700
    height = 420
    
    cat_score = predictions.get('Cat', 0) if predictions else 0
    dog_score = predictions.get('Dog', 0) if predictions else 0
    
    c_conv = "#06B6D4"   # Cyan for conv
    c_mv = "#8B5CF6"     # Purple for MobileViT blocks
    c_cat = "#F97316"
    c_dog = "#3B82F6"
    
    # Determine input image section
    if image_b64:
        input_image_section = f'''
            <defs>
                <clipPath id="imgClip2">
                    <rect x="37" y="142" width="66" height="66" rx="4"/>
                </clipPath>
            </defs>
            <rect x="35" y="140" width="70" height="70" fill="#1a1a1a" stroke="#22c55e" stroke-width="2" rx="6"/>
            <image x="37" y="142" width="66" height="66" 
                   href="{image_b64}" 
                   preserveAspectRatio="xMidYMid slice"
                   clip-path="url(#imgClip2)"/>
        '''
    else:
        input_image_section = '''
            <rect x="35" y="140" width="70" height="70" fill="#1a1a1a" stroke="#333" rx="6"/>
            <text x="70" y="175" fill="#444" font-size="20" font-family="Inter" text-anchor="middle">?</text>
        '''
    
    svg = f'''
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
        <defs>
            <filter id="shadow2" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.3"/>
            </filter>
            <marker id="arrow2" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#444"/>
            </marker>
        </defs>
        
        <!-- Background -->
        <rect width="{width}" height="{height}" fill="#0a0a0a" rx="16"/>
        <rect x="1" y="1" width="{width-2}" height="{height-2}" fill="none" stroke="#222" rx="15"/>
        
        <!-- Title -->
        <text x="{width/2}" y="28" fill="#fff" font-size="15" font-family="Inter, sans-serif" 
              text-anchor="middle" font-weight="600">MobileViTv3-S Architecture</text>
        <text x="{width/2}" y="46" fill="#666" font-size="11" font-family="Inter, sans-serif" 
              text-anchor="middle">Lightweight Vision Transformer for Mobile Devices</text>
        
        <!-- Input Image -->
        {input_image_section}
        <text x="70" y="230" fill="#888" font-size="10" font-family="Inter" text-anchor="middle">Input</text>
        <text x="70" y="243" fill="#666" font-size="9" font-family="Inter" text-anchor="middle">256 × 256</text>
        
        <!-- Arrow -->
        <line x1="110" y1="175" x2="140" y2="175" stroke="#444" stroke-width="1.5" marker-end="url(#arrow2)"/>
        
        <!-- Architecture blocks placeholder -->
        <rect x="145" y="120" width="400" height="180" fill="#111" stroke="#333" rx="12" stroke-dasharray="6,3"/>
        
        <text x="345" y="175" fill="#555" font-size="14" font-family="Inter" text-anchor="middle" font-weight="500">
            Custom Model - To Be Trained
        </text>
        <text x="345" y="200" fill="#444" font-size="11" font-family="Inter" text-anchor="middle">
            Conv Stem → MobileViT Blocks ×3 → Global Pool → Classifier
        </text>
        <text x="345" y="225" fill="#666" font-size="10" font-family="Inter" text-anchor="middle">
            Efficient architecture optimized for mobile deployment
        </text>
        <text x="345" y="270" fill="#555" font-size="9" font-family="Inter" text-anchor="middle">
            Parameters: 5.8M | FLOPs: ~1.8G
        </text>
        
        <!-- Arrow to output -->
        <line x1="550" y1="210" x2="580" y2="210" stroke="#444" stroke-width="1.5" marker-end="url(#arrow2)"/>
        
        <!-- Output placeholder -->
        <g transform="translate(585, 170)">
            <rect x="0" y="0" width="90" height="80" fill="#1a1a1a" stroke="#333" rx="8"/>
            <text x="45" y="20" fill="#888" font-size="9" font-family="Inter" text-anchor="middle">Output</text>
            <text x="45" y="42" fill="{c_cat}" font-size="10" font-family="Inter" text-anchor="middle">Cat: --</text>
            <text x="45" y="62" fill="{c_dog}" font-size="10" font-family="Inter" text-anchor="middle">Dog: --</text>
        </g>
        
        <!-- Status badge -->
        <rect x="280" y="340" width="140" height="36" fill="#1a1a1a" stroke="#ef4444" rx="8" opacity="0.8"/>
        <text x="{width/2}" y="363" fill="#ef4444" font-size="11" font-family="Inter" text-anchor="middle" font-weight="500">
            Not Yet Available
        </text>
        
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
    """Generate HTML for results display."""
    
    if predictions is None:
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
                body {
                    margin: 0; padding: 0;
                    background: #0a0a0a;
                    font-family: 'Inter', sans-serif;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    min-height: 280px;
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
            body {{
                margin: 0; padding: 24px;
                background: #0a0a0a;
                font-family: 'Inter', sans-serif;
                border: 1px solid #222;
                border-radius: 16px;
                box-sizing: border-box;
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
    # You observed that the brief “duplicate uploader” flicker disappears after switching
    # model once and switching back. We reproduce that behavior automatically ONCE on
    # first app open, before rendering any UI, so the user never notices.
    #
    # This is an interview-task UX workaround, not a production pattern.
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
    
    selected_model = 'vit' if "ViT" in model_choice else 'mobilevit'
    
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
    col_left, col_center, col_right = st.columns([1, 2.5, 1])
    
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
        else:
            network_html = generate_mobilevit_architecture_html(predictions, image_b64)
        
        components.html(network_html, height=440)
        
        # Status badge
        status = "Ready" if selected_model == 'vit' else "Model Not Available"
        status_color = "#22c55e" if selected_model == 'vit' else "#ef4444"
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
        st.markdown('<p class="section-title">Classification Results</p>', unsafe_allow_html=True)
        
        results_html = generate_results_html(predictions, predicted_class)
        components.html(results_html, height=300)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
