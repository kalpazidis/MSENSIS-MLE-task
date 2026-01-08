# =============================================================================
# MSENSIS ML Engineering Task - Neural Network Visualization Module
# =============================================================================
"""
This module provides SVG-based neural network architecture visualization.

It generates interactive visual representations of the neural network
architectures used in the classification task.

Author: Interview Candidate
Date: January 2026
"""

from typing import List, Dict, Tuple, Optional


class NetworkVisualizer:
    """
    Generates SVG visualizations of neural network architectures.
    
    Creates visual representations showing layers, connections, and
    activation patterns for different model architectures.
    
    Attributes:
        model_type (str): Type of model to visualize
    """
    
    # =========================================================================
    # Architecture Definitions
    # =========================================================================
    
    # ViT (Vision Transformer) architecture simplified representation
    VIT_ARCHITECTURE = {
        'name': 'Vision Transformer (ViT-Base)',
        'layers': [
            {'name': 'Input\n224×224×3', 'neurons': 3, 'type': 'input'},
            {'name': 'Patch\nEmbedding', 'neurons': 16, 'type': 'embedding'},
            {'name': 'Transformer\nBlock ×12', 'neurons': 12, 'type': 'transformer'},
            {'name': 'MLP\nHead', 'neurons': 8, 'type': 'mlp'},
            {'name': 'Output', 'neurons': 2, 'type': 'output'}
        ],
        'total_params': '86M',
        'description': 'Pre-trained on ImageNet-21k, fine-tuned on cats vs dogs'
    }
    
    # MobileViTv3-S architecture simplified representation
    MOBILEVIT_ARCHITECTURE = {
        'name': 'MobileViTv3-S',
        'layers': [
            {'name': 'Input\n256×256×3', 'neurons': 3, 'type': 'input'},
            {'name': 'Conv\nStem', 'neurons': 8, 'type': 'conv'},
            {'name': 'MobileViT\nBlock ×3', 'neurons': 10, 'type': 'mobilevit'},
            {'name': 'Global\nPool', 'neurons': 6, 'type': 'pool'},
            {'name': 'Output', 'neurons': 2, 'type': 'output'}
        ],
        'total_params': '5.8M',
        'description': 'Lightweight transformer for mobile devices'
    }
    
    # Color scheme for different layer types
    LAYER_COLORS = {
        'input': '#4CAF50',      # Green
        'embedding': '#2196F3',   # Blue
        'transformer': '#9C27B0', # Purple
        'mlp': '#FF9800',         # Orange
        'output': '#F44336',      # Red
        'conv': '#00BCD4',        # Cyan
        'mobilevit': '#673AB7',   # Deep Purple
        'pool': '#795548'         # Brown
    }
    
    def __init__(self, model_type: str = 'vit'):
        """
        Initialize the visualizer for a specific model type.
        
        Args:
            model_type (str): 'vit' or 'mobilevit'
        """
        self.model_type = model_type
        self.architecture = (
            self.VIT_ARCHITECTURE if model_type == 'vit' 
            else self.MOBILEVIT_ARCHITECTURE
        )
    
    # =========================================================================
    # SVG Generation Methods
    # =========================================================================
    
    def generate_svg(
        self, 
        width: int = 700, 
        height: int = 400,
        predictions: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Generate an SVG representation of the neural network.
        
        Args:
            width (int): SVG canvas width
            height (int): SVG canvas height
            predictions (dict, optional): Prediction scores to display on output layer
            
        Returns:
            str: SVG markup string
        """
        layers = self.architecture['layers']
        num_layers = len(layers)
        
        # Calculate spacing
        layer_spacing = width / (num_layers + 1)
        vertical_center = height / 2
        
        # Start SVG
        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            self._generate_defs(),
            f'<rect width="{width}" height="{height}" fill="#0a0a0f" rx="12"/>',
        ]
        
        # Generate connections first (so they appear behind nodes)
        svg_parts.append(self._generate_connections(layers, layer_spacing, vertical_center, height))
        
        # Generate layer nodes
        for i, layer in enumerate(layers):
            x = layer_spacing * (i + 1)
            layer_predictions = predictions if layer['type'] == 'output' else None
            svg_parts.append(
                self._generate_layer(layer, x, vertical_center, height, layer_predictions)
            )
        
        # Add title
        svg_parts.append(
            f'<text x="{width/2}" y="30" fill="#ffffff" font-size="16" '
            f'font-family="Segoe UI, Arial" text-anchor="middle" font-weight="600">'
            f'{self.architecture["name"]}</text>'
        )
        
        # Add params info
        svg_parts.append(
            f'<text x="{width/2}" y="{height-15}" fill="#888888" font-size="12" '
            f'font-family="Segoe UI, Arial" text-anchor="middle">'
            f'Parameters: {self.architecture["total_params"]}</text>'
        )
        
        svg_parts.append('</svg>')
        
        return '\n'.join(svg_parts)
    
    def _generate_defs(self) -> str:
        """
        Generate SVG definitions (gradients, filters, etc.).
        
        Returns:
            str: SVG defs element
        """
        defs = ['<defs>']
        
        # Glow filter for nodes
        defs.append('''
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
        ''')
        
        # Gradients for each layer type
        for layer_type, color in self.LAYER_COLORS.items():
            defs.append(f'''
                <radialGradient id="grad_{layer_type}" cx="50%" cy="30%" r="70%">
                    <stop offset="0%" style="stop-color:{color};stop-opacity:1" />
                    <stop offset="100%" style="stop-color:{self._darken_color(color)};stop-opacity:1" />
                </radialGradient>
            ''')
        
        defs.append('</defs>')
        return '\n'.join(defs)
    
    def _generate_layer(
        self, 
        layer: Dict, 
        x: float, 
        center_y: float, 
        height: float,
        predictions: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Generate SVG for a single layer with its neurons.
        
        Args:
            layer (dict): Layer configuration
            x (float): X position
            center_y (float): Y center position
            height (float): Canvas height
            predictions (dict, optional): Predictions for output layer
            
        Returns:
            str: SVG group element for the layer
        """
        parts = [f'<g class="layer" data-type="{layer["type"]}">']
        
        num_neurons = layer['neurons']
        neuron_radius = min(15, (height - 100) / (num_neurons * 3))
        spacing = min(30, (height - 100) / num_neurons)
        
        # Calculate starting Y position to center neurons
        start_y = center_y - (num_neurons - 1) * spacing / 2
        
        color = self.LAYER_COLORS.get(layer['type'], '#888888')
        
        # Draw neurons
        for j in range(num_neurons):
            y = start_y + j * spacing
            
            # Special handling for output neurons with predictions
            if predictions and layer['type'] == 'output':
                label = 'Cat' if j == 0 else 'Dog'
                score = predictions.get(label, 0)
                opacity = 0.3 + (score * 0.7)  # Scale opacity based on prediction
                
                parts.append(
                    f'<circle cx="{x}" cy="{y}" r="{neuron_radius + 5}" '
                    f'fill="url(#grad_{layer["type"]})" opacity="{opacity}" filter="url(#glow)"/>'
                )
                # Add score label
                parts.append(
                    f'<text x="{x + neuron_radius + 25}" y="{y + 5}" '
                    f'fill="#ffffff" font-size="11" font-family="Segoe UI, Arial">'
                    f'{label}: {score:.1%}</text>'
                )
            else:
                parts.append(
                    f'<circle cx="{x}" cy="{y}" r="{neuron_radius}" '
                    f'fill="url(#grad_{layer["type"]})" opacity="0.9"/>'
                )
        
        # Add layer label
        label_y = center_y + (num_neurons * spacing / 2) + 35
        name_lines = layer['name'].split('\n')
        for idx, line in enumerate(name_lines):
            parts.append(
                f'<text x="{x}" y="{label_y + idx * 14}" fill="#cccccc" '
                f'font-size="10" font-family="Segoe UI, Arial" text-anchor="middle">'
                f'{line}</text>'
            )
        
        parts.append('</g>')
        return '\n'.join(parts)
    
    def _generate_connections(
        self, 
        layers: List[Dict], 
        spacing: float, 
        center_y: float,
        height: float
    ) -> str:
        """
        Generate connection lines between layers.
        
        Args:
            layers (list): Layer configurations
            spacing (float): Horizontal spacing between layers
            center_y (float): Vertical center
            height (float): Canvas height
            
        Returns:
            str: SVG group element with connection lines
        """
        parts = ['<g class="connections" opacity="0.3">']
        
        for i in range(len(layers) - 1):
            layer1 = layers[i]
            layer2 = layers[i + 1]
            
            x1 = spacing * (i + 1)
            x2 = spacing * (i + 2)
            
            # Calculate neuron positions for both layers
            n1 = layer1['neurons']
            n2 = layer2['neurons']
            spacing1 = min(30, (height - 100) / n1)
            spacing2 = min(30, (height - 100) / n2)
            
            start_y1 = center_y - (n1 - 1) * spacing1 / 2
            start_y2 = center_y - (n2 - 1) * spacing2 / 2
            
            # Draw simplified connections (not all-to-all for clarity)
            for j in range(min(n1, 4)):  # Limit connections for visual clarity
                y1 = start_y1 + (j * (n1 - 1) / 3 if n1 > 1 else 0) * spacing1
                for k in range(min(n2, 4)):
                    y2 = start_y2 + (k * (n2 - 1) / 3 if n2 > 1 else 0) * spacing2
                    parts.append(
                        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                        f'stroke="#4a90d9" stroke-width="0.5"/>'
                    )
        
        parts.append('</g>')
        return '\n'.join(parts)
    
    def _darken_color(self, hex_color: str) -> str:
        """
        Darken a hex color for gradient effect.
        
        Args:
            hex_color (str): Hex color string (e.g., '#FF0000')
            
        Returns:
            str: Darkened hex color
        """
        # Remove # and convert to RGB
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Darken by 40%
        factor = 0.6
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
        
        return f'#{r:02x}{g:02x}{b:02x}'
    
    # =========================================================================
    # Architecture Info Methods
    # =========================================================================
    
    def get_architecture_info(self) -> Dict:
        """
        Get detailed architecture information.
        
        Returns:
            Dictionary with architecture details
        """
        return self.architecture
    
    def get_layer_summary(self) -> List[Dict]:
        """
        Get a summary of all layers.
        
        Returns:
            List of layer summaries
        """
        return [
            {
                'name': layer['name'].replace('\n', ' '),
                'type': layer['type'],
                'neurons': layer['neurons']
            }
            for layer in self.architecture['layers']
        ]

