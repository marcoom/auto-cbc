"""
Configuration constants for AutoCBC application.

SPDX-License-Identifier: AGPL-3.0-only
Copyright (c) 2025 Marco Mongi
"""

import numpy as np
from pathlib import Path

# File paths
MODEL_DIR = Path("models")
DETECTION_MODEL_PATH = MODEL_DIR / "cbc_detection.pt"
SAM_MODEL_PATH = MODEL_DIR / "sam2.1_t.pt"

# Supported image formats
SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]

# Cell type names (must match YOLO model classes)
CELL_TYPES = {
    "RBC": "RBC",
    "WBC": "WBC", 
    "PLATELET": "Platelet"
}

# Color scheme for visualization (pastel palette from notebook)
COLORS = {
    'RBC': np.array([255, 150, 150], dtype=np.uint8),        # pastel red
    'WBC': np.array([255, 255, 150], dtype=np.uint8),        # yellow
    'Platelet': np.array([150, 180, 255], dtype=np.uint8),   # pastel blue
}

# Color scheme for pie chart (RGB format for plotly)
PIE_CHART_COLORS = {
    'RBC': 'rgb(255, 100, 100)',
    'WBC': 'rgb(255, 255, 100)', 
    'Platelet': 'rgb(100, 100, 255)'
}

# Detection parameters
DETECTION_CONFIDENCE = 0.5
DETECTION_IOU = 0.7

# Visualization parameters
DEFAULT_ALPHA = 0.35
DEFAULT_BORDER_WIDTH = 2
DEFAULT_BORDER_DARKEN_FACTOR = 0.7

# UI Configuration
TRANSPARENCY_MIN = 0
TRANSPARENCY_MAX = 100
TRANSPARENCY_DEFAULT = 70