"""
SAM2 segmentation logic for AutoCBC application.

SPDX-License-Identifier: AGPL-3.0-only
Copyright (c) 2025 Marco Mongi
"""

from ultralytics import SAM
import streamlit as st
from config import SAM_MODEL_PATH, MODEL_DIR


@st.cache_resource
def load_segmentation_model():
    """
    Load the SAM2 segmentation model.
    
    Returns:
        SAM: Loaded SAM model instance.
        
    Raises:
        Exception: If model loading fails.
    """
    try:
        # Ensure models directory exists
        MODEL_DIR.mkdir(exist_ok=True)
        
        # Load model with explicit path to models/ folder
        model = SAM(str(SAM_MODEL_PATH))
        return model
    except Exception as e:
        raise Exception(f"Failed to load segmentation model: {str(e)}")


def segment_cells(model, image_path, detection_boxes):
    """
    Perform cell segmentation using SAM2 model with bounding boxes from YOLO.
    
    Args:
        model: Loaded SAM model instance.
        image_path (str): Path to the input image.
        detection_boxes: Bounding boxes from YOLO detection (xyxy format).
        
    Returns:
        segmentation: SAM segmentation results object.
        
    Raises:
        Exception: If segmentation fails.
    """
    try:
        if detection_boxes is None or len(detection_boxes) == 0:
            raise ValueError("No bounding boxes provided for segmentation")
            
        results = model(image_path, bboxes=detection_boxes, verbose=False)
        return results[0]  # Return first (and only) result
    except Exception as e:
        raise Exception(f"Segmentation failed: {str(e)}")


def get_segmentation_summary(segmentation):
    """
    Get a summary of segmentation results.
    
    Args:
        segmentation: SAM segmentation results object.
        
    Returns:
        dict: Summary containing mask information.
    """
    has_masks = segmentation.masks is not None
    mask_count = len(segmentation.masks.data) if has_masks else 0
    
    return {
        "has_masks": has_masks,
        "mask_count": mask_count
    }