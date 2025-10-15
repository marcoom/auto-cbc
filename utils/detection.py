"""
YOLO detection logic for AutoCBC application.

SPDX-License-Identifier: AGPL-3.0-only
Copyright (c) 2025 Marco Mongi
"""

from ultralytics import YOLO
import streamlit as st
from pathlib import Path
from config import DETECTION_MODEL_PATH, DETECTION_CONFIDENCE, DETECTION_IOU


@st.cache_resource
def load_detection_model():
    """
    Load the YOLO detection model.
    
    Returns:
        YOLO: Loaded YOLO model instance.
        
    Raises:
        FileNotFoundError: If the model file is not found.
        Exception: If model loading fails.
    """
    if not DETECTION_MODEL_PATH.exists():
        raise FileNotFoundError(f"Detection model not found at {DETECTION_MODEL_PATH}")
    
    try:
        model = YOLO(str(DETECTION_MODEL_PATH))
        return model
    except Exception as e:
        raise Exception(f"Failed to load detection model: {str(e)}")


def detect_cells(model, image_path, confidence=DETECTION_CONFIDENCE, iou=DETECTION_IOU):
    """
    Perform cell detection on an image using YOLO model.
    
    Args:
        model: Loaded YOLO model instance.
        image_path (str): Path to the input image.
        confidence (float): Confidence threshold for detection.
        iou (float): IoU threshold for Non-Maximum Suppression.
        
    Returns:
        detection: YOLO detection results object.
        
    Raises:
        Exception: If detection fails.
    """
    try:
        results = model.predict(
            image_path, 
            save=False, 
            conf=confidence, 
            iou=iou,
            verbose=False
        )
        return results[0]  # Return first (and only) result
    except Exception as e:
        raise Exception(f"Detection failed: {str(e)}")


def get_detection_summary(detection):
    """
    Get a summary of detection results.
    
    Args:
        detection: YOLO detection results object.
        
    Returns:
        dict: Summary containing total detections and class names.
    """
    total_detections = len(detection.boxes) if detection.boxes is not None else 0
    class_names = list(detection.names.values()) if detection.names else []
    
    return {
        "total_detections": total_detections,
        "class_names": class_names,
        "has_detections": total_detections > 0
    }