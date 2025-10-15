"""
Image overlay and visualization for AutoCBC application.
Based on the visualization code from the main_flow_simplified.ipynb notebook.

SPDX-License-Identifier: AGPL-3.0-only
Copyright (c) 2025 Marco Mongi
"""

import numpy as np
from PIL import Image
import cv2
from config import COLORS, DEFAULT_ALPHA, DEFAULT_BORDER_WIDTH, DEFAULT_BORDER_DARKEN_FACTOR


def _find_class_id(names_dict, target_name):
    """Find class ID by name in detection results."""
    target = target_name.strip().lower()
    for cid, name in names_dict.items():
        if str(name).strip().lower() == target:
            return int(cid)
    return None


def _resize_mask_bool(mask_bool, target_hw):
    """Resize boolean mask to target height and width."""
    h, w = target_hw
    if mask_bool.shape == (h, w):
        return mask_bool
    m = (mask_bool.astype(np.uint8) * 255)
    m_resized = Image.fromarray(m).resize((w, h), resample=Image.Resampling.NEAREST)
    return (np.array(m_resized) > 127)


def _darken(color_rgb_uint8, factor=0.7):
    """Darken an RGB uint8 color by factor (0..1)."""
    return np.clip(color_rgb_uint8.astype(np.float32) * float(factor), 0, 255).astype(np.uint8)


def _make_border(mask_bool, width_px):
    """
    Create inner border of given pixel width using erosion.
    border = mask & ~erode(mask, iterations=width_px)
    """
    if width_px <= 0:
        return np.zeros_like(mask_bool, dtype=bool)
    m = (mask_bool.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(m, kernel, iterations=int(width_px))
    border = cv2.bitwise_and(m, cv2.bitwise_not(eroded))
    return border > 0


def build_class_masks(detection, segmentation, img_shape_hw):
    """
    Build individual masks for each detected cell from detection and segmentation results.

    Args:
        detection: YOLO detection results.
        segmentation: SAM segmentation results.
        img_shape_hw (tuple): Target image shape (height, width).

    Returns:
        tuple: (individual_cells, bg_mask) where:
            - individual_cells: list of (mask, class_name) tuples, one per cell
            - bg_mask: boolean numpy array for background pixels

    Raises:
        ValueError: If expected cell types not found in detection results.
    """
    # SAM masks -> numpy bool (N, Hm, Wm)
    masks_t = segmentation.masks.data  # torch.Tensor [N, Hm, Wm]
    masks_np = masks_t.cpu().numpy().astype(bool)

    # Class IDs for each bbox (same order as bboxes passed to SAM)
    cls_ids = detection.boxes.cls.cpu().numpy().astype(int)

    n = min(len(cls_ids), masks_np.shape[0])
    masks_np = masks_np[:n]
    cls_ids = cls_ids[:n]

    names = detection.names
    rbc_id = _find_class_id(names, 'RBC')
    wbc_id = _find_class_id(names, 'WBC')
    pl_id = _find_class_id(names, 'Platelet')

    if any(x is None for x in [rbc_id, wbc_id, pl_id]):
        raise ValueError(
            f"Expected class names not found in detection.names: {names}. "
            "Adjust _find_class_id if your model uses different labels."
        )

    # Map class IDs to names
    class_map = {rbc_id: 'RBC', wbc_id: 'WBC', pl_id: 'Platelet'}

    H, W = img_shape_hw
    individual_cells = []
    union_mask = np.zeros((H, W), dtype=bool)

    for i in range(n):
        cid = int(cls_ids[i])
        if cid in class_map:
            m = _resize_mask_bool(masks_np[i], (H, W))
            individual_cells.append((m, class_map[cid]))
            union_mask |= m

    bg_mask = ~union_mask  # Background = pixels that belong to no cell

    return individual_cells, bg_mask


def render_overlay(
    img_rgb,
    individual_cells, bg_mask,
    show_rbc=True, show_wbc=True, show_platelet=True, show_bg=False,
    alpha=DEFAULT_ALPHA,
    draw_borders=True,
    border_width=DEFAULT_BORDER_WIDTH,
    border_darken_factor=DEFAULT_BORDER_DARKEN_FACTOR
):
    """
    Render overlay on base image with alpha blending.

    Args:
        img_rgb (numpy.ndarray): Base RGB image.
        individual_cells (list): List of (mask, class_name) tuples for individual cells.
        bg_mask (numpy.ndarray): Boolean mask for background pixels.
        show_rbc, show_wbc, show_platelet, show_bg (bool): Visibility toggles for each type.
        alpha (float): Alpha blending factor (0-1).
        draw_borders (bool): Whether to draw borders around cells.
        border_width (int): Border width in pixels.
        border_darken_factor (float): Factor to darken border colors.

    Returns:
        tuple: (output_image, overlay_only) as numpy arrays.

    Notes:
        - Borders are drawn for each individual cell independently.
        - Borders are drawn in the same hue but darkened by border_darken_factor.
        - Border width is applied as an inner border.
        - If cell masks overlap, later cells override earlier ones in the overlay.
    """
    H, W = img_rgb.shape[:2]
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    comp_mask = np.zeros((H, W), dtype=bool)

    # Define visibility flags
    show_flags = {'RBC': show_rbc, 'WBC': show_wbc, 'Platelet': show_platelet}

    # Helper to paint fill + optional border for one individual cell
    def _paint_cell(mask_bool, base_color):
        nonlocal overlay, comp_mask
        if not np.any(mask_bool):
            return
        # Fill
        overlay[mask_bool] = base_color
        comp_mask |= mask_bool

        # Border (inner) - drawn for THIS individual cell
        if draw_borders and border_width > 0:
            border_mask = _make_border(mask_bool, border_width)
            if np.any(border_mask):
                overlay[border_mask] = _darken(base_color, border_darken_factor)
                comp_mask |= border_mask

    # Process each individual cell
    for cell_mask, class_name in individual_cells:
        if show_flags.get(class_name, False):
            _paint_cell(cell_mask, COLORS[class_name])

    # Background has no border
    if show_bg:
        overlay[bg_mask] = COLORS['Background']
        comp_mask |= bg_mask

    # Alpha blend only where overlay is present
    out = img_rgb.copy().astype(np.float32)
    overlay_f = overlay.astype(np.float32)
    mask3 = np.stack([comp_mask]*3, axis=-1)

    out[mask3] = (1.0 - alpha) * out[mask3] + alpha * overlay_f[mask3]
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out, overlay


def create_overlay_image(img_rgb, detection, segmentation,
                        show_rbc=True, show_wbc=True, show_platelet=True, show_bg=False,
                        transparency=DEFAULT_ALPHA * 100):
    """
    Create overlay image from detection and segmentation results.

    Args:
        img_rgb (numpy.ndarray): Base RGB image.
        detection: YOLO detection results.
        segmentation: SAM segmentation results.
        show_rbc, show_wbc, show_platelet, show_bg (bool): Visibility toggles.
        transparency (int): Transparency percentage (0-100).

    Returns:
        numpy.ndarray: Overlay image as RGB array.

    Raises:
        Exception: If overlay creation fails.
    """
    try:
        H, W = img_rgb.shape[:2]
        alpha = transparency / 100.0

        # Build individual cell masks
        individual_cells, bg_mask = build_class_masks(detection, segmentation, (H, W))

        # Render overlay
        overlay_img, _ = render_overlay(
            img_rgb,
            individual_cells, bg_mask,
            show_rbc=show_rbc,
            show_wbc=show_wbc,
            show_platelet=show_platelet,
            show_bg=show_bg,
            alpha=alpha
        )

        return overlay_img
    except Exception as e:
        raise Exception(f"Overlay creation failed: {str(e)}")