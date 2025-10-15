"""
Image overlay and visualization for AutoCBC application.
Based on the visualization code from the main_flow_simplified.ipynb notebook.

SPDX-License-Identifier: AGPL-3.0-only
Copyright (c) 2025 Marco Mongi
"""

import numpy as np
from PIL import Image
import cv2
import plotly.graph_objects as go
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
            - individual_cells: list of (mask, class_name, confidence) tuples, one per cell
            - bg_mask: boolean numpy array for background pixels

    Raises:
        ValueError: If expected cell types not found in detection results.
    """
    # SAM masks -> numpy bool (N, Hm, Wm)
    masks_t = segmentation.masks.data  # torch.Tensor [N, Hm, Wm]
    masks_np = masks_t.cpu().numpy().astype(bool)

    # Class IDs for each bbox (same order as bboxes passed to SAM)
    cls_ids = detection.boxes.cls.cpu().numpy().astype(int)

    # Confidence scores for each detection
    confidences = detection.boxes.conf.cpu().numpy().astype(float)

    n = min(len(cls_ids), masks_np.shape[0])
    masks_np = masks_np[:n]
    cls_ids = cls_ids[:n]
    confidences = confidences[:n]

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
            individual_cells.append((m, class_map[cid], confidences[i]))
            union_mask |= m

    bg_mask = ~union_mask  # Background = pixels that belong to no cell

    return individual_cells, bg_mask


def render_overlay(
    img_rgb,
    individual_cells, bg_mask,
    show_rbc=True, show_wbc=True, show_platelet=True,
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
        show_rbc, show_wbc, show_platelet (bool): Visibility toggles for each type.
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
    for cell_data in individual_cells:
        # Support both old format (mask, class_name) and new format (mask, class_name, confidence)
        if len(cell_data) == 2:
            cell_mask, class_name = cell_data
        else:
            cell_mask, class_name, _ = cell_data  # Ignore confidence for static overlay

        if show_flags.get(class_name, False):
            _paint_cell(cell_mask, COLORS[class_name])

    # Alpha blend only where overlay is present
    out = img_rgb.copy().astype(np.float32)
    overlay_f = overlay.astype(np.float32)
    mask3 = np.stack([comp_mask]*3, axis=-1)

    out[mask3] = (1.0 - alpha) * out[mask3] + alpha * overlay_f[mask3]
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out, overlay


def create_overlay_image(img_rgb, detection, segmentation,
                        show_rbc=True, show_wbc=True, show_platelet=True,
                        transparency=DEFAULT_ALPHA * 100):
    """
    Create overlay image from detection and segmentation results.

    Args:
        img_rgb (numpy.ndarray): Base RGB image.
        detection: YOLO detection results.
        segmentation: SAM segmentation results.
        show_rbc, show_wbc, show_platelet (bool): Visibility toggles.
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
            alpha=alpha
        )

        return overlay_img
    except Exception as e:
        raise Exception(f"Overlay creation failed: {str(e)}")


def create_interactive_overlay(img_rgb, detection, segmentation,
                               show_rbc=True, show_wbc=True, show_platelet=True,
                               transparency=DEFAULT_ALPHA * 100):
    """
    Create interactive Plotly overlay figure from detection and segmentation results.

    Args:
        img_rgb (numpy.ndarray): Base RGB image.
        detection: YOLO detection results.
        segmentation: SAM segmentation results.
        show_rbc, show_wbc, show_platelet (bool): Visibility toggles.
        transparency (int): Transparency percentage (0-100).

    Returns:
        plotly.graph_objects.Figure: Interactive Plotly figure with hover functionality.

    Raises:
        Exception: If overlay creation fails.
    """
    try:
        H, W = img_rgb.shape[:2]
        alpha = transparency / 100.0

        # Build individual cell masks with confidence
        individual_cells, bg_mask = build_class_masks(detection, segmentation, (H, W))

        # Define visibility flags
        show_flags = {'RBC': show_rbc, 'WBC': show_wbc, 'Platelet': show_platelet}

        # Create Plotly figure
        fig = go.Figure()

        # Add base image as background
        fig.add_trace(go.Image(
            z=img_rgb,
            hoverinfo='skip'
        ))

        # Process each individual cell
        for cell_mask, class_name, confidence in individual_cells:
            if not show_flags.get(class_name, False):
                continue

            # Extract contours from mask
            mask_uint8 = cell_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Get color for this cell type
            color = COLORS[class_name]
            r, g, b = int(color[0]), int(color[1]), int(color[2])

            # Darken color for border
            border_color = _darken(color, DEFAULT_BORDER_DARKEN_FACTOR)
            br, bg_val, bb = int(border_color[0]), int(border_color[1]), int(border_color[2])

            # Process each contour (usually one per cell, but could be multiple)
            for contour in contours:
                # Simplify contour to reduce complexity
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Extract coordinates
                if len(approx) >= 3:  # Need at least 3 points for a polygon
                    x_coords = approx[:, 0, 0].tolist()
                    y_coords = approx[:, 0, 1].tolist()

                    # Close the polygon
                    x_coords.append(x_coords[0])
                    y_coords.append(y_coords[0])

                    # Add filled polygon for cell with hover
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        fill='toself',
                        fillcolor=f'rgba({r},{g},{b},{alpha})',
                        line=dict(
                            width=DEFAULT_BORDER_WIDTH,
                            color=f'rgb({br},{bg_val},{bb})'
                        ),
                        mode='lines',
                        hoverinfo='text',
                        text=f"<b>{class_name}</b><br>Confidence: {confidence*100:.1f}%",
                        hoverlabel=dict(
                            bgcolor=f'rgb({r},{g},{b})',
                            font_size=16,
                            font_color='black'
                        ),
                        showlegend=False
                    ))

        # Configure layout
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(
                visible=False,
                range=[0, W],
                showgrid=False
            ),
            yaxis=dict(
                visible=False,
                range=[H, 0],  # Invert Y axis to match image coordinates
                scaleanchor='x',
                scaleratio=1,
                showgrid=False
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='closest',
            autosize=True
        )

        return fig

    except Exception as e:
        raise Exception(f"Interactive overlay creation failed: {str(e)}")