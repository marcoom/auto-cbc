"""
Cell counting and statistics for AutoCBC application.

SPDX-License-Identifier: AGPL-3.0-only
Copyright (c) 2025 Marco Mongi
"""

import plotly.express as px
import pandas as pd
from config import PIE_CHART_COLORS


def count_cells(detection):
    """
    Count the number of detected cells by type.

    Args:
        detection: The detection object from YOLO model prediction.

    Returns:
        dict: Dictionary where keys are cell type names and values are their counts.
    """
    if detection.boxes is None or len(detection.boxes) == 0:
        return {}
    
    class_ids = list(detection.names.keys())
    counts = {}
    
    for class_id in class_ids:
        class_name = detection.names.get(class_id)
        count = (detection.boxes.cls == class_id).sum().item()
        counts[class_name] = count

    return counts


def calculate_percentages(cell_counts):
    """
    Calculate percentage distribution of cell types.
    
    Args:
        cell_counts (dict): Dictionary with cell type counts.
        
    Returns:
        dict: Dictionary with cell type percentages.
    """
    total = sum(cell_counts.values())
    if total == 0:
        return {}
    
    percentages = {}
    for cell_type, count in cell_counts.items():
        percentages[cell_type] = (count / total) * 100
    
    return percentages


def create_pie_chart(cell_counts):
    """
    Create a pie chart of cell counts.

    Args:
        cell_counts (dict): Dictionary where keys are cell type names and values are their counts.
        
    Returns:
        plotly.graph_objects.Figure: Pie chart figure.
    """
    if not cell_counts or sum(cell_counts.values()) == 0:
        # Return empty figure if no data
        fig = px.pie(values=[], names=[], title="No cells detected")
        return fig

    df_counts = pd.DataFrame(list(cell_counts.items()), columns=['Cell Type', 'Count'])

    fig = px.pie(
        df_counts,
        values='Count',
        names='Cell Type',
        color='Cell Type',
        color_discrete_map=PIE_CHART_COLORS,
        hover_data=['Count'],
        labels={'Count': 'Count'}
    )

    fig.update_layout(
        title={
            'text': "Percentage of Detected Cell Types",
            'y':1,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    # Update traces to show text inside and customize appearance
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',  # Show only percentage and label, count on hover
        hovertemplate='<b>%{label}</b>: %{value} (%{percent})<extra></extra>'
    )

    fig.update_layout(showlegend=False)

    fig.update_layout(
        font=dict(
            size=20
        )
    )

    return fig


def get_metrics_summary(cell_counts):
    """
    Get a comprehensive summary of metrics.
    
    Args:
        cell_counts (dict): Dictionary with cell type counts.
        
    Returns:
        dict: Summary containing counts, percentages, and totals.
    """
    total_cells = sum(cell_counts.values())
    percentages = calculate_percentages(cell_counts)
    
    return {
        "counts": cell_counts,
        "percentages": percentages,
        "total_cells": total_cells,
        "has_cells": total_cells > 0
    }