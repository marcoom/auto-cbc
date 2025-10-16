"""
AutoCBC - Automated Blood Cell Counting Streamlit Application

SPDX-License-Identifier: AGPL-3.0-only
Copyright (c) 2025 Marco Mongi
"""

import streamlit as st
import tempfile
import os
from PIL import Image
import numpy as np
from pathlib import Path

# Import utility modules
from utils.detection import load_detection_model, detect_cells, get_detection_summary
from utils.segmentation import load_segmentation_model, segment_cells, get_segmentation_summary
from utils.metrics import count_cells, create_pie_chart
from utils.visualization import create_interactive_overlay
from config import SUPPORTED_FORMATS, TRANSPARENCY_MIN, TRANSPARENCY_MAX, TRANSPARENCY_DEFAULT


def get_example_images():
    """
    Get list of example images from media/test_images directory.
    
    Returns:
        list: List of image file paths.
    """
    examples_dir = Path("media/test_images")
    if not examples_dir.exists():
        return []
    
    image_files = []
    for ext in SUPPORTED_FORMATS:
        image_files.extend(examples_dir.glob(f"*.{ext}"))
    
    return sorted(image_files)


def create_thumbnail(image_path, size=(250, 250)):
    """
    Create thumbnail for image display.
    
    Args:
        image_path (Path): Path to image file.
        size (tuple): Thumbnail size.
        
    Returns:
        PIL.Image: Thumbnail image.
    """
    try:
        img = Image.open(image_path)
        img.thumbnail(size, Image.Resampling.LANCZOS)
        return img
    except Exception:
        return None


def setup_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="AutoCBC - Automated Blood Cell Counter",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def display_header():
    """Display application header."""
    st.title("🔬 AutoCBC - Automated Blood Cell Counter")
    st.markdown("""
    Automated detection and counting of blood cells (RBC, WBC, Platelets) from microscopy images.
    Upload an image or take a photo to get started.
    """)


def validate_image_format(uploaded_file):
    """
    Validate uploaded image format.
    
    Args:
        uploaded_file: Streamlit UploadedFile object.
        
    Returns:
        bool: True if format is supported, False otherwise.
    """
    if uploaded_file is None:
        return False
    
    file_extension = uploaded_file.name.split('.')[-1].lower()
    return file_extension in SUPPORTED_FORMATS


def save_uploaded_file(uploaded_file):
    """
    Save uploaded file to temporary location.
    
    Args:
        uploaded_file: Streamlit UploadedFile object.
        
    Returns:
        str: Path to saved temporary file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def create_sidebar_controls():
    """
    Create sidebar controls for visualization settings.

    Returns:
        dict: Dictionary containing all control values.
    """
    st.sidebar.header("🎛️ Visualization Controls")

    # Transparency slider
    transparency = st.sidebar.slider(
        "Overlay Transparency (%)",
        min_value=TRANSPARENCY_MIN,
        max_value=TRANSPARENCY_MAX,
        value=TRANSPARENCY_DEFAULT,
        help="Adjust transparency of cell overlays"
    )
    
    st.sidebar.subheader("Cell Type Visibility")
    
    # Cell type toggles
    show_rbc = st.sidebar.checkbox("🔴 Red Blood Cells (RBC)", value=True)
    show_wbc = st.sidebar.checkbox("🟡 White Blood Cells (WBC)", value=True)
    show_platelet = st.sidebar.checkbox("🔵 Platelets", value=True)
    
    # Detection parameters
    st.sidebar.subheader("⚙️ Detection Settings")
    confidence = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Minimum confidence threshold for cell detection"
    )
    
    iou = st.sidebar.slider(
        "IoU Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Intersection over Union threshold for non-maximum suppression"
    )

    # Layout selection
    layout = st.sidebar.selectbox(
        "Choose layout:",
        ["Horizontal", "Vertical"],
        index=0,
        help="Horizontal: Image and results side-by-side (optimal for wide screens)\nVertical: Image on top, results below"
    )

    return {
        'layout': layout,
        'transparency': transparency,
        'show_rbc': show_rbc,
        'show_wbc': show_wbc,
        'show_platelet': show_platelet,
        'confidence': confidence,
        'iou': iou
    }


def process_image(image_path, controls):
    """
    Process image through the complete pipeline.
    
    Args:
        image_path (str): Path to image file.
        controls (dict): Dictionary containing control values.
        
    Returns:
        tuple: (detection_results, segmentation_results, cell_counts, overlay_image) or None if error.
    """
    try:
        # Load models
        with st.spinner("Loading AI models..."):
            detection_model = load_detection_model()
            segmentation_model = load_segmentation_model()
        
        # Run detection
        with st.spinner("Detecting cells..."):
            detection = detect_cells(
                detection_model, 
                image_path, 
                confidence=controls['confidence'],
                iou=controls['iou']
            )
            detection_summary = get_detection_summary(detection)
        
        if not detection_summary['has_detections']:
            st.warning("No cells detected in the image. Try adjusting the detection confidence.")
            return None
        
        # Check if detection boxes are available
        if detection.boxes is None:
            st.error("Detection failed. No bounding boxes generated.")
            return None
        
        # Run segmentation
        with st.spinner("Segmenting cells..."):
            segmentation = segment_cells(
                segmentation_model, 
                image_path, 
                detection.boxes.xyxy
            )
            segmentation_summary = get_segmentation_summary(segmentation)
        
        if not segmentation_summary['has_masks']:
            st.error("Segmentation failed. No masks generated.")
            return None
        
        # Count cells
        with st.spinner("Counting cells..."):
            cell_counts = count_cells(detection)
        
        # Create interactive overlay
        with st.spinner("Creating visualization..."):
            # Load original image and ensure RGB format
            original_img = Image.open(image_path)
            if original_img.mode != 'RGB':
                original_img = original_img.convert('RGB')
            img_array = np.array(original_img)

            overlay_fig = create_interactive_overlay(
                img_array,
                detection,
                segmentation,
                show_rbc=controls['show_rbc'],
                show_wbc=controls['show_wbc'],
                show_platelet=controls['show_platelet'],
                transparency=controls['transparency']
            )

        return detection, segmentation, cell_counts, overlay_fig
        
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        return None


def display_results(original_img, overlay_fig, cell_counts, layout_mode="Horizontal"):
    """
    Display processing results.

    Args:
        original_img (PIL.Image): Original image.
        overlay_fig (plotly.graph_objects.Figure): Interactive overlay figure.
        cell_counts (dict): Cell count dictionary.
        layout_mode (str): Layout mode - "Horizontal" or "Vertical".
    """

    if layout_mode == "Horizontal":
        # Horizontal layout: Image on left, charts/metrics on right

        # Create two columns: wider for image, narrower for results
        img_col, results_col = st.columns([2, 1], vertical_alignment='center')

        # Display interactive overlay in left column
        with img_col:
            st.plotly_chart(overlay_fig, use_container_width=True, key="overlay_horizontal")

        # Display results in right column
        with results_col:

            # Pie chart
            if cell_counts and sum(cell_counts.values()) > 0:
                fig = create_pie_chart(cell_counts)
                st.plotly_chart(fig)

            # Metrics in a single row (one column per cell type)
            st.markdown("**Cell Counts:**")
            num_metrics = len(cell_counts) + 1  # +1 for total
            metric_cols = st.columns(num_metrics)

            # Display each cell type count in its own column
            for idx, (cell_type, count) in enumerate(cell_counts.items()):
                with metric_cols[idx]:
                    st.metric(label=cell_type, value=count)

            # Total count in last column
            with metric_cols[-1]:
                st.metric(label="Total Cells", value=sum(cell_counts.values()))

    else:
        # Vertical layout: Image on top, charts/metrics below (original behavior)

        st.plotly_chart(overlay_fig, use_container_width=True, key="overlay_vertical")

        # Display metrics

        # Create two columns: pie chart and metrics
        chart_col, metrics_col = st.columns([1, 1], vertical_alignment='center')

        # Pie chart in left column
        with chart_col:
            if cell_counts and sum(cell_counts.values()) > 0:
                fig = create_pie_chart(cell_counts)
                st.plotly_chart(fig)

        # Metrics in right column (one row per cell type)
        with metrics_col:
            st.markdown("**Cell Counts:**")
            # Display each cell type count in its own row
            for cell_type, count in cell_counts.items():
                cols = st.columns(1)
                with cols[0]:
                    st.metric(label=cell_type, value=count)

            # Total count in its own row
            cols = st.columns(1)
            with cols[0]:
                st.metric(label="Total Cells", value=sum(cell_counts.values()))


def main():
    """Main application function."""
    # Setup page
    setup_page()
    display_header()
    
    # Check if model file exists
    from config import DETECTION_MODEL_PATH
    if not DETECTION_MODEL_PATH.exists():
        st.error(f"""
        **Detection model not found!**
        
        Please ensure the trained YOLO model is available at: `{DETECTION_MODEL_PATH}`
        
        You can download it from the training repository:
        https://github.com/marcoom/yolo-auto-cbc-training
        """)
        st.stop()
    
    # Create sidebar controls
    controls = create_sidebar_controls()
    
    # Input Section with expandable container
    with st.expander("📤 Input & Analysis", expanded=not st.session_state.get('show_results', False)):
        # File upload section
        st.subheader("📤 Upload Image")
        
        # Choose input method
        input_method = st.radio(
            "Choose input method:",
            ["📁 Upload File", "📷 Take Photo"],
            horizontal=True
        )
        
        uploaded_file = None
        
        if input_method == "📁 Upload File":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=SUPPORTED_FORMATS,
                help=f"Supported formats: {', '.join([fmt.upper() for fmt in SUPPORTED_FORMATS])}"
            )
            
            # Show example images gallery below the file uploader
            if uploaded_file is None:  # Only show examples if no file is uploaded
                example_images = get_example_images()
                
                if example_images:
                    with st.expander("📋 Or Select an Example Image", expanded=False):
                        # Create columns for thumbnail display
                        cols_per_row = 4
                        rows = [example_images[i:i + cols_per_row] for i in range(0, len(example_images), cols_per_row)]
                        
                        for row in rows:
                            cols = st.columns(len(row))
                            for i, img_path in enumerate(row):
                                with cols[i]:
                                    # Check if this image is currently selected
                                    is_selected = (
                                        'selected_example_image' in st.session_state and 
                                        st.session_state.selected_example_image == str(img_path)
                                    )
                                    
                                    # Create container with border if selected
                                    with st.container(border=is_selected, horizontal_alignment="center"):
                                        # Create thumbnail
                                        thumbnail = create_thumbnail(img_path)
                                        if thumbnail:
                                            st.image(thumbnail, width='content')
                                            if st.button(f"Select", key=f"select_{img_path.name}"):
                                                # Store selected image in session state
                                                st.session_state.selected_example_image = str(img_path)
                                                st.rerun()
            
            # Check if an example image was selected
            if uploaded_file is None and 'selected_example_image' in st.session_state:
                selected_image_path = st.session_state.selected_example_image
                
                # Read the selected image and convert it to a format similar to uploaded file
                with open(selected_image_path, 'rb') as f:
                    img_bytes = f.read()
                
                # Create a mock uploaded file object
                class MockUploadedFile:
                    def __init__(self, name, content):
                        self.name = name
                        self._content = content
                    
                    def getvalue(self):
                        return self._content
                
                uploaded_file = MockUploadedFile(Path(selected_image_path).name, img_bytes)
                
                # Add a button to clear selection
                if st.button("🗑️ Clear Selection"):
                    del st.session_state.selected_example_image
                    st.rerun()
        
        elif input_method == "📷 Take Photo":
            camera_image = st.camera_input("Take a photo")
            if camera_image is not None:
                uploaded_file = camera_image
        
        # Process uploaded image
        if uploaded_file is not None:
            # Validate format
            if not validate_image_format(uploaded_file):
                st.error(f"Unsupported file format. Please use: {', '.join([fmt.upper() for fmt in SUPPORTED_FORMATS])}")
                st.stop()
            
            # Save to temporary file
            temp_path = save_uploaded_file(uploaded_file)
            
            try:
                # Load and display original image, ensure RGB format
                original_img = Image.open(temp_path)
                if original_img.mode != 'RGB':
                    original_img = original_img.convert('RGB')
                
                st.success(f"Image uploaded successfully!")
                
                # Process button
                if st.button("🚀 Analyze Blood Cells", type="primary"):
                    # Process image
                    results = process_image(temp_path, controls)
                    
                    if results is not None:
                        detection, segmentation, cell_counts, _ = results
                        
                        # Store results in session state for real-time updates
                        st.session_state.results = {
                            'original_img': original_img,
                            'detection': detection,
                            'segmentation': segmentation,
                            'cell_counts': cell_counts,
                            'temp_path': temp_path
                        }
                        # Enable results expansion after processing
                        st.session_state.show_results = True
                        st.rerun()
            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    # Results Section with expandable container
    with st.expander("📊 Results", expanded=st.session_state.get('show_results', False)):
        # Display results if available and update overlay based on current controls
        if 'results' in st.session_state:
            results = st.session_state.results

            # Recreate interactive overlay with current settings
            try:
                current_overlay_fig = create_interactive_overlay(
                    np.array(results['original_img']),
                    results['detection'],
                    results['segmentation'],
                    show_rbc=controls['show_rbc'],
                    show_wbc=controls['show_wbc'],
                    show_platelet=controls['show_platelet'],
                    transparency=controls['transparency']
                )

                display_results(
                    results['original_img'],
                    current_overlay_fig,
                    results['cell_counts'],
                    layout_mode=controls['layout']
                )
            except Exception as e:
                st.error(f"Failed to update overlay: {str(e)}")
                # Create a simple figure with just the original image as fallback
                import plotly.graph_objects as go
                fallback_fig = go.Figure()
                fallback_fig.add_trace(go.Image(z=np.array(results['original_img'])))
                fallback_fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False)
                )
                display_results(
                    results['original_img'],
                    fallback_fig,
                    results['cell_counts'],
                    layout_mode=controls['layout']
                )
        else:
            st.info("Process an image above to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **AutoCBC** - Automated Blood Cell Counter  
    Built using Streamlit, YOLO, and SAM2  
    License: AGPL-3.0 | © 2025 Marco Mongi
    """)


if __name__ == "__main__":
    main()
