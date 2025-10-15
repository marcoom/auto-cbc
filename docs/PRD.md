# Product Requirements Document (PRD)
## Automatic Blood Cell Counter

---

## 1. Overview

### 1.1 Product Description
AutoCBC (Automatic Complete Blood Count) is a Streamlit-based application for automatic detection and counting of blood cells (red blood cells, white blood cells, and platelets) from microscopy images. The system uses a two-stage deep learning pipeline combining object detection (YOLOv11 Nano) and instance segmentation (SAM2).

### 1.2 Target Platform
- Web application built with Streamlit framework
- The Web app could potentially be accessed from a PC or a Smartphone

### 1.3 License
- **Project License**: AGPL-3.0 (required by Ultralytics YOLO dependency)
- The project shall comply with AGPL-3.0 terms and conditions
- Source code availability is mandatory when deployed as a service
---

## 2. Functional Requirements

### 2.1 Image Input
- **FR-1.1**: The system shall allow users to upload images from their device gallery
- **FR-1.2**: The system shall allow users to capture images using their device camera
- **FR-1.3**: Supported image formats: JPEG, PNG, JPG
- **FR-1.4**: The system shall validate image format and display appropriate error messages for unsupported formats

### 2.2 Cell Detection and Segmentation
- **FR-2.1**: The system shall use a pre-trained YOLOv11 Nano model for initial cell detection
- **FR-2.2**: The YOLO model shall be obtained from the training project: https://github.com/marcoom/yolo-auto-cbc-training
- **FR-2.3**: Model training is out of scope for this project
- **FR-2.4**: The system shall detect three cell types:
  - Red Blood Cells (RBC)
  - White Blood Cells (WBC)
  - Platelets
- **FR-2.5**: The system shall use SAM2 (Segment Anything Model 2) for instance segmentation
- **FR-2.6**: Each bounding box from YOLO shall be used as a prompt for SAM2 segmentation
- **FR-2.7**: The system shall maintain cell type classification from YOLO detection through segmentation

### 2.3 Visualization
- **FR-3.1**: The system shall display the original image with colored segmentation overlays
- **FR-3.2**: Color scheme:
  - Red Blood Cells: Red
  - White Blood Cells: Yellow
  - Platelets: Blue
  - Background: Black mask
- **FR-3.3**: The system shall provide a global slider control for overlay transparency (0-100%)
- **FR-3.4**: The system shall provide independent checkboxes to toggle visibility of:
  - Red Blood Cells overlay
  - White Blood Cells overlay
  - Platelets overlay
  - Background mask
- **FR-3.5**: The transparency slider shall affect all visible overlays simultaneously
- **FR-3.6**: Unchecked cell types shall not be displayed in the overlay
- **FR-3.7**: The overlay visibility shall be adjustable in real-time without reprocessing
- **FR-3.8**: When cells overlap, the last drawn layer shall be visible (painter's algorithm)
- **FR-3.9**: Background shall be defined as all pixels that do not belong to any detected cell (inverse of the union of all cell segmentation masks)

### 2.4 Metrics and Results
- **FR-4.1**: The system shall count total cells for each type
- **FR-4.2**: The system shall calculate percentage distribution of each cell type
- **FR-4.3**: The system shall display results in a pie chart showing:
  - Percentage of each cell type
  - Total count for each cell type
- **FR-4.4**: The pie chart shall use the same color scheme as the segmentation overlay

### 2.5 User Interface
- **FR-5.1**: The interface shall be intuitive and require minimal user interaction
- **FR-5.2**: All UI text, labels, and messages shall be in English
- **FR-5.3**: The system shall provide clear loading indicators during processing
- **FR-5.4**: Results shall be displayed immediately after processing completes

---

## 3. Technical Requirements

### 3.1 Architecture
- **TR-1.1**: The application shall be built using Streamlit framework
- **TR-1.2**: The system shall follow a modular architecture with clear separation of concerns:
  - Image input/preprocessing module
  - Detection module (YOLO)
  - Segmentation module (SAM2)
  - Visualization module
  - Metrics calculation module

### 3.2 Models
- **TR-2.1**: Detection model: Pre-trained YOLOv11 Nano from https://github.com/marcoom/yolo-auto-cbc-training
- **TR-2.2**: The trained model file (cbc_detection.pt) shall be imported from the training project
- **TR-2.3**: Segmentation model: SAM2 (Meta)
- **TR-2.4**: Models shall be loaded efficiently to minimize startup time
- **TR-2.5**: Model inference shall be optimized for reasonable processing time
- **TR-2.6**: Model training and retraining is not part of this project's scope

### 3.3 Dependencies
- **TR-3.1**: Python 3.12
- **TR-3.2**: Core libraries:
  - streamlit
  - ultralytics (for YOLOv11)
  - segment-anything-2
  - opencv-python
  - numpy
  - pillow
  - matplotlib or plotly (for pie chart)
- **TR-3.3**: All dependencies shall be specified in a requirements.txt file
- **TR-3.4**: Third-party licenses:
  - Ultralytics YOLO (AGPL-3.0) - unmodified
  - SAM2 (Apache-2.0)
  - cctorch (BSD-3-Clause)

### 3.4 Containerization
- **TR-4.1**: The project shall include a Dockerfile for building a container image
- **TR-4.2**: The Docker image shall include all necessary dependencies
- **TR-4.3**: The Dockerfile shall use Python 3.12 as base image
- **TR-4.4**: Container configuration shall be optimized for Streamlit deployment
---

## 4. Code Quality Standards

### 4.1 Clean Code Principles
- **CQ-1.1**: Follow Clean Code principles throughout the codebase
- **CQ-1.2**: Code shall be self-explanatory with meaningful variable and function names
- **CQ-1.3**: Functions shall be small and focused on a single responsibility
- **CQ-1.4**: Avoid code duplication (DRY principle)
- **CQ-1.5**: Keep complexity low - prefer simple, straightforward implementations over clever solutions

### 4.2 Python Best Practices
- **CQ-2.1**: Follow Pythonic coding style and idioms
- **CQ-2.2**: Use appropriate Python data structures and built-in functions
- **CQ-2.3**: Follow PEP 8 style guide for code formatting
- **CQ-2.4**: Use type hints where they improve code clarity
- **CQ-2.5**: Handle exceptions appropriately and gracefully

### 4.3 Documentation
- **CQ-3.1**: Follow PEP 257 conventions for docstrings
- **CQ-3.2**: All public functions and classes shall have docstrings
- **CQ-3.3**: Docstrings shall include:
  - Brief description
  - Parameters with types
  - Return values with types
  - Raised exceptions (if any)
- **CQ-3.4**: All documentation shall be in English

### 4.4 Comments
- **CQ-4.1**: Add comments only when they provide value beyond what the code expresses
- **CQ-4.2**: Comments shall explain WHY, not WHAT (the code should be self-explanatory for WHAT)
- **CQ-4.3**: Avoid redundant comments that simply restate the code
- **CQ-4.4**: Update or remove comments when code changes
- **CQ-4.5**: All comments shall be in English

### 4.5 Code Organization
- **CQ-5.1**: Use meaningful module and file names
- **CQ-5.2**: Group related functions and classes logically
- **CQ-5.3**: Keep files focused and reasonably sized
- **CQ-5.4**: Use consistent naming conventions throughout the project

### 4.6 Maintainability
- **CQ-6.1**: Code shall be easy to understand for future modifications
- **CQ-6.2**: Magic numbers shall be replaced with named constants
- **CQ-6.3**: Configuration parameters shall be easily accessible and modifiable
- **CQ-6.4**: The codebase shall be structured to facilitate testing and debugging

---

## 5. User Experience Requirements

### 5.1 Usability
- **UX-1.1**: The application shall be usable without technical knowledge
- **UX-1.2**: Processing feedback shall be clear and informative
- **UX-1.3**: Error messages shall be user-friendly and actionable

### 5.2 Performance
- **UX-2.1**: Image processing shall complete in reasonable time (target: < 30 seconds for typical images)
- **UX-2.2**: The UI shall remain responsive during processing
- **UX-2.3**: Overlay transparency adjustments shall be instantaneous

---

## 6. Deliverables

### 6.1 Code
- **D-1.1**: Complete Streamlit application
- **D-1.2**: requirements.txt with all dependencies
- **D-1.3**: README.md with setup and usage instructions (in English)
- **D-1.4**: Dockerfile for building container image
- **D-1.5**: LICENSE file (AGPL-3.0)
- **D-1.6**: THIRD_PARTY_LICENSES directory with third-party license files

### 6.2 Project Structure
```
auto-cbc/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Project dependencies
├── README.md                  # Setup and usage instructions
├── Dockerfile                 # Docker container configuration
├── LICENSE                    # AGPL-3.0 license
├── THIRD_PARTY_LICENSES/      # Third-party licenses
│   ├── LICENSE_SAM2
│   └── LICENSE_cctorch
├── models/                    # Model files (imported from training project)
│   └── cbc_detection.pt       # Pre-trained YOLOv11 Nano model
├── docs/                      # Relevant documents
├── notebooks/                 # Jupyter Notebooks
├── utils/                     # Utility functions
│   ├── detection.py           # YOLO detection logic
│   ├── segmentation.py        # SAM2 segmentation logic
│   ├── visualization.py       # Image overlay and visualization
│   └── metrics.py             # Counting and statistics
└── config.py                  # Configuration constants
```

**Related Repositories:**
- Main project: https://github.com/marcoom/auto-cbc
- YOLO training project: https://github.com/marcoom/yolo-auto-cbc-training

---

## 7. Success Criteria

### 7.1 Functional Success
- The application successfully detects and segments all three cell types
- Cell counting is accurate and consistent
- Visualization clearly distinguishes different cell types
- The application runs on both desktop and mobile browsers

### 7.2 Code Quality Success
- Code passes PEP 8 linting with minimal exceptions
- All functions have appropriate docstrings
- Code is understandable without excessive comments
- The codebase follows established Python best practices

---

## 8. Out of Scope

The following items are explicitly out of scope for this version:
- Model training and fine-tuning (handled by separate training project: https://github.com/marcoom/yolo-auto-cbc-training)
- Automated testing (unit tests, integration tests, etc.)
- User authentication or data persistence
- Batch processing of multiple images
- Export of results to external formats (PDF, Excel, etc.)
- Advanced image preprocessing or enhancement
- Comparison with ground truth or accuracy metrics
- Historical analysis or trend tracking
- Integration with laboratory information systems

---

## 9. Future Considerations

Potential enhancements for future iterations:
- Export functionality for reports
- Batch processing capabilities
- Cell morphology analysis
- Abnormality detection
- Result comparison and tracking over time