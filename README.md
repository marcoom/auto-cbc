<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- Copyright (c) 2025 Marco Mongi -->

# AutoCBC
Automated Blood Cell Counting using AI — A Streamlit-based application that analyzes blood smear images to detect, count, and segment red blood cells, white blood cells, and platelets using YOLO and SAM models.

## Installation

### Prerequisites

- Python 3.12.3
- NVIDIA GPU with CUDA support (optional, but recommended for faster training)

### Steps

1. **Clone the Repository**

     ```bash
   git clone https://github.com/marcoom/auto-cbc.git
   cd auto-cbc
   ```

2. **Create a Virtual Environment**

     ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**

     ```bash
   pip install -r requirements.txt
   ```

## Licenses

AutoCBC is released under **AGPL-3.0**. See the [`LICENSE`](./LICENSE) file.

It includes/uses third-party software:

- **Ultralytics YOLO** (AGPL-3.0) — unmodified
- **SAM 2** (Apache-2.0) — see THIRD_PARTY_LICENSES/LICENSE_SAM2
- **cctorch** (BSD-3-Clause) — see THIRD_PARTY_LICENSES/LICENSE_cctorch

If you use the online Streamlit service, the corresponding source code is available in this repository in accordance with AGPL-3.0.
