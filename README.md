# pytorch-graphstam

**pytorch-graphstam** is a Python library built on [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) to simplify working with graph neural networks based forecasting models.

---

## Table of Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Library Installation](#library-installation)
- [Quick Start](#quick-start)

---

## Installation

### Prerequisites

1. **PyTorch (version >= 2.5.1)**  
   - Find detailed instructions at the [official PyTorch website](https://pytorch.org/get-started/locally/).  
   - Example installation command for CUDA 11.8:
     ```bash
     pip install torch --index-url https://download.pytorch.org/whl/cu118
     ```
   - **Note**: Replace `cu118` with the CUDA version appropriate for your GPU setup.

2. **PyTorch Geometric (version >= 2.6.1)**  
   - Refer to the [official PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
   - Basic installation:
     ```bash
     pip install torch_geometric
     ```
   - Additional dependencies for CUDA 11.8 (example):
     ```bash
     pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
       -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
     ```
   - **Note**: Ensure that your `torch_geometric` and related dependencies match the PyTorch and CUDA versions installed. 

### Library Installation

Once the prerequisites are installed, install **pytorch-graphstam** from GitHub:

```bash
pip install git+https://github.com/rsscml/pytorch-graphstam
```

## Quick Start

See the examples directory for complete worked out notebooks & docs directory for notes on architecture & other nuances.
