# Experiment.ipynb: Deep Learning Experiments
**Transformer Language Model & OCR Text Detection**

*(Vietnamese version below / Phiên bản Tiếng Việt ở phía dưới)*

---

## 🇬🇧 [ENGLISH] Project Description

This repository contains a unified Jupyter Notebook (`Experiment.ipynb`) that implements two distinct Deep Learning projects. The notebook is designed for educational purposes, demonstrating the implementation of modern architecture concepts in PyTorch.

### 1. Mini Transformer Language Model
A character-level language model trained on the **Tiny Shakespeare** dataset to explore and compare efficient attention mechanisms.

**Key Features:**
* **Architecture**: A custom Transformer implementation built from scratch.
* **Attention Mechanisms**:
    * `self`: Standard Multi-Head Self-Attention.
    * `mqa`: Multi-Query Attention (shares Key/Value heads for memory efficiency).
    * `gqa`: Grouped-Query Attention (a balance between performance and speed).
* **Visualization**: Includes utilities to generate Attention Heatmaps to visualize model focus.
* **Auto-download**: Automatically fetches the dataset upon execution.

**Configuration (`CONFIG` dict):**
* `ATTN_TYPE`: Choose between 'self', 'mqa', or 'gqa'.
* `SEQ_LEN`: Context window size (Default: 48).
* `EMBED_DIM`: Embedding dimension (Default: 192).

### 2. Text Detection (OCR)
A Computer Vision project focused on detecting text regions in images using Semantic Segmentation.

**Key Features:**
* **Architecture**: ResNet18-based Encoder-Decoder network.
    * *Encoder*: Pre-trained ResNet18 (Feature extraction).
    * *Decoder*: Upsampling layers to produce a binary segmentation mask.
* **Pipeline**: Image preprocessing -> Segmentation Prediction -> Contour Extraction (OpenCV) -> Bounding Box Visualization.
* **Dataset**: Custom `TextOCRDataset` handling images and CSV-based bounding box annotations.

### Requirements
To run the notebook, install the following dependencies:
```bash
pip install torch torchvision opencv-python matplotlib pandas tqdm requests evaluate datasets
