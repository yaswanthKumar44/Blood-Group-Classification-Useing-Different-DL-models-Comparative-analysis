# Blood Group Classification Using Different DL Models - Comparative Analysis

![GitHub License](https://img.shields.io/github/license/yaswanthKumar44/Blood-Group-Classification-Useing-Different-DL-models-Comparative-analysis)
![GitHub Stars](https://img.shields.io/github/stars/yaswanthKumar44/Blood-Group-Classification-Useing-Different-DL-models-Comparative-analysis)

## Overview

This repository contains the source code, notebooks, and assets for a comparative study of various deep‑learning models applied to blood group classification. The project explores multiple architectures (EfficientNetV2‑S, ConvNeXt, MobileViT, ResNet‑50, RepVGG, etc.) and provides a **Flask API** for inference, as well as Jupyter notebooks for training and analysis.

## Dataset

- **Source:** Publicly available blood‑cell image datasets (e.g., `Blood-Cell-Image-Data`).
- **Structure:** Images are organized by blood group (`A`, `B`, `AB`, `O`).
- **Pre‑processing:** Images are resized to 224×224, normalized, and optionally augmented.

## Model Architectures

| Architecture        | Accuracy (%) |
| ------------------- | ------------ |
| EfficientNet‑V2‑S | 98.14        |
| ConvNeXt‑Tiny      | 97.21        |
| MobileViT‑V2       | 97.07        |
| ResNet‑50          | 97.20        |
| RepVGG              | 96.98        |
| ...                 | ...          |

Each model folder contains the training script, saved `.pth` checkpoint, and a Jupyter notebook that reproduces the results.

## Installation

```bash
# Clone the repository
git clone https://github.com/yaswanthKumar44/Blood-Group-Classification-Useing-Different-DL-models-Comparative-analysis.git
cd Blood-Group-Classification-Useing-Different-DL-models-Comparative-analysis

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` file is generated from the notebooks and includes PyTorch, torchvision, Flask, and other utilities.

## Usage

### Training a Model

```bash
python train.py --model efficientnet_v2s --epochs 30 --batch-size 32
```

The script will save the checkpoint in the corresponding model folder.

### Inference via Flask API

```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000`. Upload an image of a blood cell to get the predicted blood group and a Grad‑CAM heat‑map.

### Running a Notebook

Open any `.ipynb` file in Jupyter:

```bash
jupyter notebook
```

Select the notebook of interest (e.g., `EfficientNet_V2_S.ipynb`).

## Project Structure

```
├── Efficientnet_V2s_98.14/
│   ├── EfficientNet_V2_S.ipynb   # Training & analysis notebook
│   ├── best_model.pth            # Saved checkpoint
│   └── ...
├── ConvNext_Tiny_97.21/
│   └── ...
├── MobileVitv2_97.07/
│   └── ...
├── app.py                       # Flask API entry point
├── static/                      # CSS, JS, images for the web UI
├── templates/                   # HTML templates for Flask
├── requirements.txt
├── README.md                    # **This file**
└── .gitignore
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and ensure they pass existing notebooks.
4. Submit a pull request with a clear description of the changes.

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The original dataset providers.
- The PyTorch and Hugging Face communities for model implementations.
- All contributors who helped improve the models and documentation.

---
