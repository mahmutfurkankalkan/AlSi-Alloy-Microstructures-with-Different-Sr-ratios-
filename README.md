# AlSi Alloy Microstructures with Different Sr Ratios

This repository provides a set of Python scripts originally developed for the automated analysis of Al-Si alloy microstructures modified with varying Strontium (Sr) levels. The workflows include segmentation, classification, and quantitative feature extraction (e.g., SDAS measurement), and are intended for use in **Google Colab** environments.

> ⚠️ **Note**: The microstructure images have been removed from this repository. If you require access to the original dataset used in this study, please contact the author directly via GitHub or institutional email.

---

## 📁 Repository Contents

- **`Modification Level Classification.py`**: Predicts the alloy modification level based on microstructural patterns.
- **`Segmentation of AlSi alloys.py`**: Segments primary and eutectic phases in SEM images.
- **`SDAS_Measurement.py`**: Identifies parallel dendritic features and calculates Secondary Dendrite Arm Spacing (SDAS) metrics.

---

## ⚙️ Requirements

- Python ≥ 3.7
- OpenCV
- NumPy
- PIL
- Matplotlib

All scripts are developed and tested in **Google Colab**, and optimized for notebook-based workflows.

---

## ⚠️ Important Considerations

- **Image Characteristics Matter**  
  The scripts assume uniform grayscale SEM images (ideally 512×512 pixels). If using images with different resolution, magnification, or quality, preprocessing steps (e.g., resizing, thresholding) should be adapted accordingly.

- **Measurement Units**  
  SDAS and area measurements are based on pixel distances. For absolute values (e.g., microns), calibration based on image scale bars is required.

- **Tool Usage Scope**  
  These tools are intended for academic and research purposes. Further engineering effort is needed to adapt them for large-scale or production-grade applications.

---

## 📌 Citation Requirement

If you use or adapt these scripts in your research, **citation of the original study is mandatory**. This repository is part of an academic work, and proper attribution supports transparency and scholarly recognition.

_You may cite as:_  
**Kalkan, M.F.** (2025). *Automated Microstructure Analysis and SDAS Measurement of AlSi Alloys Using Python-based Image Processing*. [GitHub Repository]

---

## 📬 Image Access

Although the images have been removed from this repository for space and distribution reasons, **they are available upon reasonable request**. Please reach out to the author via GitHub or academic email to request access for research or validation purposes.

---

## 👤 Author

Mahmut Furkan Kalkan  
(Feel free to contact for data requests, feedback, or collaboration)

---

## 📜 License

A suitable license such as MIT or CC-BY-NC can be added depending on your preferred distribution terms.
