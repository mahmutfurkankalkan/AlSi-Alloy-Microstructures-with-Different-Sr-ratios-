# AlSi Alloy Microstructures with Different Sr Ratios

This repository provides a set of Python scripts originally developed for the automated analysis of Al-Si alloy microstructures modified with varying Strontium (Sr) levels. The workflows include segmentation, classification, data augmentation, and quantitative feature extraction (e.g., SDAS measurement), and are intended for use in **Google Colab** environments.

> ⚠️ **Note**: The microstructure images have been removed from this repository. If you require access to the original dataset used in this study, please contact the author directly via GitHub or institutional email.

---

## 📁 Repository Contents

- **`Modification Level Classification.py`**  
  Predicts the alloy modification level based on microstructural patterns using classification techniques. Includes basic data augmentation such as resizing and inversion to enhance model robustness.

- **`Segmentation of AlSi alloys.py`**  
  Segments primary α(Al) and eutectic Si phases in grayscale SEM images using classical image processing techniques. Designed for binary mask generation and phase separation.

- **`SDAS_Measurement.py`**  
  Identifies neighboring dendritic arms with parallel orientation and computes the Secondary Dendrite Arm Spacing (SDAS) using geometric and spatial analysis.

---

## ⚙️ Requirements

- Python ≥ 3.7  
- OpenCV  
- NumPy  
- PIL  
- Matplotlib  

> All scripts are developed and tested in **Google Colab**, and are optimized for notebook-based execution in cloud environments.

---

## ⚠️ Important Considerations

### 🖼️ Image Characteristics Matter
The scripts assume grayscale SEM images of fixed size (ideally **512×512 pixels**). If using images with different resolution, magnification, or contrast, you should adjust preprocessing steps such as thresholding, resizing, and filtering.

### 📏 Measurement Units
All area and length measurements (e.g., SDAS) are calculated in **pixel units**. To convert values to microns, appropriate **scale calibration** is required based on magnification or embedded scale bars in the original images.

### ⚙️ Augmentation
Several scripts include simple image augmentation (e.g., inversion, resizing) to account for variability in sample appearance. These are intended to enhance robustness in segmentation and classification stages.

### 🧪 Tool Usage Scope
These scripts are designed for **academic and research purposes only**. They are not production-grade tools and may require adaptation for deployment in automated or industrial pipelines.

---

## 📌 Citation Requirement

If you use or adapt these scripts or workflows in your research, **citation of the original study is mandatory**. This repository is part of a peer-reviewed academic study, and proper attribution supports transparency and research integrity.

> **Suggested citation:**  
> **Kalkan, M.F.** (2025). *Automated Microstructure Analysis and SDAS Measurement of AlSi Alloys Using Python-based Image Processing*. GitHub Repository.

---

## 📬 Image Access

This repository provides the full analysis pipeline. For those interested in applying the scripts to relevant microstructure datasets, example images may be provided upon academic request. Please contact the author for further details.
---

## 👤 Author

**Mahmut Furkan Kalkan**  
📧 *(Contact via GitHub or academic institution for data access and collaboration inquiries.)*

---

## 📜 License

A suitable open-source license (e.g., **MIT** or **CC-BY-NC**) can be applied depending on intended distribution.  
Please contact the author if you have specific reuse scenarios in mind.

