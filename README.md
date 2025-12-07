# Color Based Object Tracking on the Fruits360 Dataset

This project implements a simple color based tracker for a single fruit class (Apple Golden 1) using HSV segmentation and a learned mask area threshold.  
All experiments are run in a single Jupyter notebook.

---

## 1. Project structure

Suggested layout of the repository:

- `README.md` (this file)  
- `color_tracker.ipynb`  main notebook with all code, figures and results  
- `requirements.txt`     Python dependencies (optional but recommended)  

The notebook contains:

1. Dataset download and loading  
2. HSV range estimation for Apple Golden 1  
3. Mask generation and mask area ratio computation  
4. Threshold search on the training subset  
5. Evaluation on the test subset  
6. Plots and figures used in the report  

---

## 2. Requirements

Tested with:

- Python 3.10 or newer  
- Jupyter Notebook or JupyterLab  

Python packages:

- `torch`  
- `torchvision`  
- `numpy`  
- `matplotlib`  
- `opencv-python`  
- `kagglehub`  

Install them with:

```bash
pip install torch torchvision numpy matplotlib opencv-python kagglehub
```

If `torch` is already installed in your environment, you can skip it here.

---

## 3. Dataset

The project uses the Fruits360 dataset (100 x 100 version) from Kaggle.

- Dataset: `moltean/fruits`  
- Subset used:  
  - 5000 training images  
  - 1000 test images  
- Target class: `Apple Golden 1` (class index 16 in the full dataset)  

The notebook downloads the dataset automatically using `kagglehub`.  
No manual download is required.

Dataset download code used in the notebook:

```python
import kagglehub
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random

root_path = kagglehub.dataset_download("moltean/fruits")
data_root = os.path.join(root_path, "fruits-360_100x100", "fruits-360")
```

---

## 4. How to run the notebook

1. Clone or download this repository.  

   ```bash
   git clone <repo_url>
   cd <repo_folder>
   ```

2. (Optional but recommended) create and activate a virtual environment.  

   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux or macOS
   venv\Scripts\activate         # Windows
   ```

3. Install the required packages.  

   ```bash
   pip install torch torchvision numpy matplotlib opencv-python kagglehub jupyter
   ```

4. Launch Jupyter.  

   ```bash
   jupyter notebook
   ```

5. Open `color_tracker.ipynb` and run all cells in order.  
   The notebook will:
   - download the dataset using `kagglehub`  
   - build training and test subsets  
   - estimate HSV bounds for Apple Golden 1  
   - search for the best mask area threshold  
   - evaluate the tracker and print metrics  
   - generate all figures used in the report  

---

## 5. Reproducing key results

To reproduce the main results reported:

1. Run the cell that creates the training and test subsets:

   - Train subset: 5000 images  
   - Test subset: 1000 images  
   - Target counts: 18 in train, 4 in test  

2. Run the HSV range estimation cell.  
   You should obtain bounds similar to:

   - lower HSV: `[18, 40, 132]`  
   - upper HSV: `[25, 176, 255]`  

3. Run the mask area ratio computation and threshold search.  
   The best threshold on the training subset should be near `0.435` with:

   - training accuracy around `0.993`  
   - training recall around `0.833`  
   - training F1 around `0.462`  

4. Run the test evaluation cell.  
   The confusion matrix on 1000 test images should match:

   - TN = 980, FP = 16, FN = 0, TP = 4  
   - Accuracy ≈ 0.984  
   - Precision ≈ 0.200  
   - Recall = 1.000  
   - F1 ≈ 0.333  

5. Run the plotting cells to generate the figures:

   - Histogram of mask area ratios for target and non target images  
   - Confusion matrix heatmap  
   - Successful detection examples (Apple Golden 1 and their masks)  
   - False positive examples (other yellow fruits and their masks)  


---

## 6. Code overview

The most important functions defined in the notebook are:

- `tensor_to_bgr(tensor_img)`  
  Converts a PyTorch tensor to an OpenCV BGR image.

- `estimate_hsv_range_for_target_tight(...)`  
  Samples HSV pixels from target images and computes percentile based lower and upper bounds.

- `color_track_mask(img_bgr, lower_hsv, upper_hsv)`  
  Produces a binary mask using `cv2.inRange` and morphological operations.

- `mask_area_ratio(img_bgr, lower_hsv, upper_hsv)`  
  Computes the ratio of mask pixels to total pixels.

- `evaluate_threshold(thresh, pos_ratios, neg_ratios)`  
  Computes accuracy, precision, recall, and F1 for a candidate threshold.

- `is_target_image_with_thresh(img_bgr, lower_hsv, upper_hsv, thresh)`  
  Applies the decision rule for a single image.

These functions mirror the methods described in the report and make it easy to extend the project.

---

## 7. Notes and extensions

Ideas for further work:

- Test the pipeline with other target classes from Fruits360.  
- Use adaptive thresholds that depend on image brightness.  
- Combine color masks with simple shape features to reduce false positives.  
- Extend from still images to video sequences and integrate a tracking algorithm such as CAMShift.
