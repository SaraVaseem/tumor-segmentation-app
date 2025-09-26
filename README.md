# Brain Tumor Segmentation with Mask R-CNN

## Overview

Brain tumor segmentation is a critical task in radiology, used to monitor tumor growth and support treatment decisions. Traditionally, this process is performed manually by radiologists, which is both time-consuming and requires a high level of precision.

This project automates segmentation by combining a **deep learning model** with a **full-stack web application**. My team and I trained a **Mask R-CNN** model using the **PixelLib** library on MRI scans from the **Br35H** and **Figshare** datasets. We then deployed the model in a Flask backend with a Next.js frontend, allowing users to **drag and drop an MRI scan (PNG)** and instantly receive a segmented tumor overlay.

I documented this project in more detail on my [LinkedIn post](https://www.linkedin.com/posts/sara-vaseem-347a3b22a_i-am-super-excited-to-announce-that-my-team-activity-7316556072729657345-hoAn?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAADl2VycBxHI6rMge8tDtqmURZAokD971lOY).

---

## Motivation

* Manual segmentation is labor-intensive and subject to human error.
* Automated segmentation can reduce workload, accelerate decision-making, and improve treatment consistency.
* A web-based interface makes the tool accessible to non-technical users, bridging research and clinical usability.

---

## Methodology

### Machine Learning

* **Model**: Mask R-CNN with a ResNet-101 backbone.
* **Framework**: [PixelLib](https://github.com/ayoolaolafenwa/PixelLib)
* **Datasets**:

  * [Br35H](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)
  * [Figshare Brain Tumor Dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
* **Training Process**:

  * Converted LabelMe JSON annotations to binary masks for pixel-level evaluation.
  * Two-phase training (heads → all layers).
  * Fine-tuned confidence thresholds to balance precision/recall.
* **Metrics (min\_confidence = 0.90)**:

  * IoU: 68%
  * Precision: 77%
  * Recall: 82%

### Web Application

* **Backend**: Flask

  * Hosts the trained Mask R-CNN model.
  * Handles image uploads and runs inference.
  * Returns segmentation masks as JSON + processed PNGs.

* **Frontend**: Next.js

  * Drag-and-drop upload for MRI scans.
  * Displays segmented tumor overlay directly in the browser.
  * Provides a simple, user-friendly interface for testing and demonstrations.

---

## Results

* End-to-end pipeline: MRI upload → backend inference → segmented overlay in browser.
* Prioritized **recall** over precision to minimize false negatives in medical use cases.
* Logged JSON metrics for reproducibility and dataset split tracking.
* Awarded **First Place** at a research symposium at *California State University, Northridge*.

---

## Challenges

1. **Mask Compatibility**

   * LabelMe polygons vs. Mask R-CNN masks required conversion to binary masks for IoU/precision/recall.

2. **Environment Drift**

   * Running PixelLib with CUDA demanded pinned versions of TensorFlow, CUDA, and dependencies to ensure stable training.

3. **Model Constraints**

   * PixelLib’s abstraction limited access to non-maximum suppression (NMS). Instead, we parameterized minimum confidence to study trade-offs.

4. **Deployment**

   * Packaging a GPU-trained model into a Flask backend while ensuring smooth integration with Next.js was non-trivial.
   * Solved by containerizing dependencies and documenting version requirements.

---

## Impact

This project bridges **AI research and real-world usability** by combining a state-of-the-art segmentation model with an accessible web interface. By collaborating with radiologists, we demonstrated the potential for AI-assisted radiology tools that are both technically sound and practical for clinical workflows.

---

## Installation & Usage

### Backend (Flask)

```bash
# Clone repository
git clone https://github.com/SaraVaseem/tumor-segmentation-app.git
cd tumor-segmentation-app/server

# Install dependencies
pip install -r requirements.txt

# Run Flask server
python app.py
```

### Frontend (Next.js)

```bash
cd ../src/app

# Install dependencies
npm install

# Run Next.js app
npm run dev
```

### Usage

* Open `http://localhost:3000` in your browser.
* Drag and drop an MRI scan (PNG).
* Segmentation results appear directly in the browser.

---

## References

* He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2018). Mask R-CNN. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.
* [PixelLib](https://github.com/ayoolaolafenwa/PixelLib)
* [Br35H Dataset](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)
* [Figshare Brain Tumor Dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)

---
