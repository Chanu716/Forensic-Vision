# Deep Learning-Based Forgery Identification and Localization in Videos

## Paper Details

- **Title:** Deep learning-based forgery identification and localization in videos
- **Authors:** Raghavendra Gowda, Digambar Pawar
- **Journal:** Signal, Image and Video Processing
- **Year:** 2023
- **DOI:** https://doi.org/10.1007/s11760-022-02433-7

## Summary

This paper proposes a system for detecting **inter-frame video forgery**, specifically **frame insertion** and **frame deletion**. The method combines three main ideas:

1. **Absolute difference preprocessing** between consecutive frames to expose temporal inconsistencies.
2. A **3D Convolutional Neural Network (3DCNN)** to classify a video clip as authentic, insertion forgery, or deletion forgery.
3. **MS-SSIM (Multi-Scale Structural Similarity)** to localize where the tampering occurred in the video.

The authors report that this approach performs better than several conventional and deep-learning-based baselines, especially under different compression settings and post-processing operations.

## Section-by-Section Explanation

## 1. Introduction

The paper explains that digital video tampering has become easy because of modern editing tools, which makes video authentication important in forensic settings.

The authors distinguish between:

- **Intra-frame forgery:** changes within a frame
- **Inter-frame forgery:** changes across time between frames

This work focuses on **inter-frame forgery**, specifically:

- frame insertion
- frame deletion

The authors argue that many previous approaches rely on hand-crafted features, which are often sensitive to compression, blur, noise, and lighting changes.

## 2. Related Work

Previous methods are grouped into:

- **Feature-engineering-based methods**
- **Deep-learning-based methods**

Feature-based methods often depend on motion vectors, optical flow, GOP structure, or brightness features. The paper argues these methods can be computationally expensive and less robust for long or heavily processed videos.

Deep-learning approaches improved performance, but prior work had limitations such as:

- focusing only on duplication forgery
- using 2D CNNs that do not model temporal information well
- limited generalization across datasets

## 3. Challenges and Contributions

The main challenge is extracting robust features from long, high-quality videos while preserving temporal evidence of tampering.

The paper’s contributions are:

1. A **3DCNN** for learning high-dimensional spatio-temporal features.
2. An **absolute difference algorithm** to reduce temporal redundancy and expose tampering artifacts.
3. **MS-SSIM-based localization** of frame insertion and deletion.
4. Creation of forged inter-frame videos using the **UCF-101** dataset.

## 4. Proposed Methodology

The full pipeline has two stages:

- **Detection**
- **Localization**

### 4.1 Inter-frame Forgery Detection

#### 4.1.1 Video Frame Preprocessing

The authors compute the absolute pixel-wise difference between consecutive frames. This emphasizes temporal changes and suppresses redundant information from nearly identical neighboring frames.

They also split a video into **groups of 49 frames** to make training and inference manageable.

#### 4.1.2 3DCNN Model

The classifier includes:

- Conv3D layers
- ReLU activations
- Batch Normalization
- MaxPooling3D
- Global Average Pooling
- Dense layers
- Softmax output

The reason for using **3D CNNs** is that they capture both **spatial** and **temporal** features directly, which makes them better suited for inter-frame forgery detection than standard 2D CNNs.

### 4.2 Inter-frame Forgery Localization

After classifying a video, the paper localizes the tampered region using **MS-SSIM** between consecutive frames.

#### 4.2.1 MS-SSIM

MS-SSIM measures similarity using:

- luminance
- contrast
- structure

It is used here to detect abrupt frame-to-frame changes caused by insertion or deletion.

#### 4.2.2 Localization Algorithm

The algorithm compares each frame with the next one. If the MS-SSIM score falls below a threshold of **0.8**, that region is flagged as suspicious.

The paper states:

- **Frame insertion** usually causes **two discontinuities**
- **Frame deletion** usually causes **one discontinuity**

## 5. Results and Analysis

### 5.1 Datasets

The authors use:

- **UCF-101** to generate forged videos
- **VIFFD** for evaluation

Forgery clips are created using `ffmpeg` by inserting or deleting **10 to 150 frames**. They also test the method under:

- compression
- Gaussian blur
- Gaussian noise
- brightness variation

### 5.2 Implementation Details

The implementation uses:

- NVIDIA RTX 2080 (8 GB)
- Intel i7 CPU
- 32 GB RAM
- Python
- OpenCV
- ffmpeg
- Keras 2.6.0
- TensorFlow 2.6.0

The paper also shows a GUI prototype for forensic investigators to browse, inspect, and play suspicious videos.

### 5.3 Experimental Analysis

The model is trained on UCF-101 and tested on VIFFD.

Reported test performance on VIFFD:

- **Accuracy:** 0.98
- **Frame Deletion:** Precision 1.00, Recall 0.92, F1 0.96
- **Frame Insertion:** Precision 0.95, Recall 1.00, F1 0.98
- **Authentic:** Precision 1.00, Recall 1.00, F1 1.00

The paper also shows:

- insertion forgery gives **two low-similarity dips**
- deletion forgery gives **one low-similarity dip**

### 5.4 Comparison and Ablation

The proposed model is compared against several conventional and deep-learning methods and is reported to outperform them.

The ablation study is especially important:

- **Without** the difference algorithm: accuracy = **0.8234**
- **With** the difference algorithm: accuracy = **0.9817**

This suggests that the absolute-difference preprocessing step makes a major contribution to performance.

## Key Equations

## 1. Absolute Difference Between Consecutive Frames

\[
P_f(m,n) = |K_f(m,n) - K_{f+1}(m,n)|
\]

This computes the pixel-wise difference between frame `f` and frame `f+1`.

\[
D_k(m,n)=
\begin{cases}
1, & \text{if } P_f(m,n) > f \\
0, & \text{otherwise}
\end{cases}
\]

This thresholding step highlights suspicious motion or inconsistency regions.

## 2. Group-of-Frames Representation

\[
V_1 = \sum_{p=1}^{P} G_p
\]

This expresses a video as a collection of grouped frame segments used for processing.

## 3. SSIM Formula

\[
SSIM(x,y) = [L(x,y)]^\alpha [C(x,y)]^\beta [S(x,y)]^\gamma
\]

Where:

- `L(x, y)` = luminance similarity
- `C(x, y)` = contrast similarity
- `S(x, y)` = structural similarity

This is the basis for measuring similarity between adjacent frames during localization.

## 4. Evaluation Metrics

\[
Accuracy = \frac{TP + TN}{TP + FP + FN + TN}
\]

\[
Precision = \frac{TP}{TP + FP}
\]

\[
Recall = \frac{TP}{TP + FN}
\]

\[
F1 = \frac{2 \cdot precision \cdot recall}{precision + recall}
\]

These are used to evaluate classification quality.

## Figures

### Figure 1

Shows the complete pipeline for forgery detection and localization.

### Figure 2

Shows the architecture of the proposed 3DCNN model.

### Figure 3

Shows sample UCF-101 videos for:

- authentic
- insertion forgery
- deletion forgery

### Figure 4

Shows a GUI-based sample forensic application for browsing and inspecting classified videos.

### Figure 5

Shows training versus testing accuracy over epochs.

### Figure 6

Shows training versus testing loss over epochs.

### Figure 7

Shows localization of **frame insertion forgery** using MS-SSIM. The key signature is **two falling peaks**.

### Figure 8

Shows localization of **frame deletion forgery** using MS-SSIM. The key signature is **one falling peak**.

## Conclusions

The paper concludes that:

- A **3DCNN** is effective for detecting inter-frame video forgery.
- The **absolute difference algorithm** significantly improves detection by reducing temporal redundancy.
- **MS-SSIM** is useful for localizing where insertion or deletion occurred.
- The method performs well even when the original source video is unavailable.

## Limitations and Future Work

The paper only addresses:

- frame insertion
- frame deletion

Future work proposed by the authors includes:

- frame duplication detection
- frame shuffling detection
- creation of a standard inter-frame forgery dataset for future research

## Final Takeaway

The main practical contribution of the paper is not just the use of a 3DCNN, but the combination of:

- frame differencing for temporal artifact exposure
- 3DCNN for spatio-temporal classification
- MS-SSIM for localization

Among these, the ablation study suggests that the **absolute difference preprocessing step** is a major reason for the model’s strong performance.
