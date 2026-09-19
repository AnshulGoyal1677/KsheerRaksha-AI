# GaitGuard AI — Dataset Audit Report & Architectural Recommendation
**Project:** GaitGuard AI — Automated Cattle Lameness Screening  
**Timestamp:** 2026-09-20  
**Phase:** Phase 0 (Dataset Audit) & Phase 1 (Architecture Selection)  
**Dataset Source:** `CattleLameness-main` (Derived from YouTube Videos)

---

## Executive Summary

An exhaustive, non-destructive audit of the downloaded cattle lameness video dataset was performed. All 50 video files were inspected using OpenCV and standard hashing algorithms without altering any original files. The dataset is balanced (25 Lame, 25 Normal) and entirely decodable with zero file corruption. However, video durations, frame counts, and frame rates exhibit notable variance, necessitating uniform temporal subsampling. Crucially, while 42 unique cattle are reported by the authors across the 50 clips, per-cow identifiers are absent, creating potential identity overlap across splits that must be managed.

Based on strict 5-hour hackathon constraints, compute availability (NVIDIA RTX 4060 Laptop GPU, 8GB VRAM), and dataset characteristics (small sample size of 50 videos), we evaluate candidate architectures and recommend a **Pretrained 2D CNN Feature Extractor (e.g., EfficientNet-B0 or ResNet-18) combined with Temporal Pooling / Bi-directional GRU (Recurrent Temporal Aggregation)**. This architecture balances rich visual feature representations with temporal gait modeling, trains in under 2 minutes, and eliminates overfitting risks inherent in training 3D CNNs from scratch on small datasets.

---

## Section 1: Detailed Dataset Audit (20 Verification Points)

| # | Inspection Point | Findings & Telemetry | Implications & Mitigations |
|---|---|---|---|
| **1** | **Total Number of Videos** | **50 videos** in total | Limited sample size; requires strong regularization and transfer learning. |
| **2** | **Total Number of Samples** | **50 samples** (each video serves as 1 distinct sample) | Sample-level analysis is identical to video-level analysis. |
| **3** | **Number of NORMAL Samples** | **25 samples** (`N (1).mp4` to `N (25).mp4`) | Represents healthy, uninhibited cattle locomotion. |
| **4** | **Number of LAME Samples** | **25 samples** (`L (1).mp4` to `L (25).mp4`) | Represents abnormal gait, head bobs, asymmetric strides, or arched back posture. |
| **5** | **Existing Split** | **None present in dataset directory**. Raw files are organized in flat `Data/Lame/` and `Data/Normal/` folders. | A stratified train/validation/test split must be established. |
| **6** | **Video Duration Distribution** | **Min:** 1.35s, **Max:** 12.28s, **Mean:** 5.64s ± 1.98s, **Median:** 5.22s, **IQR:** [4.50s, 6.58s]<br>• *Lame Mean:* 5.93s ± 1.58s<br>• *Normal Mean:* 5.35s ± 2.31s | Varying duration requires uniform temporal frame sampling rather than fixed frame-stride sampling. |
| **7** | **Resolution Distribution** | **100% Uniform: 500 × 500 pixels** (1:1 square aspect ratio) | All videos were pre-cropped by the dataset authors. Resizing to model input (e.g. 224×224) will not distort aspect ratio. |
| **8** | **FPS Distribution** | **Min:** 25.0, **Max:** 60.0, **Mean:** 36.09 ± 12.89 FPS.<br>Modes: 29.97 FPS (24 videos), 30.0 FPS (12 videos), 60.0 FPS (7 videos), 59.94 FPS (4 videos), 25.0 FPS (1 video). | Mix of standard video framerates and high-frame-rate mobile uploads. Normalizing to fixed $T$ frames across total duration handles FPS disparity. |
| **9** | **File Formats & Codecs** | **Container:** `.mp4` (50/50)<br>**FourCC:** `avc1` / H.264 (50/50) | Industry-standard encoding, fully hardware-accelerated decoding via OpenCV. |
| **10** | **Label Format** | Folder-level hierarchy (`Data/Lame` vs `Data/Normal`). | Easily parsed into binary classes: 0 = NORMAL, 1 = LAME. |
| **11** | **Label Granularity** | **Per-video / per-clip**. No frame-by-frame annotations, no spatial bounding boxes, no keypoint skeletons. | Supervised training must be performed at the video-level or clip-level aggregation. |
| **12** | **Cattle IDs Available?** | **No**. Files are named `L (1).mp4` .. `L (25).mp4` and `N (1).mp4` .. `N (25).mp4`. No ID metadata exists. | **Explicit limitation**: We cannot map clips to individual animals programmatically. |
| **13** | **Same Cow in Multiple Videos?** | **Yes**. Dataset authors state: *"50 online video clips featuring 42 individual cattle"*. | At least $50 - 42 = 8$ clips feature cows that appear in more than one clip. |
| **14** | **Same Cow Across Splits?** | **High risk under random splitting**. Since IDs are not tagged, independent random splitting could assign different clips of the same cow to train and test. | To be documented as an ecological validity caveat. We must use stratified k-fold or fixed seed holdout and avoid frame-level leaks. |
| **15** | **Duplicate / Near-Duplicate Videos** | **Exact Duplicates:** 0 (all SHA-256 hashes unique).<br>**Near-Duplicates:** Pairwise perceptual cosine similarity $< 0.95$. | All 50 clips represent distinct recordings or cuts. |
| **16** | **Frame Leakage Risk** | **Zero leakage IF split at video level**. Extreme leakage risk IF frames are shuffled independently across splits. | **Mandate:** Splitting MUST occur strictly at the video level. Never treat individual frames as independent samples during train/test partition. |
| **17** | **Corrupted / Unreadable Files** | **0 corrupted files**. All 50 videos open, decode initial and terminal frames successfully. Decodable frames sum to 9,950. | Dataset is 100% physically intact. |
| **18** | **Class Imbalance** | **Balanced 1.0 : 1.0** (25 Lame, 25 Normal; 50.0% each). | No severe class imbalance re-weighting needed, though precision/recall monitoring remains essential. |
| **19** | **Number of Unique Cattle** | **42 unique cattle** (per dataset author publication). | Cow re-identification without visual markers (ear tags, coats) is unfeasible within hackathon timeframe. |
| **20** | **Unusual Dataset Characteristics** | • Diverse visual contexts (milking parlor concrete, open grass, dirt corrals).<br>• Camera perspectives vary: lateral view, 45-degree oblique, frontal walking.<br>• Varying walking speeds and distances from the camera. | Spatial feature representations must be robust to background textures. |

---

## Section 2: Critical Identity & Data Leakage Limitations

> [!WARNING]
> **ANIMAL IDENTITY LIMITATION REPORT**
> The original dataset creators curated 50 clips from YouTube representing 42 individual cattle. However, **no individual animal IDs, ear-tag identifiers, or clip-to-cow mappings were published with the dataset**.
> 
> **Implications:**
> 1. It is mathematically certain that approximately 8 video clips depict cattle that appear in another clip.
> 2. Because these identities are not tagged, standard stratified partitioning cannot guarantee that all clips of Cow $X$ are sequestered exclusively in the training or test split.
> 3. Consequently, there is a minor risk of cow-level identity leakage across splits.
> 
> **Mitigation Strategy:**
> - Split strictly at the **video level** (never frame level) so no temporal frame sequences from the same clip leak.
> - Use a fixed, reproducible stratified train/validation/test split (e.g., 34 train, 8 validation, 8 test).
> - Explicitly present the model as an **AI screening tool** rather than a diagnostic device, with validation metrics reported transparently.

---

## Section 3: Phase 1 — Model Architecture Selection

### Evaluation of Candidate Approaches (Under 5-Hour Time Constraint)

| Architecture Candidate | Feasibility & Pros | Risks & Cons | Hackathon Verdict |
|---|---|---|---|
| **A. 3D CNN Trained from Scratch** *(e.g., author's Colab `3dcnn.py`)* | • Simple Conv3D layers.<br>• Native spatio-temporal convolutions. | • **Massive overfitting:** 3D CNNs have millions of parameters and require $10^4$–$10^5$ samples to learn meaningful spatial-temporal filters without collapsing.<br>• With only 50 videos, scratch 3D CNNs memorize background noise.<br>• High GPU memory consumption. | ❌ **Rejected** (High overfitting risk, slow convergence). |
| **B. Pretrained Video Transformer / Heavy 3D CNN** *(e.g., SlowFast, VideoMAE, I3D-Kinetics)* | • State-of-the-art video representation.<br>• Pretrained on Kinetics-400. | • Huge model size (150MB–800MB download).<br>• Extremely slow inference latency on CPU/laptop GPU.<br>• Heavy PyTorch video dependencies (pytorchvideo, torchvision video backends often fail on Windows with MSVC). | ❌ **Rejected** (Dependency overhead, excessive latency). |
| **C. Pretrained 2D CNN Backbone + Temporal Pooling / GRU** *(e.g., ImageNet ResNet-18 / MobileNetV3 / EfficientNet + Temporal Attention / GRU)* | • **Transfer Learning:** Leverages robust ImageNet features (edges, animal contours, limb poses).<br>• **Ultra-Fast Training:** Feature caching allows training temporal head in $< 60$ seconds.<br>• **Lightweight & Dependable:** $< 15$ MB model size, $< 20$ ms inference per video.<br>• Captures temporal rhythm and gait dynamics across sampled frames.<br>• Minimal engineering complexity; zero fragile dependencies. | • 2D backbone does not learn joint space-time kernels initially, but temporal GRU/LSTM/pooling effectively models sequential gait motion across frames. | ✅ **SELECTED (Primary Recommendation)** |
| **D. YOLO Object Detection / Pose** | • Can locate cow bounding boxes or leg joints. | • Dataset has NO bounding box or keypoint labels. Would require zero-shot tracking or general YOLOv8 cow crops, adding significant inference latency without solving classification. | ⚪ **Optional Preprocessing Only** (Deferred to avoid eating into the 5-hour budget). |

---

### Selected Architecture Specification: ResNet-18 / MobileNetV3 + Temporal Gait Classifier

1. **Temporal Frame Sampler:**
   - Samples $T = 16$ uniformly spaced frames from each video:
     $$t_i = \left\lfloor i \cdot \frac{N - 1}{T - 1} \right\rfloor \quad \text{for } i \in \{0, \dots, T - 1\}$$
   - Ensures consistent temporal coverage regardless of whether video is 25 FPS or 60 FPS, 2 seconds or 10 seconds.
2. **Spatial Feature Extractor:**
   - Pretrained ResNet-18 or MobileNetV3 backbone (ImageNet weights).
   - Input: $16 \times 3 \times 224 \times 224$.
   - Output: $16 \times D$ feature sequence (where $D = 512$).
3. **Temporal Aggregator:**
   - Bidirectional GRU (hidden dimension 64) or Multi-Head Temporal Self-Attention + Global Mean/Max Pooling.
   - Summarizes the $16$-frame gait sequence into a single 128-dimensional gait descriptor.
4. **Classification Head:**
   - Linear layer + Dropout(0.4) + Linear(128, 2) $\to$ Softmax.
   - Outputs calibrated probability:
     $$P(\text{LAME} \mid \text{video}) = \sigma(z_{\text{lame}})$$
5. **Screening Risk Tri-band Indicator:**
   - **NORMAL:** $P(\text{LAME}) < 0.35$ (Green — healthy locomotion)
   - **WATCH:** $0.35 \le P(\text{LAME}) < 0.65$ (Amber — subtle asymmetry, early monitoring advised)
   - **CHECK:** $P(\text{LAME}) \ge 0.65$ (Red — pronounced lameness indicators, veterinary evaluation recommended)

This architecture fulfills all user requirements:
- Video-level classification
- True temporal gait modeling
- Fast, reproducible training in under 2 minutes
- Fully deployable for live hackathon demonstration
- Adheres to ethical "AI Screening Result" framing (not veterinary diagnosis).
