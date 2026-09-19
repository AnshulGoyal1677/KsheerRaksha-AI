# Mastitis Early Screening Module (Modular Extension Architecture)

> [!NOTE]
> **Status: Architectural Scaffold / Future Roadmap**
> No validated, labelled mastitis dataset is currently present in this repository. 
> Consistent with veterinary AI governance and hackathon integrity rules, **no mastitis model is claimed to be trained** in this version.

---

## 1. Overview & Long-Term Vision

In the GaitGuard master cattle welfare architecture, herd health monitoring branches into two complementary visual screening streams:

```
                  RGB Camera Stream / Parlor Walkway
                                  │
                  Cow Detection & Tracking (YOLO / ByteTrack)
                                  │
                  Health-Specific Feature Extraction
                 ┌────────────────┴────────────────┐
                 │                                 │
                 ▼                                 ▼
         [Active in Prototype]            [Planned Extension]
            LAMENESS BRANCH                 MASTITIS BRANCH
                 │                                 │
     Locomotion / Spatiotemporal             Udder Thermal & Visual
      BiGRU Gait Classification               Morphological Analysis
                 │                                 │
                 └────────────────┬────────────────┘
                                  ▼
                     GaitGuard Central Dashboard
```

---

## 2. Intended Future Pipeline

Once a clinically validated dataset of udder imagery (with ground-truth somatic cell count [SCC] or California Mastitis Test [CMT] records) is acquired, the mastitis screening pipeline will operate as follows:

```
Input Video / Image Stream
           │
           ▼
[Step 1] Udder Region Detection & Localization (YOLOv8-Udder)
           │
           ▼
[Step 2] Anatomical Udder Cropping & Aspect-Ratio Normalization
           │
           ▼
[Step 3] Multi-Modal Feature Extraction (Visual Redness/Swelling + Infrared/Thermal Gradients)
           │
           ▼
[Step 4] Mastitis Risk Classifier (ConvNeXt-Tiny / ViT Backbone)
           │
           ▼
[Step 5] Continuous Herd Udder Health Risk Score (Low / Watch / High Risk)
```

---

## 3. Screening Categorization (Non-Diagnostic)

Like the lameness branch, mastitis outputs will be strictly scoped as **early AI decision-support screening** rather than veterinary diagnoses:

| Screening Status | Clinical Indicator | Recommended Herd Management Action |
| :--- | :--- | :--- |
| **NORMAL** | Symmetric udder morphology, uniform coloration, normal teat posture | Routine milking hygiene protocol |
| **WATCH** | Minor asymmetry or slight localized erythema | Flag cow for manual California Mastitis Test (CMT) at next milking |
| **CHECK** | Significant swelling, pronounced redness, or heat signature | Immediate veterinary / quarter-milking inspection |

---

## 4. Dataset Requirements for Training

To train and validate the mastitis branch in future iterations, the following data standard is required:
1. High-resolution udder images across diverse cattle breeds (Holstein Friesian, Jersey, Gir, Sahiwal).
2. Paired clinical labels: Somatic Cell Count (SCC > 200,000 cells/mL threshold) or bacteriological culture validation.
3. Multi-angle milking parlor video captures under variable lighting and soil conditions.
