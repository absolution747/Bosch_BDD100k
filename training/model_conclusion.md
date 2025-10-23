# YOLOv8m-P2 for BDD100K Dataset

## 📦 Overview

The BDD100K dataset presents a diverse, real-world driving scenario with:

- **10 classes** (cars, people, riders, lights, signs, etc.)
- **Highly imbalanced class distribution** (cars dominate; trains are rare)
- **Multi-scale object sizes** — from tiny traffic lights to large buses/trucks
- **High contextual co-occurrence** (cars, lights, and signs appear together)
- **Dense scenes** with frequent partial occlusion

To handle these challenges, the YOLOv8m architecture was adapted and fine-tuned to improve small-object detection and robustness to context diversity.

---

## 🏗️ Model Architecture (yolov8m_P2.yaml)

### 🔹 Base Architecture

The model is based on **YOLOv8-Medium (YOLOv8m)** — a balanced trade-off between accuracy and computational cost.

| Component | Role | Notes |
|-----------|------|-------|
| Depth multiple = 0.67 | Controls model depth | Same as YOLOv8m baseline |
| Width multiple = 0.75 | Controls channel width | Retains medium model capacity |
| Max channels = 768 | Limits top feature map width | Efficient for GPU memory (T4) |

This configuration ensures efficient training on a **T4 GPU (16 GB)** at **1280×1280 resolution** without out-of-memory issues. However the training for demostration purpose was done on **c4d-highcpu-8** provided by the ***GCP free tier VM***

### 🔹 Added P2 Detection Head

To improve detection of tiny objects (e.g., traffic lights, signs, distant persons), a **P2 detection head** was added.

| Feature Map | Stride | Target Objects | Source Layers |
|-------------|--------|----------------|---------------|
| P2 | 4 | Small objects (e.g., traffic light, traffic sign) | C2f(128) |
| P3 | 8 | Medium objects (car, person, rider) | C2f(256) |
| P4 | 16 | Large objects (bus, truck) | C2f(512) |
| P5 | 32 | Very large objects (train, bus close-up) | C2f(768) |

This modification allows the model to preserve fine spatial detail early in the feature pyramid while maintaining strong semantic features in higher layers.

**✅ Expected Benefit:**
- Better detection of small objects (AP_small ↑)
- Improved recall for distant or partially occluded instances
- Enhanced context-awareness between co-located classes

### 🔹 Backbone & Neck Design

| Component | Description |
|-----------|-------------|
| Backbone (CSPDarknet-like) | Feature extraction using C2f modules for gradient flow and efficiency |
| SPPF (Spatial Pyramid Pooling – Fast) | Captures multi-scale context before detection |
| Neck (PAN-like path) | Top-down + bottom-up aggregation for strong feature reuse |
| Detection Layer (Detect[P2, P3, P4, P5]) | Multi-scale predictions for balanced object sensitivity |

---

## ⚙️ Training Hyperparameters (hyp.yaml)

### 🔹 Core Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| epochs | 50 | Sufficient for fine-tuning from pretrained weights |
| batch | 8 | Fits T4 GPU memory for 1280px input |
| imgsz | 1280 | Larger input size improves small object recognition |
| optimizer | SGD | Stable convergence for detection; better generalization than AdamW in this setup |
| lr0 | 0.01 | Standard starting LR for SGD fine-tuning |
| lrf | 0.01 | Decays LR to 1% of initial value (smooth cosine schedule) |
| momentum | 0.937 | Helps maintain velocity across noisy gradients |
| weight_decay | 0.0005 | Prevents overfitting |
| warmup_epochs | 3 | Gradual learning rate ramp-up for stability |

**✅ Reasoning:**  
The LR schedule (lr0=0.01, lrf=0.01) is tuned for transfer learning — fast adaptation early, smooth convergence later. The 50-epoch window balances efficiency and overfitting prevention. 

**Caveat:** 
However during the actual training the model only trained for 10 epochs for the intrest of time practiccality 

### 🔹 Loss Weighting

| Loss Type | Value | Purpose |
|-----------|-------|---------|
| box | 7.5 | Prioritize accurate localization |
| cls | 1.0 | Balanced classification confidence |
| dfl | 1.5 | Smooths bounding box regression for variable shapes |

**✅ Why:**  
The dataset contains objects of diverse aspect ratios (e.g., tall lights, wide trains). A higher DFL weight (1.5) helps the model learn uncertainty in box edges, improving tolerance to varying shapes.

---

## 🧩 Data Augmentations (hyp.yaml + BDD100k_sample.yaml)

| Augmentation | Value | Reasoning |
|--------------|-------|-----------|
| hsv_h | 0.015 | Slight hue variation to simulate weather/time shifts |
| hsv_s | 0.7 | Strong saturation variation for illumination diversity |
| hsv_v | 0.4 | Brightness variation for night/day scenes |
| degrees | 0.0 | Rotation disabled (rotating vertical lights/signs degrades realism) |
| translate | 0.1 | Moderate positional jitter for robustness |
| scale | 0.5 | Scale variation (zoom 0.5×–1.5×) for multi-distance objects |
| shear | 0.0 | Disabled to preserve geometric fidelity |
| perspective | 0.0 | Disabled to prevent unrealistic distortions in structured scenes |
| flipud | 0.0 | Vertical flips unrealistic for road scenes |
| fliplr | 0.5 | Horizontal flips realistic and effective |
| mosaic | 1.0 | Critical for small object visibility and class balance |
| mixup | 0.0 | Disabled — distorts small features |
| close_mosaic | 10 | Disables mosaic during last 10 epochs to stabilize convergence |

**✅ Overall Strategy:**
- Focused on realistic color & scale variation
- Avoided geometric transformations that harm structured traffic layouts
- Mosaic augmentation is the key to compensating for class imbalance and tiny object scarcity

---

## 📊 Dataset Configuration (BDD100k_sample.yaml)

| Field | Description |
|-------|-------------|
| path | /workspace/data/bdd100k_subset_yolo |
| train/val splits | 800 train / 200 val |
| Classes | 10 classes matching YOLOv8 architecture |
| Input size | Optimized for 1280×1280 |
| Augmentations | Mirror those in hyp.yaml for consistency |

**✅ Ensures class name and order consistency across architecture and training definitions.**

---

## 🎯 Summary of Design Philosophy

| Design Aspect | Choice | Rationale |
|---------------|--------|-----------|
| Architecture | YOLOv8m + P2 | Multi-scale balance; enhances small-object recall |
| Resolution | 1280×1280 | Captures fine detail for small objects |
| Loss Weights | box=7.5, cls=1.0, dfl=1.5 | Prioritize localization and shape tolerance |
| Augmentations | Realistic, structure-preserving | Retain context and geometric integrity |
| Optimizer | SGD, lr decay (0.01→0.0001) | Stable and generalizable fine-tuning |
| Training Schedule | 50 epochs on T4 GPU 10 on CPU, warmup 3 | Efficient convergence with pretrained weights |
| Evaluation Goals | High mAP@small and balanced per-class recall | Dataset-sensitive optimization |