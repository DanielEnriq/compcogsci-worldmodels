# CCS World Models Project  
**What Does a World Model Represent?**  
Computational Cognitive Science — NYU Spring 2026  

---

## Overview

This project investigates a core question in computational cognitive science:

> **What properties must internal representations have to support flexible, goal-directed planning?**

Modern world models differ not just in performance, but in their **architectural inductive biases** — which implicitly define *what a “state” is*.

We evaluate whether these biases produce representations aligned with cognitive theories of planning, specifically:

- **Invariance to task-irrelevant features**
- **Latent geometry reflecting causal structure (not perception)**
- **Generalization across goals**

---

## Models Compared

We evaluate three world models using a unified interface:

| Model | Inductive Bias | Representation Pressure |
|------|------|------------------------|
| **LeWorldModel (LeWM)** | Minimal predictive (JEPA) | Predict next latent only |
| **DINO-WM** | Pretrained semantic vision | Inherits DINOv2 features |
| **PLDM** | Latent dynamics model | Planning-oriented latent |

⚠️ Note: DreamerV3 is **not used** — it is not available in the stable-worldmodel framework.

---

## Environments

We use environments from the **stable-worldmodel benchmark**:

- **PushT-v1**
  - Task: Push T-shaped object to target
  - Task-relevant: position, orientation
  - Task-irrelevant: background color, texture

- (Optional) **PointMaze-v1**
  - Task: Navigate to goal
  - Task-relevant: agent position
  - Task-irrelevant: walls, texture

These environments allow **controlled perturbations (FoVs)** for Experiment 1.

---

## Experiments

### Experiment 1 — Distractor Sensitivity
Does the latent representation change when only *task-irrelevant features* change?

Metric:
- Latent distance between states with identical task state but different FoVs

---

### Experiment 2 — Geometric Causal Fidelity (CPAR)
Does latent distance reflect:

- causal distance (planning steps), or
- perceptual similarity?

Metric:
- **CPAR = corr(latent, causal) / corr(latent, pixel)**

---

### Experiment 3 — Goal Generalization
Do representations support planning under distribution shift?

Metric:
- success rate under:
  - near goals
  - moderate shift
  - far (OOD) goals

---

## Repository Structure

```
ccs-world-models/
│
├── src/
│   ├── utils.py              # shared encoding + distance functions
│   ├── load_model.py         # model loading logic
│   └── encode.py             # encoder wrappers
│
├── experiments/
│   ├── exp1_distractor.py
│   ├── exp2_geometry.py
│   └── exp3_goals.py
│
├── notebooks/
│   └── sanity_check.ipynb    # Person A setup + verification
│
├── results/
│   ├── raw/
│   └── figures/
│
├── data/                     # stored on Google Drive
├── checkpoints/              # stored on Google Drive
│
├── requirements.txt
└── README.md
```

---

## Setup Instructions (Person A)

All GPU work is done in **Google Colab**.

### 1. Clone repo

```bash
git clone https://github.com/YOUR_USERNAME/ccs-world-models.git
cd ccs-world-models
```

---

### 2. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### 3. Install dependencies

```bash
pip install stable-worldmodel
pip install torch torchvision
pip install matplotlib seaborn scipy
```

---

### 4. Set paths

```python
DRIVE_ROOT = "/content/drive/MyDrive/ccs_project"

DATA_PATH = f"{DRIVE_ROOT}/data"
CKPT_PATH = f"{DRIVE_ROOT}/checkpoints"
RESULTS_PATH = f"{DRIVE_ROOT}/results"
```

---

### 5. Download data + checkpoints

You must manually place:

```
/drive/.../checkpoints/
    lewm/
    dinowm/
    pldm/
```

---

### 6. Verify environment

```python
from stable_worldmodel.envs import make_env

env = make_env("PushT-v1")
obs = env.reset()
```

---

### 7. Verify FoVs

```python
obs1 = env.reset(options={"background_color": "red"})
obs2 = env.reset(options={"background_color": "blue"})
```

You should see **visual differences with identical task state**.

---

### 8. Load models

```python
from src.load_model import load_model

model, encoder = load_model("lewm", CKPT_PATH)
z = encoder(obs)
```

---

### 9. CRITICAL: DINO-WM pooling

DINO returns patch tokens — you MUST do:

```python
z = z.mean(axis=0)
```

If you skip this → results are invalid.

---

### 10. Sanity check

Test:

- same state + different FoV → small latent distance
- different state → large distance

Save:

```
results/sanity_check_latents.npz
```

---

## Shared API (utils.py)

All team members use:

```python
encode_obs(encoder, obs, model_name)
latent_dist(z1, z2)
encode_batch(...)
```

Person A owns this file.

---

## Roles

### Person A (You — DL Lead)
- Setup Colab + repo
- Load all models
- Implement utils.py
- Run Experiment 1

### Person B
- Experiment 2 (geometry / CPAR)

### Person C
- Experiment 3 (goal generalization)

---

## Deliverables

By end of setup:

- [ ] All 3 models load
- [ ] FoVs verified
- [ ] Latent shapes confirmed
- [ ] utils.py complete
- [ ] sanity_check.ipynb runs cleanly

---

## Core Hypothesis

Architectural inductive bias determines whether a model learns:

> **causal, planning-relevant representations**  
> vs  
> **perceptual or semantic representations**

If correct:

- LeWM → best alignment with causal structure  
- DINO-WM → distracted by visual features  
- PLDM → intermediate  

---

## Notes

- This project evaluates **internal representations**, not just performance
- No human participants required
- Fully computational, reproducible setup

---

## References

See paper draft for full citations:

- Craik (1943)
- Tolman (1948)
- Givan et al. (2003)
- Ferns et al. (2004)
- Hafner et al. (DreamerV3)
- Maes et al. (LeWM)
- Zhou et al. (DINO-WM)
- Terver et al. (stable-worldmodel)