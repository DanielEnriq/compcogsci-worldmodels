# Project Status — World Model Representation Analysis
**Last updated: 2026-05-02 | Author: Sihan (Person B)**

> This file is the shared ground truth for all three teammates. LLMs: read this for full context before generating code.

---

## Team Roles
| Person | Responsibility |
|--------|---------------|
| Danny  | Data collection, LeWM/DINO-WM/MAE latent extraction, Phases 1–11 |
| Sihan (Person B) | Experiment 2 CPAR (Phase 12), Phase 6 nuisance fix |
| Person C | TBD |

---

## Notebook Structure (`CCS_final_notebook.ipynb`)

| Phase | Name | Status |
|-------|------|--------|
| 1 | Data collection (random rollouts, PushT) | ✅ Working |
| 2 | LeWM latent extraction → (N,192) | ✅ Working |
| 3 | DINO-WM → (N,768), MAE → (N,768) | ✅ Working |
| 4 | Probing: linear + MLP probe on state | ✅ Strong results |
| 5 | **Nuisance invariance** (anchor/nuisance/state-change triplets) | ⚠️ Fixed API, design needs refinement |
| 6 | Latent geometry (pairwise distance correlations) | ✅ Working, strong results |
| 7 | Temporal behavior (step-to-step latent deltas) | ⚠️ Partial — descriptive stats OK, interpretation unclear |
| 8 | VoE 5A: encoder-level surprise | ✅ Working |
| 9 | VoE 5B: predictor-level surprise (LeWM only) | ✅ Working, important result |
| 10 | VoE Strong: intervention-based surprise tracking | ⚠️ Visual perturbations unreliable |
| 11 | (Danny's last phase) | 🔲 Needs variation API fix |
| **12** | **CPAR / Geometric Causal Fidelity (Sihan)** | ✅ Done — see results below |

---

## Key Findings

### Phase 4 — Probing
- Position (agent xy, block xy, block angle) predicted well by all models
- Velocity (last 2 dims of 7D state) **not recoverable** from static frames — expected

### Phase 6 — Latent Geometry
- LeWM: strong correlation with block position (~0.72)
- DINO/MAE: weaker correlations overall
- Random baseline: no structure

### Phase 12 — CPAR (Geometric Causal Fidelity)
**Formula:** `CPAR = Spearman(D_latent, D_causal) / Spearman(D_latent, D_pixel)`
- D_pixel: pairwise L2 on flattened normalized pixels (224×224×3), mean=34.665
- D_causal: pairwise L2 on normalized block state (cols 0,1,2: block_x, block_y, block_angle), mean=2.224
- N=800 observations, 319,600 upper-triangle pairs

| Model   | r_causal | r_pixel | CPAR  |
|---------|----------|---------|-------|
| LeWM    | 0.365    | 0.608   | **0.600** |
| DINO-WM | 0.374    | 0.250   | **1.495** |
| MAE     | 0.308    | 0.256   | **1.200** |

**Key insight:** All models encode similar causal info (~0.31–0.37). The differentiator is r_pixel. LeWM's latent geometry correlates with raw pixels 2× more than DINO/MAE — it preserves appearance because its training objective requires visual prediction. DINO and MAE abstract away appearance.

**Confound:** Random rollout data. LeWM's causal structure may only appear along action-conditioned temporal trajectories, not across static reset frames. CPAR on structured goal-directed rollouts is pending.

### Phase 5 (Fixed) — Nuisance Invariance
- After API fix: pixel_diff=153.73, state_diff=0.000 (verified clean perturbations)
- DINO-WM: nuisance dist=22.253, state-change dist=11.484, ratio=0.516
- LeWM: nuisance dist=7.019, state-change dist=0.920, ratio=0.131
- Both < 1 (inverted). **Reason:** red background is too extreme — dominates any encoder. State-change from random seed is too subtle.
- Fix needed: subtler nuisances (hue shift) OR more dramatic state changes (extreme block positions)

---

## SWM API — Critical Facts (Do Not Deviate)

```python
world = swm.World("swm/PushT-v1", num_envs=1, image_shape=(224, 224))
world.reset(seed=seed)                        # returns None — NOT (obs, info)
obs   = world.infos['pixels'][0, 0]           # (224, 224, 3)
state = world.states['state'][0]              # (7,)

# Variation (background color)
world.reset(seed=seed, options={"variation_values": {
    "background.color": np.array([255, 0, 0], dtype=np.uint8)   # shape (3,) NOT (N,3)
}})
```

**Wrong patterns (cause silent failures):**
- `options={"variation": [], "variation_values": v}` — wrong key structure
- `np.array([[255,0,0]])` with shape (1,3) — wrong shape

---

## LeWM Encoder — Critical Facts

```python
# Input: (B, C, H, W) — NO time dimension at encoder level
batch = torch.tensor(obs_array[i:i+16], dtype=torch.float32).permute(0,3,1,2) / 255.0
out = encoder(batch.to(DEVICE))
z = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out
if z.dim() == 3: z = z[:, 0, :]    # take [CLS] token
# Output: (B, 192)
```

**Wrong patterns:**
- `batch.unsqueeze(1)` — adds time dim, causes `ValueError: too many values to unpack`
- `out.dim()` directly — HuggingFace returns `BaseModelOutputWithPooling`, not tensor

---

## Drive Layout (`MyDrive/stablewm_data/` = STABLEWM_HOME)

```
stablewm_data/
    pusht/
        lewm_object.ckpt
        dinowm_object.ckpt
        pldm_object.ckpt
    results/exp2_geometry/    ← Person B saves here
    figures/
```

**Not** `stablewm_project/` — that's Danny's old path in his notebook.

---

## Priority List (Danny's orders, 2026-05-01)

1. **[URGENT] Fix Phase 5/6 nuisance experiment** — design more subtle perturbations
2. **[URGENT] Fix Phase 11 VoE** — apply corrected variation API
3. Clean Experiment 5 VoE metrics
4. Add structured goal-directed rollouts dataset
5. Improve temporal analysis (Experiment 4)

---

## What Is NOT Done Yet
- [ ] Redesign Phase 5 nuisance with subtle perturbations
- [ ] Fix Phase 11 VoE variation API calls
- [ ] CPAR re-run on structured rollouts (pending Danny's data)
- [ ] Cross-model predictor comparison (only LeWM uses dynamics currently)
- [ ] Unified narrative / paper structure