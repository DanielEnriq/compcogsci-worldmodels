## Experiment 2B — Controlled Nuisance vs State Separation

### Original Issue

The first version of Experiment 2 attempted to build triplets of the form:

- **Anchor:** baseline image/state
- **Nuisance:** same physical state, different visual appearance
- **State-change:** different physical state, visually similar appearance

The goal was to test whether model representations separate visual nuisance variation from true physical state variation.

However, the original implementation was unreliable because the SWM variation API was being used incorrectly. In particular:

- Some visual perturbations produced no pixel changes.
- Some attempted visual changes silently failed.
- Some perturbations risked changing physical state, breaking the nuisance assumption.
- Some nuisance pairs were effectively identical to anchor images.
- The initial red-background perturbation was too visually extreme, producing a pixel difference much larger than the intended state-change contrast.

As a result, the original Experiment 2 result was not interpretable.

### Redesign

We redesigned the experiment using direct, controlled SWM variation values.

The key API fix was to use:

```python
world.reset(seed=seed, options={"variation_values": {
    "background.color": np.array([252, 252, 255], dtype=np.uint8),
    "block.start_position": pos
}})

MERGE CONFLICT