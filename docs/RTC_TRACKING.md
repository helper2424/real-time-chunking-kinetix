# RTC Flow Matching Tracking and Visualization

This document describes the tracking and visualization features for Real-Time Chunking (RTC) flow matching steps.

## Overview

The `realtime_action` method has been upgraded to track all intermediate values during the flow matching process. This allows you to:
- Monitor the evolution of states (`x_t`) across denoising steps
- Analyze velocity fields (`v_t`)
- Visualize RTC corrections applied at each step
- Track prediction errors relative to the previous action chunk

## Usage

### Enabling Tracking

To enable tracking, set `return_tracking=True` when calling `realtime_action`:

```python
from model import FlowPolicy

# ... initialize model ...

action, tracked_steps = model.realtime_action(
    rng=rng,
    obs=obs,
    num_steps=20,
    prev_action_chunk=prev_action_chunk,
    inference_delay=2,
    prefix_attention_horizon=5,
    prefix_attention_schedule="exp",
    max_guidance_weight=10.0,
    return_tracking=True,  # Enable tracking
)
```

### Tracked Data Structure

The `tracked_steps` dictionary contains the following arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `x_t` | `[num_steps, batch, horizon, action_dim]` | State at each denoising step |
| `v_t` | `[num_steps, batch, horizon, action_dim]` | Velocity field (before correction) |
| `correction` | `[num_steps, batch, horizon, action_dim]` | RTC guidance correction applied |
| `error` | `[num_steps, batch, horizon, action_dim]` | Weighted error between prediction and previous chunk |
| `x_1_pred` | `[num_steps, batch, horizon, action_dim]` | Predicted final state at each step |
| `time` | `[num_steps]` | Timestep values (0 to 1) |

## Visualization Functions

The `visualize_rtc.py` module provides several plotting functions:

### 1. Flow Matching Steps Plot

Visualize x_t, v_t, and correction for a single action dimension over all steps:

```python
from visualize_rtc import plot_flow_matching_steps

fig = plot_flow_matching_steps(
    tracked_steps,
    batch_idx=0,
    action_dim_idx=0,
    horizon_idx=0,
)
fig.savefig("flow_steps.png")
```

### 2. Error Magnitude Plot

Plot the error magnitude across all action dimensions:

```python
from visualize_rtc import plot_error_over_steps

fig = plot_error_over_steps(tracked_steps, batch_idx=0)
fig.savefig("error_magnitude.png")
```

### 3. Comparison Grid

Compare x_t, v_t, and correction across multiple action dimensions:

```python
from visualize_rtc import plot_comparison_grid

fig = plot_comparison_grid(
    tracked_steps,
    batch_idx=0,
    action_dim_indices=[0, 1, 2],
    horizon_idx=0,
)
fig.savefig("comparison_grid.png")
```

### 4. Save All Plots

Generate and save all visualization plots at once:

```python
from visualize_rtc import save_tracking_plots

saved_files = save_tracking_plots(
    tracked_steps,
    output_dir="./rtc_plots",
    batch_idx=0,
    prefix="rtc",
)
print(f"Saved: {saved_files}")
```

## Example

See `examples/visualize_flow_matching.py` for a complete working example that loads a trained checkpoint:

```bash
# Basic usage with default parameters
python examples/visualize_flow_matching.py <run_path>

# With specific level and parameters
python examples/visualize_flow_matching.py \
    <run_path> \
    --level-path worlds/l/grasp_easy.json \
    --config.num-flow-steps 20 \
    --config.inference-delay 3 \
    --config.execute-horizon 6 \
    --output-dir rtc_plots

# Visualize different action dimensions
python examples/visualize_flow_matching.py \
    <run_path> \
    --config.action-dim-indices 0 1 2 3
```

This will:
1. Load a trained policy checkpoint from the specified run directory
2. Reset the environment with the specified level
3. Run a single `realtime_action` inference with tracking enabled
4. Generate visualization plots showing x_t, v_t, and correction over flow matching steps
5. Save all plots to the specified output directory

### Command-line Arguments

- `run_path` (required): Path to training run directory containing checkpoint subdirectories
- `--level-path`: Path to level JSON file (default: `worlds/l/grasp_easy.json`)
- `--config.step`: Which checkpoint step to load (default: `-1` for latest)
- `--config.num-flow-steps`: Number of flow matching steps (default: `20`)
- `--config.inference-delay`: Inference delay parameter (default: `3`)
- `--config.execute-horizon`: Execute horizon parameter (default: `6`)
- `--config.max-guidance-weight`: Maximum RTC guidance weight (default: `5.0`)
- `--config.prefix-attention-schedule`: Prefix attention schedule: `"exp"`, `"linear"`, `"ones"`, or `"zeros"` (default: `"exp"`)
- `--config.action-dim-indices`: Which action dimensions to visualize in comparison grid (default: `0 1 2`)
- `--output-dir`: Directory to save plots (default: `rtc_plots`)
- `--seed`: Random seed (default: `0`)

## Interpreting the Plots

### State Trajectory (x_t)
- Shows how the noisy initial state evolves toward the final action
- Should generally show convergence from random noise to structured action

### Velocity Field (v_t)
- The uncorrected velocity predicted by the flow model
- Indicates the direction the model wants to move in action space

### RTC Correction
- The adjustment applied by Real-Time Chunking guidance
- Helps maintain temporal consistency with the previous action chunk
- Larger corrections indicate stronger guidance to match the prefix

### Error Magnitude
- Measures how far the current prediction deviates from the previous chunk
- Should generally decrease as guidance corrections are applied
- High error indicates mismatch between current and previous predictions

## Performance Considerations

Tracking adds computational overhead:
- Additional memory to store intermediate arrays
- Slightly slower execution due to dictionary construction

For production inference, use `return_tracking=False` (default) to disable tracking.

## Comparison with LeRobot Implementation

This implementation is inspired by [LeRobot PR #1698](https://github.com/huggingface/lerobot/pull/1698) and provides:
- Similar tracking capabilities for RTC debugging
- Comprehensive visualization utilities
- Time-based organization of tracked data
- Support for batched inference visualization

## Dependencies

The visualization module requires:
- `matplotlib` for plotting
- `numpy` for array operations
- `jax` for array conversion

Install with:
```bash
pip install matplotlib numpy jax
```
