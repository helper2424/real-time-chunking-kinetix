# RTC Visualization Examples

This directory contains example scripts for visualizing Real-Time Chunking (RTC) flow matching behavior.

## visualize_flow_matching.py

This script loads a trained policy checkpoint and runs a **single inference** with tracking enabled to generate flow matching visualizations. Unlike `src/eval_flow.py` which runs multiple evaluations across different configurations, this script focuses on detailed visualization of one inference run.

### Quick Start

```bash
# Basic usage (loads latest checkpoint)
python visualize_flow_matching.py /path/to/run/directory

# Specify a particular level
python visualize_flow_matching.py /path/to/run/directory \
    --level-path worlds/l/catapult.json

# Customize flow matching parameters
python visualize_flow_matching.py /path/to/run/directory \
    --config.num-flow-steps 30 \
    --config.inference-delay 2 \
    --config.execute-horizon 4 \
    --config.max-guidance-weight 10.0
```

### What it does

1. **Loads checkpoint**: Finds and loads a trained policy from the run directory
2. **Initializes environment**: Sets up the Kinetix environment with the specified level
3. **Runs inference once**: Executes `realtime_action` with tracking enabled
4. **Generates plots**: Creates visualizations showing:
   - Flow matching steps (x_t, v_t, correction over time)
   - Error magnitude evolution
   - Comparison grid across multiple action dimensions

### Output

All plots are saved to the specified output directory (default: `rtc_plots/`):
- `rtc_flow_steps_batch0.png` - Flow matching trajectory for a single action dimension
- `rtc_error_batch0.png` - Error magnitude over denoising steps
- `rtc_comparison_grid_batch0.png` - Multi-dimension comparison view
- `rtc_comparison_dims_*.png` - Custom dimension comparison

### Key Parameters

#### Checkpoint Selection
- `--config.step`: Which checkpoint to load (`-1` for latest, `0` for first, etc.)

#### Flow Matching
- `--config.num-flow-steps`: Number of ODE solver steps (higher = slower but more accurate)
- `--config.inference-delay`: How many actions to execute from previous chunk
- `--config.execute-horizon`: How many actions to execute per chunk
- `--config.max-guidance-weight`: Maximum RTC guidance weight (controls correction strength)
- `--config.prefix-attention-schedule`: Attention schedule (`"exp"`, `"linear"`, `"ones"`, `"zeros"`)

#### Visualization
- `--config.batch-idx`: Which batch element to visualize (if batch size > 1)
- `--config.action-dim-indices`: Which action dimensions to show in comparison grid (e.g., `0 1 2`)
- `--config.horizon-idx`: Which horizon timestep to visualize
- `--output-dir`: Where to save the plots

### Example Commands

```bash
# Visualize first 4 action dimensions with more flow steps
python visualize_flow_matching.py runs/my_experiment \
    --config.num-flow-steps 50 \
    --config.action-dim-indices 0 1 2 3

# Compare different guidance schedules
python visualize_flow_matching.py runs/my_experiment \
    --config.prefix-attention-schedule linear \
    --output-dir plots/linear_schedule

python visualize_flow_matching.py runs/my_experiment \
    --config.prefix-attention-schedule exp \
    --output-dir plots/exp_schedule

# Visualize different levels
python visualize_flow_matching.py runs/my_experiment \
    --level-path worlds/l/grasp_easy.json \
    --output-dir plots/grasp_easy

python visualize_flow_matching.py runs/my_experiment \
    --level-path worlds/l/catapult.json \
    --output-dir plots/catapult
```

### Differences from eval_flow.py

| Feature | visualize_flow_matching.py | src/eval_flow.py |
|---------|---------------------------|------------------|
| Purpose | Detailed visualization of one inference | Quantitative evaluation across multiple runs |
| Inference runs | **1 single run** | Multiple runs with different configs |
| Tracking | ✅ Enabled (with `return_tracking=True`) | ❌ Disabled (performance focus) |
| Output | Visualization plots | CSV with metrics, videos |
| Speed | Fast (one inference) | Slower (many inferences) |
| Use case | Understanding flow matching behavior | Comparing methods and hyperparameters |

### Requirements

Same dependencies as the main project, plus:
- `matplotlib` for plotting
- `numpy` for array operations

### Troubleshooting

**Error: "No such file or directory"**
- Make sure the `run_path` points to a valid training run directory
- The directory should contain numbered subdirectories (e.g., `0/`, `1/`, etc.) with `policies/` folders

**Error: "level not found"**
- Check that the level path is correct relative to the current working directory
- Default levels are in `worlds/l/` directory

**Plots look strange or empty**
- Try increasing `--config.num-flow-steps` for smoother trajectories
- Adjust `--config.max-guidance-weight` if corrections are too large/small
- Use different `--config.action-dim-indices` to visualize other dimensions

For more details, see `../docs/RTC_TRACKING.md`
