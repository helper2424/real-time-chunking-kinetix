"""Visualize RTC tracking with real trained model and environment data.

This script loads a trained policy checkpoint and runs a single inference
with tracking enabled to generate flow matching visualizations.
"""

import dataclasses
import pathlib
import pickle
import sys

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as wrappers
import tyro

sys.path.append("../src")

import model as _model
import train_expert
from visualize_rtc import (
    plot_flow_matching_steps,
    plot_error_over_steps,
    plot_comparison_grid,
    save_tracking_plots,
)


@dataclasses.dataclass(frozen=True)
class VisualizeConfig:
    step: int = -1
    num_flow_steps: int = 20
    inference_delay: int = 3
    execute_horizon: int = 6

    # RTC parameters
    prefix_attention_schedule: _model.PrefixAttentionSchedule = "exp"
    max_guidance_weight: float = 5.0

    # Model config
    model: _model.ModelConfig = _model.ModelConfig()

    # Visualization options
    batch_idx: int = 0
    action_dim_indices: tuple[int, ...] = (0, 1, 2)
    horizon_idx: int = 0


def main(
    run_path: str,
    level_path: str = "worlds/l/grasp_easy.json",
    config: VisualizeConfig = VisualizeConfig(),
    seed: int = 0,
    output_dir: str = "rtc_plots",
):
    """Run single inference with tracking and generate visualizations.

    Args:
        run_path: Path to training run directory containing checkpoints
        level_path: Path to level JSON file to use for inference
        config: Configuration for inference and visualization
        seed: Random seed
        output_dir: Directory to save visualization plots
    """
    print(f"Loading checkpoint from: {run_path}")
    print(f"Using level: {level_path}")

    # Setup environment (same as eval_flow.py)
    static_env_params = kenv_state.StaticEnvParams(
        **train_expert.LARGE_ENV_PARAMS,
        frame_skip=train_expert.FRAME_SKIP,
        screen_dim=train_expert.SCREEN_DIM,
    )
    env_params = kenv_state.EnvParams()

    # Load level
    levels = train_expert.load_levels([level_path], static_env_params, env_params)
    level = jax.tree.map(lambda x: x[0], levels)

    # Create environment
    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)
    env = wrappers.LogWrapper(wrappers.AutoReplayWrapper(train_expert.NoisyActionWrapper(env)))

    # Get observation and action dimensions
    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), level, env_params)[0].shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")

    # Load policy checkpoint
    level_name = level_path.replace("/", "_").replace(".json", "")
    log_dirs = list(filter(lambda p: p.is_dir() and p.name.isdigit(), pathlib.Path(run_path).iterdir()))
    log_dirs = sorted(log_dirs, key=lambda p: int(p.name))

    checkpoint_path = log_dirs[config.step] / "policies" / f"{level_name}.pkl"
    print(f"Loading checkpoint: {checkpoint_path}")

    with checkpoint_path.open("rb") as f:
        state_dict = pickle.load(f)

    # Create policy and load weights
    rng = jax.random.key(seed)
    policy_rng, reset_rng, action_rng = jax.random.split(rng, 3)

    policy = _model.FlowPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config.model,
        rngs=nnx.Rngs(policy_rng),
    )

    graphdef, state = nnx.split(policy)
    state.replace_by_pure_dict(state_dict)
    policy = nnx.merge(graphdef, state)

    print("Policy loaded successfully")

    # Reset environment and get initial observation
    obs, env_state = env.reset_to_level(reset_rng, level, env_params)
    print(f"Initial observation shape: {obs.shape}")

    # Add batch dimension if needed
    if obs.ndim == 1:
        obs = obs[None, :]

    # Generate initial action chunk (this will be the "previous" chunk)
    prev_action_chunk = policy.action(action_rng, obs, config.num_flow_steps)
    print(f"Initial action chunk shape: {prev_action_chunk.shape}")

    # Calculate prefix attention horizon
    prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon

    print(f"\nRunning realtime_action with tracking...")
    print(f"  num_flow_steps: {config.num_flow_steps}")
    print(f"  inference_delay: {config.inference_delay}")
    print(f"  execute_horizon: {config.execute_horizon}")
    print(f"  prefix_attention_horizon: {prefix_attention_horizon}")
    print(f"  prefix_attention_schedule: {config.prefix_attention_schedule}")
    print(f"  max_guidance_weight: {config.max_guidance_weight}")

    # Run realtime_action with tracking enabled
    action, tracked_steps = policy.realtime_action(
        rng=action_rng,
        obs=obs,
        num_steps=config.num_flow_steps,
        prev_action_chunk=prev_action_chunk,
        inference_delay=config.inference_delay,
        prefix_attention_horizon=prefix_attention_horizon,
        prefix_attention_schedule=config.prefix_attention_schedule,
        max_guidance_weight=config.max_guidance_weight,
        return_tracking=True,
    )

    print(f"\nAction shape: {action.shape}")
    print(f"Tracked steps keys: {tracked_steps.keys()}")
    print(f"Number of steps tracked: {tracked_steps['x_t'].shape[0]}")

    # Convert to numpy for visualization
    tracked_steps = jax.device_get(tracked_steps)

    # Generate visualizations
    print(f"\nGenerating visualizations (batch_idx={config.batch_idx})...")

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save all plots
    saved_files = save_tracking_plots(
        tracked_steps,
        output_dir=output_dir,
        batch_idx=config.batch_idx,
        prefix="rtc",
    )

    print(f"\nSaved {len(saved_files)} plots to {output_dir}/:")
    for file in saved_files:
        print(f"  - {pathlib.Path(file).name}")

    # Also generate comparison grid with specified dimensions
    fig = plot_comparison_grid(
        tracked_steps,
        batch_idx=config.batch_idx,
        action_dim_indices=list(config.action_dim_indices),
        horizon_idx=config.horizon_idx,
    )
    comparison_path = pathlib.Path(output_dir) / f"rtc_comparison_dims_{config.action_dim_indices}_batch{config.batch_idx}.png"
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
    print(f"  - {comparison_path.name}")

    print(f"\nDone! All visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    tyro.cli(main)
