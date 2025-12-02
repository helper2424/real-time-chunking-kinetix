"""Visualize RTC tracking with real trained model and environment data.

This script loads a trained policy checkpoint and runs inference WITH and WITHOUT
RTC to compare their behavior side by side.
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
    plot_rtc_comparison,
    plot_rtc_comparison_grid,
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
    output_dir: str = "rtc_comparison_plots",
):
    """Run inference with and without RTC, then generate comparison visualizations.

    Args:
        run_path: Path to training run directory containing checkpoints
        level_path: Path to level JSON file to use for inference
        config: Configuration for inference and visualization
        seed: Random seed
        output_dir: Directory to save visualization plots
    """
    print("=" * 80)
    print("RTC vs No-RTC Comparison Visualization")
    print("=" * 80)
    print(f"\nLoading checkpoint from: {run_path}")
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
    policy_rng, reset_rng, prev_rng, rtc_rng, no_rtc_rng = jax.random.split(rng, 5)

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

    # STEP 1: Generate initial action chunk (this will be the "previous" chunk for RTC)
    print("\n" + "=" * 80)
    print("STEP 1: Generating previous action chunk")
    print("=" * 80)
    prev_action_chunk = policy.action(prev_rng, obs, config.num_flow_steps)
    print(f"Previous action chunk shape: {prev_action_chunk.shape}")

    # Calculate prefix attention horizon
    prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon

    # STEP 2: Run inference WITH RTC (realtime_action with tracking)
    print("\n" + "=" * 80)
    print("STEP 2: Running inference WITH RTC")
    print("=" * 80)
    print(f"  num_flow_steps: {config.num_flow_steps}")
    print(f"  inference_delay: {config.inference_delay}")
    print(f"  execute_horizon: {config.execute_horizon}")
    print(f"  prefix_attention_horizon: {prefix_attention_horizon}")
    print(f"  prefix_attention_schedule: {config.prefix_attention_schedule}")
    print(f"  max_guidance_weight: {config.max_guidance_weight}")

    rtc_action, rtc_tracked = policy.realtime_action(
        rng=rtc_rng,
        obs=obs,
        num_steps=config.num_flow_steps,
        prev_action_chunk=prev_action_chunk,
        inference_delay=config.inference_delay,
        prefix_attention_horizon=prefix_attention_horizon,
        prefix_attention_schedule=config.prefix_attention_schedule,
        max_guidance_weight=config.max_guidance_weight,
        return_tracking=True,
    )

    print(f"RTC action shape: {rtc_action.shape}")
    print(f"RTC tracked keys: {rtc_tracked.keys()}")
    print(f"RTC steps tracked: {rtc_tracked['x_t'].shape[0]}")

    # STEP 3: Run inference WITHOUT RTC (regular action with tracking)
    print("\n" + "=" * 80)
    print("STEP 3: Running inference WITHOUT RTC")
    print("=" * 80)
    print(f"  num_flow_steps: {config.num_flow_steps}")

    no_rtc_action, no_rtc_tracked = policy.action(
        rng=no_rtc_rng,
        obs=obs,
        num_steps=config.num_flow_steps,
        return_tracking=True,
    )

    print(f"No-RTC action shape: {no_rtc_action.shape}")
    print(f"No-RTC tracked keys: {no_rtc_tracked.keys()}")
    print(f"No-RTC steps tracked: {no_rtc_tracked['x_t'].shape[0]}")

    # Convert to numpy for visualization
    rtc_tracked = jax.device_get(rtc_tracked)
    no_rtc_tracked = jax.device_get(no_rtc_tracked)

    # STEP 4: Generate comparison visualizations
    print("\n" + "=" * 80)
    print("STEP 4: Generating comparison visualizations")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Batch index: {config.batch_idx}")

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Plot 1: Side-by-side comparison for single dimension
    print("\nGenerating RTC vs No-RTC comparison plot...")
    fig1 = plot_rtc_comparison(
        rtc_tracked=rtc_tracked,
        no_rtc_tracked=no_rtc_tracked,
        batch_idx=config.batch_idx,
        action_dim_idx=config.action_dim_indices[0] if config.action_dim_indices else 0,
        horizon_idx=config.horizon_idx,
    )
    path1 = pathlib.Path(output_dir) / f"rtc_comparison_batch{config.batch_idx}.png"
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path1.name}")

    # Plot 2: Multi-dimension grid comparison
    print("\nGenerating multi-dimension grid comparison...")
    fig2 = plot_rtc_comparison_grid(
        rtc_tracked=rtc_tracked,
        no_rtc_tracked=no_rtc_tracked,
        batch_idx=config.batch_idx,
        action_dim_indices=list(config.action_dim_indices),
        horizon_idx=config.horizon_idx,
    )
    path2 = pathlib.Path(output_dir) / f"rtc_comparison_grid_batch{config.batch_idx}.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path2.name}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Calculate difference in final actions
    action_diff = jnp.linalg.norm(rtc_action - no_rtc_action)
    print(f"Final action difference (L2 norm): {action_diff:.4f}")

    # Calculate average correction magnitude
    if "correction" in rtc_tracked:
        avg_correction = jnp.mean(jnp.abs(rtc_tracked["correction"]))
        print(f"Average RTC correction magnitude: {avg_correction:.4f}")

    print(f"\nAll visualizations saved to: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    tyro.cli(main)
