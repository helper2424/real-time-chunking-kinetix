"""Example script demonstrating how to use RTC tracking and visualization."""

import jax
import jax.numpy as jnp
from flax import nnx

import sys
sys.path.append("../src")

from model import FlowPolicy, ModelConfig
from visualize_rtc import (
    plot_flow_matching_steps,
    plot_error_over_steps,
    plot_comparison_grid,
    save_tracking_plots,
)


def main():
    # Initialize model
    config = ModelConfig(
        channel_dim=256,
        channel_hidden_dim=512,
        token_hidden_dim=64,
        num_layers=4,
        action_chunk_size=8,
    )

    obs_dim = 10
    action_dim = 7
    batch_size = 2

    rng = jax.random.PRNGKey(0)
    model_rng, action_rng = jax.random.split(rng)

    # Create model
    model = FlowPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config,
        rngs=nnx.Rngs(model_rng),
    )

    # Create sample observation and previous action chunk
    obs = jax.random.normal(jax.random.PRNGKey(1), (batch_size, obs_dim))
    prev_action_chunk = jax.random.normal(jax.random.PRNGKey(2), (batch_size, config.action_chunk_size, action_dim))

    # Run realtime_action with tracking enabled
    print("Running realtime_action with tracking...")
    action, tracked_steps = model.realtime_action(
        rng=action_rng,
        obs=obs,
        num_steps=20,
        prev_action_chunk=prev_action_chunk,
        inference_delay=2,
        prefix_attention_horizon=5,
        prefix_attention_schedule="exp",
        max_guidance_weight=10.0,
        return_tracking=True,
    )

    print(f"Action shape: {action.shape}")
    print(f"Tracked steps keys: {tracked_steps.keys()}")
    print(f"Number of steps tracked: {tracked_steps['x_t'].shape[0]}")

    # Visualize the tracking data
    print("\nGenerating plots...")

    # Plot 1: Flow matching steps for a single dimension
    fig1 = plot_flow_matching_steps(
        tracked_steps,
        batch_idx=0,
        action_dim_idx=0,
        horizon_idx=0,
    )
    fig1.savefig("flow_matching_steps.png", dpi=150, bbox_inches="tight")
    print("Saved: flow_matching_steps.png")

    # Plot 2: Error magnitude over steps
    fig2 = plot_error_over_steps(tracked_steps, batch_idx=0)
    fig2.savefig("error_magnitude.png", dpi=150, bbox_inches="tight")
    print("Saved: error_magnitude.png")

    # Plot 3: Comparison grid for multiple dimensions
    fig3 = plot_comparison_grid(
        tracked_steps,
        batch_idx=0,
        action_dim_indices=[0, 1, 2],
        horizon_idx=0,
    )
    fig3.savefig("comparison_grid.png", dpi=150, bbox_inches="tight")
    print("Saved: comparison_grid.png")

    # Or save all plots at once
    print("\nSaving all plots to output directory...")
    saved_files = save_tracking_plots(
        tracked_steps,
        output_dir="./rtc_plots",
        batch_idx=0,
        prefix="rtc",
    )
    print(f"Saved {len(saved_files)} plots:")
    for file in saved_files:
        print(f"  - {file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
