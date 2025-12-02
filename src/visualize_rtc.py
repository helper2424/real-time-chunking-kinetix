"""Visualization utilities for Real-Time Chunking flow matching steps."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_flow_matching_steps(
    tracked_steps: dict,
    batch_idx: int = 0,
    action_dim_idx: int = 0,
    horizon_idx: int = 0,
    figsize: tuple[int, int] = (15, 10),
) -> Figure:
    """Plot x_t, v_t, and correction for each flow matching step.

    Args:
        tracked_steps: Dictionary containing tracked values from realtime_action
        batch_idx: Which batch element to visualize
        action_dim_idx: Which action dimension to visualize
        horizon_idx: Which horizon step to visualize
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object
    """
    # Extract data for the specified indices
    x_t = np.array(tracked_steps["x_t"][:, batch_idx, horizon_idx, action_dim_idx])
    v_t = np.array(tracked_steps["v_t"][:, batch_idx, horizon_idx, action_dim_idx])
    correction = np.array(tracked_steps["correction"][:, batch_idx, horizon_idx, action_dim_idx])
    error = np.array(tracked_steps["error"][:, batch_idx, horizon_idx, action_dim_idx])
    time_steps = np.array(tracked_steps["time"])

    num_steps = len(x_t)
    step_indices = np.arange(num_steps)

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle(
        f"Flow Matching Steps (batch={batch_idx}, dim={action_dim_idx}, horizon={horizon_idx})",
        fontsize=14,
        fontweight="bold",
    )

    # Plot x_t (state trajectory)
    axes[0].plot(step_indices, x_t, marker="o", linewidth=2, markersize=4, color="#2E86AB")
    axes[0].set_ylabel("x_t (state)", fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("State Trajectory", fontsize=10)

    # Plot v_t (velocity)
    axes[1].plot(step_indices, v_t, marker="s", linewidth=2, markersize=4, color="#A23B72")
    axes[1].set_ylabel("v_t (velocity)", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Velocity Field", fontsize=10)

    # Plot correction
    axes[2].plot(step_indices, correction, marker="^", linewidth=2, markersize=4, color="#F18F01")
    axes[2].set_ylabel("correction", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Step", fontsize=11, fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title("RTC Correction", fontsize=10)

    plt.tight_layout()
    return fig


def plot_error_over_steps(
    tracked_steps: dict,
    batch_idx: int = 0,
    figsize: tuple[int, int] = (12, 6),
) -> Figure:
    """Plot error magnitude over flow matching steps for all action dimensions.

    Args:
        tracked_steps: Dictionary containing tracked values from realtime_action
        batch_idx: Which batch element to visualize
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object
    """
    error = np.array(tracked_steps["error"][:, batch_idx])  # [num_steps, horizon, action_dim]
    num_steps, horizon, action_dim = error.shape
    step_indices = np.arange(num_steps)

    # Calculate error magnitude across action dimensions
    error_magnitude = np.linalg.norm(error, axis=(1, 2))  # [num_steps]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(step_indices, error_magnitude, marker="o", linewidth=2, markersize=5, color="#C73E1D")
    ax.set_xlabel("Step", fontsize=11, fontweight="bold")
    ax.set_ylabel("Error Magnitude", fontsize=11, fontweight="bold")
    ax.set_title(f"Error Magnitude Over Flow Matching Steps (batch={batch_idx})", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_comparison_grid(
    tracked_steps: dict,
    batch_idx: int = 0,
    action_dim_indices: list[int] | None = None,
    horizon_idx: int = 0,
    figsize: tuple[int, int] = (18, 12),
) -> Figure:
    """Plot comparison grid showing x_t, v_t, and correction for multiple action dimensions.

    Args:
        tracked_steps: Dictionary containing tracked values from realtime_action
        batch_idx: Which batch element to visualize
        action_dim_indices: List of action dimensions to plot (if None, plots first 3)
        horizon_idx: Which horizon step to visualize
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object
    """
    x_t = np.array(tracked_steps["x_t"][:, batch_idx, horizon_idx])  # [num_steps, action_dim]
    v_t = np.array(tracked_steps["v_t"][:, batch_idx, horizon_idx])
    correction = np.array(tracked_steps["correction"][:, batch_idx, horizon_idx])

    num_steps, action_dim = x_t.shape
    step_indices = np.arange(num_steps)

    if action_dim_indices is None:
        action_dim_indices = list(range(min(3, action_dim)))

    num_dims = len(action_dim_indices)
    fig, axes = plt.subplots(num_dims, 3, figsize=figsize, sharex=True)

    if num_dims == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f"Flow Matching Comparison (batch={batch_idx}, horizon={horizon_idx})",
        fontsize=14,
        fontweight="bold",
    )

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#6A994E", "#BC4749"]

    for i, dim_idx in enumerate(action_dim_indices):
        color = colors[i % len(colors)]

        # Plot x_t
        axes[i, 0].plot(step_indices, x_t[:, dim_idx], marker="o", linewidth=2, markersize=3, color=color)
        axes[i, 0].set_ylabel(f"Dim {dim_idx}", fontsize=10, fontweight="bold")
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].set_title("x_t (state)", fontsize=11, fontweight="bold")

        # Plot v_t
        axes[i, 1].plot(step_indices, v_t[:, dim_idx], marker="s", linewidth=2, markersize=3, color=color)
        axes[i, 1].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 1].set_title("v_t (velocity)", fontsize=11, fontweight="bold")

        # Plot correction
        axes[i, 2].plot(step_indices, correction[:, dim_idx], marker="^", linewidth=2, markersize=3, color=color)
        axes[i, 2].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 2].set_title("correction", fontsize=11, fontweight="bold")

        if i == num_dims - 1:
            axes[i, 0].set_xlabel("Step", fontsize=10)
            axes[i, 1].set_xlabel("Step", fontsize=10)
            axes[i, 2].set_xlabel("Step", fontsize=10)

    plt.tight_layout()
    return fig


def save_tracking_plots(
    tracked_steps: dict,
    output_dir: str,
    batch_idx: int = 0,
    prefix: str = "rtc",
) -> list[str]:
    """Generate and save all tracking plots to files.

    Args:
        tracked_steps: Dictionary containing tracked values from realtime_action
        output_dir: Directory to save plots
        batch_idx: Which batch element to visualize
        prefix: Prefix for output filenames

    Returns:
        List of saved file paths
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    # Plot flow matching steps for first action dimension
    fig1 = plot_flow_matching_steps(tracked_steps, batch_idx=batch_idx)
    path1 = os.path.join(output_dir, f"{prefix}_flow_steps_batch{batch_idx}.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    saved_files.append(path1)
    plt.close(fig1)

    # Plot error magnitude
    fig2 = plot_error_over_steps(tracked_steps, batch_idx=batch_idx)
    path2 = os.path.join(output_dir, f"{prefix}_error_batch{batch_idx}.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    saved_files.append(path2)
    plt.close(fig2)

    # Plot comparison grid
    fig3 = plot_comparison_grid(tracked_steps, batch_idx=batch_idx)
    path3 = os.path.join(output_dir, f"{prefix}_comparison_grid_batch{batch_idx}.png")
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    saved_files.append(path3)
    plt.close(fig3)

    return saved_files


def plot_rtc_comparison(
    rtc_tracked: dict,
    no_rtc_tracked: dict,
    batch_idx: int = 0,
    action_dim_idx: int = 0,
    horizon_idx: int = 0,
    figsize: tuple[int, int] = (18, 10),
) -> Figure:
    """Compare RTC vs non-RTC flow matching side by side.

    Args:
        rtc_tracked: Tracked steps from realtime_action with RTC
        no_rtc_tracked: Tracked steps from regular action (no RTC)
        batch_idx: Which batch element to visualize
        action_dim_idx: Which action dimension to visualize
        horizon_idx: Which horizon step to visualize
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object
    """
    # Extract RTC data
    rtc_x_t = np.array(rtc_tracked["x_t"][:, batch_idx, horizon_idx, action_dim_idx])
    rtc_v_t = np.array(rtc_tracked["v_t"][:, batch_idx, horizon_idx, action_dim_idx])
    rtc_correction = np.array(rtc_tracked["correction"][:, batch_idx, horizon_idx, action_dim_idx])

    # Extract non-RTC data
    no_rtc_x_t = np.array(no_rtc_tracked["x_t"][:, batch_idx, horizon_idx, action_dim_idx])
    no_rtc_v_t = np.array(no_rtc_tracked["v_t"][:, batch_idx, horizon_idx, action_dim_idx])

    num_steps = len(rtc_x_t)
    step_indices = np.arange(num_steps)

    # Create figure with subplots (3 rows, 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True)
    fig.suptitle(
        f"RTC vs No-RTC Comparison (batch={batch_idx}, dim={action_dim_idx}, horizon={horizon_idx})",
        fontsize=16,
        fontweight="bold",
    )

    # Column titles
    axes[0, 0].text(0.5, 1.15, "Without RTC", transform=axes[0, 0].transAxes,
                    ha='center', fontsize=13, fontweight='bold', color='#555')
    axes[0, 1].text(0.5, 1.15, "With RTC", transform=axes[0, 1].transAxes,
                    ha='center', fontsize=13, fontweight='bold', color='#555')

    # Row 1: x_t comparison
    axes[0, 0].plot(step_indices, no_rtc_x_t, marker="o", linewidth=2, markersize=4, color="#2E86AB", label="No RTC")
    axes[0, 0].set_ylabel("x_t (state)", fontsize=11, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title("State Trajectory", fontsize=10)

    axes[0, 1].plot(step_indices, rtc_x_t, marker="o", linewidth=2, markersize=4, color="#2E86AB", label="RTC")
    axes[0, 1].set_ylabel("x_t (state)", fontsize=11, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title("State Trajectory", fontsize=10)

    # Row 2: v_t comparison
    axes[1, 0].plot(step_indices, no_rtc_v_t, marker="s", linewidth=2, markersize=4, color="#A23B72")
    axes[1, 0].set_ylabel("v_t (velocity)", fontsize=11, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title("Velocity Field", fontsize=10)

    axes[1, 1].plot(step_indices, rtc_v_t, marker="s", linewidth=2, markersize=4, color="#A23B72")
    axes[1, 1].set_ylabel("v_t (velocity)", fontsize=11, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title("Velocity Field", fontsize=10)

    # Row 3: Correction (only for RTC) and overlay comparison
    axes[2, 0].plot(step_indices, no_rtc_x_t, marker="o", linewidth=2, markersize=3,
                   color="#2E86AB", alpha=0.7, label="No RTC")
    axes[2, 0].plot(step_indices, rtc_x_t, marker="s", linewidth=2, markersize=3,
                   color="#F18F01", alpha=0.7, label="With RTC")
    axes[2, 0].set_ylabel("x_t overlay", fontsize=11, fontweight="bold")
    axes[2, 0].set_xlabel("Step", fontsize=11, fontweight="bold")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_title("Trajectory Comparison", fontsize=10)
    axes[2, 0].legend(loc='best')

    axes[2, 1].plot(step_indices, rtc_correction, marker="^", linewidth=2, markersize=4, color="#F18F01")
    axes[2, 1].set_ylabel("correction", fontsize=11, fontweight="bold")
    axes[2, 1].set_xlabel("Step", fontsize=11, fontweight="bold")
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_title("RTC Correction", fontsize=10)

    plt.tight_layout()
    return fig


def plot_rtc_comparison_grid(
    rtc_tracked: dict,
    no_rtc_tracked: dict,
    batch_idx: int = 0,
    action_dim_indices: list[int] | None = None,
    horizon_idx: int = 0,
    figsize: tuple[int, int] = (20, 14),
) -> Figure:
    """Compare RTC vs non-RTC for multiple action dimensions in a grid.

    Args:
        rtc_tracked: Tracked steps from realtime_action with RTC
        no_rtc_tracked: Tracked steps from regular action (no RTC)
        batch_idx: Which batch element to visualize
        action_dim_indices: List of action dimensions to plot (if None, plots first 3)
        horizon_idx: Which horizon step to visualize
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object
    """
    # Extract data
    rtc_x_t = np.array(rtc_tracked["x_t"][:, batch_idx, horizon_idx])  # [num_steps, action_dim]
    rtc_v_t = np.array(rtc_tracked["v_t"][:, batch_idx, horizon_idx])
    no_rtc_x_t = np.array(no_rtc_tracked["x_t"][:, batch_idx, horizon_idx])
    no_rtc_v_t = np.array(no_rtc_tracked["v_t"][:, batch_idx, horizon_idx])

    num_steps, action_dim = rtc_x_t.shape
    step_indices = np.arange(num_steps)

    if action_dim_indices is None:
        action_dim_indices = list(range(min(3, action_dim)))

    num_dims = len(action_dim_indices)
    fig, axes = plt.subplots(num_dims, 2, figsize=figsize, sharex=True)

    if num_dims == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f"RTC vs No-RTC Grid Comparison (batch={batch_idx}, horizon={horizon_idx})",
        fontsize=16,
        fontweight="bold",
    )

    # Column titles
    axes[0, 0].text(0.5, 1.15, "x_t Trajectory", transform=axes[0, 0].transAxes,
                    ha='center', fontsize=13, fontweight='bold', color='#555')
    axes[0, 1].text(0.5, 1.15, "v_t Velocity", transform=axes[0, 1].transAxes,
                    ha='center', fontsize=13, fontweight='bold', color='#555')

    colors_no_rtc = ["#2E86AB", "#5FA8D3", "#89CFF0"]
    colors_rtc = ["#F18F01", "#FF9E1B", "#FFAD33"]

    for i, dim_idx in enumerate(action_dim_indices):
        color_no = colors_no_rtc[i % len(colors_no_rtc)]
        color_yes = colors_rtc[i % len(colors_rtc)]

        # Plot x_t comparison
        axes[i, 0].plot(step_indices, no_rtc_x_t[:, dim_idx], marker="o", linewidth=2,
                       markersize=3, color=color_no, alpha=0.7, label="No RTC")
        axes[i, 0].plot(step_indices, rtc_x_t[:, dim_idx], marker="s", linewidth=2,
                       markersize=3, color=color_yes, alpha=0.7, label="With RTC")
        axes[i, 0].set_ylabel(f"Dim {dim_idx}", fontsize=10, fontweight="bold")
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].legend(loc='upper right', fontsize=8)

        # Plot v_t comparison
        axes[i, 1].plot(step_indices, no_rtc_v_t[:, dim_idx], marker="o", linewidth=2,
                       markersize=3, color=color_no, alpha=0.7, label="No RTC")
        axes[i, 1].plot(step_indices, rtc_v_t[:, dim_idx], marker="s", linewidth=2,
                       markersize=3, color=color_yes, alpha=0.7, label="With RTC")
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].legend(loc='upper right', fontsize=8)

        if i == num_dims - 1:
            axes[i, 0].set_xlabel("Step", fontsize=10)
            axes[i, 1].set_xlabel("Step", fontsize=10)

    plt.tight_layout()
    return fig
