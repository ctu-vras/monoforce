import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Iterable, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.collections import LineCollection
from ....configs import WorldConfig
from ..engine.engine_state import (
    PhysicsState,
    vectorize_iter_of_states,
    AuxEngineInfo,
)
from ..utils.geometry import quaternion_to_yaw


START_COLOR = "blue"
END_COLOR = "green"


def plot_grids_xyz(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    """
    Plot the x, y, z grids.

    Parameters:
        - x: x-coordinates of the grid.
        - y: y-coordinates of the grid.
        - z: z-coordinates of the grid.
        - title: Title of the plot.
    """
    fig, ax = plt.subplots(1, 3, figsize=(20, 5), dpi=200)
    x_im = ax[0].contourf(x, y, x, cmap="gray", levels=100)
    ax[0].set_title("X")
    y_im = ax[1].contourf(x, y, y, cmap="gray", levels=100)
    ax[1].set_title("Y")
    z_im = ax[2].contourf(x, y, z, cmap="inferno", levels=100)
    ax[2].set_title("Z")
    for axis in ax:
        axis.set_aspect("equal")
    plt.colorbar(x_im, ax=ax[0])
    plt.colorbar(y_im, ax=ax[1])
    plt.colorbar(z_im, ax=ax[2])
    fig.tight_layout()
    plt.show()


def plot_single_heightmap(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    start: torch.Tensor | None = None,
    end: torch.Tensor | None = None,
    **fig_opts,
) -> plt.Axes:
    """
    Plot the heightmap.

    Parameters:
        - x: x-coordinates of the grid.
        - y: y-coordinates of the grid.
        - z: z-coordinates of the grid.
        - start: Starting position of the robot (optional).
        - end: Ending position of the robot (optional).
        - fig_opts: Additional options for the figure.

    Returns:
        - Axes object.
    """
    fig = plt.figure(figsize=(11, 10), dpi=200, **fig_opts)
    gs = fig.add_gridspec(ncols=2, width_ratios=[40, 1])
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])  # colorbar axis
    im = ax.contourf(x, y, z, cmap="inferno", levels=100)
    ax.set_aspect("equal")
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_title("z")  # label
    cb.ax.tick_params(labelsize=10)
    cb.ax.locator_params(nbins=20)
    fig.tight_layout()
    if start is not None:
        ax.plot(start[0], start[1], "o", label="Start", color=START_COLOR)
        ax.text(start[0], start[1], "Start", fontsize=8, color="white")
    if end is not None:
        ax.plot(end[0], end[1], "o", label="End", color=END_COLOR)
        ax.text(end[0], end[1], "End", fontsize=8, color="white")
    return ax


def plot_heightmap_3d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, **kwargs) -> go.Figure:
    """
    Plot the 3D heightmap.

    Parameters:
        - x: x-coordinates of the grid.
        - y: y-coordinates of the grid.
        - z: z-coordinates of the grid.
        - start: Starting position of the robot (optional).
        - end: Ending position of the robot (optional).

    Returns:
        - Plotly figure.
    """
    x = x.detach().cpu()
    y = y.detach().cpu()
    z = z.detach().cpu()
    fig = go.Figure(data=[go.Surface(z=z.cpu().numpy(), x=x.cpu().numpy(), y=y.cpu().numpy())])
    max_z = z.abs().max().item()
    if "start" in kwargs:
        start = kwargs["start"].detach().cpu()
        fig.add_trace(
            go.Scatter3d(
                x=[start[0].item()],
                y=[start[1].item()],
                z=[start[2].item()],
                mode="markers+text",
                marker=dict(size=5, color=START_COLOR),
                textfont=dict(color="gray"),
                text=["Start"],
                textposition="top center",
                showlegend=False,
            )
        )
        max_z = max(max_z, start[2].abs().item())

    if "end" in kwargs:
        end = kwargs["end"].detach().cpu()
        fig.add_trace(
            go.Scatter3d(
                x=[end[0].item()],
                y=[end[1].item()],
                z=[end[2].item()],
                mode="markers+text",
                marker=dict(size=5, color=END_COLOR),
                textfont=dict(color="gray"),
                text=["End"],
                textposition="top center",
                showlegend=False,
            )
        )
        max_z = max(max_z, end[2].abs().item())
    if "robot_points" in kwargs:
        robot_points = kwargs["robot_points"].detach().cpu()
        fig.add_trace(
            go.Scatter3d(
                x=robot_points[:, 0].cpu().numpy(),
                y=robot_points[:, 1].cpu().numpy(),
                z=robot_points[:, 2].cpu().numpy(),
                mode="markers",
                marker=dict(size=2, color="black"),
                showlegend=False,
            )
        )
        max_z = max(max_z, robot_points[:, 2].abs().max().item())
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Height (Z)",
            camera_eye=dict(x=1.0, y=1.0, z=0.5),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=max_z / (2 * x.max().item())),
        ),
        width=1000,
        height=500,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def plot_birdview_trajectory(
    world_config: WorldConfig,
    states: Iterable[PhysicsState],
    robot_idx: int = 0,
    iter_step: int = 30,
    fig_opts: dict[str, Any] = {},
) -> plt.Axes:
    """
    plot the birdview trajectory of the robot.
    """
    # grids
    x_grid_arr = world_config.x_grid[robot_idx].cpu().numpy()
    y_grid_arr = world_config.y_grid[robot_idx].cpu().numpy()
    z_grid_arr = world_config.z_grid[robot_idx].cpu().numpy()

    # vectorize states
    states_vec = vectorize_iter_of_states(states)

    # create figure w/ main axis and a separate colorbar axis
    fig = plt.figure(figsize=(11, 10), dpi=200, **fig_opts)
    gs = fig.add_gridspec(ncols=2, width_ratios=[40, 1])
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])  # colorbar axis

    # surface
    cf = ax.contourf(x_grid_arr, y_grid_arr, z_grid_arr, cmap="gray", levels=20)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Birdview of trajectory")

    # direction arrows
    for i in range(0, len(states_vec.x), iter_step):
        pos = states_vec.x[i, robot_idx].cpu().numpy()
        yaw = quaternion_to_yaw(states_vec.q[i, robot_idx].cpu()).item()
        ax.text(pos[0], pos[1], str(i), fontsize=8, color="white")
        ax.arrow(
            pos[0],
            pos[1],
            0.5 * np.cos(yaw),
            0.5 * np.sin(yaw),
            head_width=0.1,
            head_length=0.1,
            fc="w",
            ec="w",
        )

    # heatmap line
    np_points = states_vec.x[:, robot_idx, :2].unsqueeze(1).cpu().numpy()
    segments = np.concatenate([np_points[:-1], np_points[1:]], axis=1)
    zs = states_vec.x[:, robot_idx, 2].cpu().numpy()
    lc = LineCollection(segments, cmap="viridis", norm=plt.Normalize(zs.min(), zs.max()))
    lc.set_array(zs)
    lc.set_linewidth(3)
    ax.add_collection(lc)

    # aspect ratio
    ax.set_aspect("equal", "box")  # 'box' or 'box-forced' if older mpl

    # colorbar w/ custom axis
    cb = fig.colorbar(lc, cax=cax)
    cb.ax.set_title("z")  # label
    cb.ax.tick_params(labelsize=10)
    cb.ax.locator_params(nbins=20)
    # you can add more ticks by specifying them directly or using locator params
    # e.g. cb.ax.locator_params(nbins=10)
    fig.tight_layout()
    return ax


def plot_3d_trajectory(
    world_config: WorldConfig,
    states: Iterable[PhysicsState],
    auxs: Iterable[AuxEngineInfo],
    robot_idx: int = 0,
    fig_opts: dict[str, Any] = {},
) -> go.Figure:
    """
    Plot the 3D trajectory of the robot interactively.
    """
    # Grids
    x_grid_arr = world_config.x_grid[robot_idx].cpu().numpy()
    y_grid_arr = world_config.y_grid[robot_idx].cpu().numpy()
    z_grid_arr = world_config.z_grid[robot_idx].cpu().numpy()
    # Vectorize states
    states_vec = vectorize_iter_of_states(states)
    aux_vec = vectorize_iter_of_states(auxs)
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "surface"}]], **fig_opts)
    # World
    fig.add_trace(
        go.Surface(x=x_grid_arr, y=y_grid_arr, z=z_grid_arr, colorscale="Cividis", showscale=False),
        row=1,
        col=1,
    )
    # Trajectory
    xs = states_vec.x[:, robot_idx, 0].cpu().numpy()
    ys = states_vec.x[:, robot_idx, 1].cpu().numpy()
    zs = states_vec.x[:, robot_idx, 2].cpu().numpy()
    fig.add_trace(
        go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="lime", width=5), name="Trajectory"),
    )
    # Robot pointcloud in last state with contact points and forces
    pts_global = aux_vec.global_robot_points[-1, robot_idx].cpu().numpy()
    in_contact = aux_vec.in_contact[-1, robot_idx].bool().squeeze(-1).cpu().numpy()
    not_contact = ~in_contact
    # Non-contact points
    fig.add_trace(
        go.Scatter3d(
            x=pts_global[not_contact, 0],
            y=pts_global[not_contact, 1],
            z=pts_global[not_contact, 2],
            mode="markers",
            marker=dict(size=2, color="black"),
            name="Non-contact points",
        ),
    )
    # Contact points
    fig.add_trace(
        go.Scatter3d(
            x=pts_global[in_contact, 0],
            y=pts_global[in_contact, 1],
            z=pts_global[in_contact, 2],
            mode="markers",
            marker=dict(size=2, color="red"),
            name="Contact points",
        ),
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Height (Z)",
            camera_eye=dict(x=1.0, y=1.0, z=0.5),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=max(abs(zs).max().item(), abs(pts_global[..., 2]).max()) / (2 * world_config.max_coord)),
        ),
        width=1000,
        height=500,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig
