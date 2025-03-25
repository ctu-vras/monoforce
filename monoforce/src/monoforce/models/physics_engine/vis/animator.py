import time
from typing import Iterable
from ..engine.engine_state import PhysicsState, AuxEngineInfo, vectorize_iter_of_states
from ..configs import WorldConfig, PhysicsEngineConfig
import mayavi.mlab as mlab


def animate_trajectory(
    world_config: WorldConfig,
    engine_config: PhysicsEngineConfig,
    states: Iterable[PhysicsState],
    aux_info: Iterable[AuxEngineInfo],
    robot_index: int = 0,
    **vis_opts,
):
    """
    Visualize the trajectory of a robot in a physics simulation.

    Besides the arguments, the function accepts the following keyword arguments:
    - window_size: tuple, default (1280, 720)
    - fps: float, default None
    - show_terrain: bool, default True
    - show_spring_forces: bool, default True
    - show_contacts: bool, default True
    - show_non_contacts: bool, default True
    - show_thrust_vectors: bool, default True
    - show_friction_forces: bool, default True

    Args:
        world_config (WorldConfig): World configuration.
        engine_config (PhysicsEngineConfig): Physics engine configuration.
        states (Iterable[PhysicsState]): Simulation states.
        aux_info (Iterable[AuxEngineInfo]): Auxiliary information for simulation states.
        robot_index (int, optional): Index of the robot to visualize. Defaults to 0.
        **vis_opts: Additional visualization options.

    Returns:
        None
    """
    # vectorize states
    states_vec = vectorize_iter_of_states(states)
    aux_info_vec = vectorize_iter_of_states(aux_info)

    # basic settings
    window_size = vis_opts.get("window_size", (1280, 720))
    freq = vis_opts.get("fps", None)
    show_terrain = vis_opts.get("show_terrain", True)
    show_spring_forces = vis_opts.get("show_spring_forces", True)
    show_contacts = vis_opts.get("show_contacts", True)
    show_non_contacts = vis_opts.get("show_non_contacts", True)
    show_thrust_vectors = vis_opts.get("show_thrust_vectors", True)
    show_friction_forces = vis_opts.get("show_friction_forces", True)
    show_act_forces = vis_opts.get("show_act_forces", True)

    # camera config
    cam_cfg = vis_opts.get("camera", None)

    # figure
    f = mlab.figure(size=window_size)

    # static terrain
    if show_terrain:
        mlab.mesh(
            world_config.x_grid[robot_index],
            world_config.y_grid[robot_index],
            world_config.z_grid[robot_index],
            colormap="terrain",
            opacity=0.8,
        )

    # unpack
    xs = states_vec.x[:, robot_index].cpu().numpy()
    robot_points = aux_info_vec.global_robot_points[:, robot_index].cpu().numpy()
    fspring = aux_info_vec.F_spring[:, robot_index].cpu().numpy()
    ffriction = aux_info_vec.F_friction[:, robot_index].cpu().numpy()
    contact_masks = aux_info_vec.in_contact[:, robot_index].squeeze(-1).cpu().numpy() > 1e-3
    thrust_vectors = aux_info_vec.global_thrust_vectors[:, robot_index].cpu().numpy()
    act_forces = aux_info_vec.act_forces[:, robot_index].cpu().numpy()

    # trajectory
    traj_vis = mlab.plot3d(xs[:, 0], xs[:, 1], xs[:, 2], color=(0, 1, 0), line_width=0.2)

    # contact points, etc.
    non_contact_point_vis = mlab.points3d(robot_points[0, :, 0], robot_points[0, :, 1], robot_points[0, :, 2], scale_factor=0.03, color=(0, 0, 0))
    non_contact_point_vis.visible = show_non_contacts

    contact_point_vis = mlab.points3d(robot_points[0, :, 0], robot_points[0, :, 1], robot_points[0, :, 2], scale_factor=0.03, color=(1, 0, 0))
    contact_point_vis.visible = show_contacts

    spring_forces_vis = mlab.quiver3d(
        robot_points[0, :, 0],
        robot_points[0, :, 1],
        robot_points[0, :, 2],
        fspring[0, :, 0],
        fspring[0, :, 1],
        fspring[0, :, 2],
        line_width=3.0,
        scale_factor=0.1,
        color=(1, 0, 0),
    )
    spring_forces_vis.visible = show_spring_forces

    thrust_vectors_vis = mlab.quiver3d(
        robot_points[0, :, 0],
        robot_points[0, :, 1],
        robot_points[0, :, 2],
        thrust_vectors[0, :, 0],
        thrust_vectors[0, :, 1],
        thrust_vectors[0, :, 2],
        line_width=3.0,
        scale_factor=0.1,
        color=(0, 0, 1),
    )
    thrust_vectors_vis.visible = show_thrust_vectors

    friction_forces_vis = mlab.quiver3d(
        robot_points[0, :, 0],
        robot_points[0, :, 1],
        robot_points[0, :, 2],
        ffriction[0, :, 0],
        ffriction[0, :, 1],
        ffriction[0, :, 2],
        line_width=3.0,
        scale_factor=0.1,
        color=(0, 1, 0),
    )

    friction_forces_vis.visible = show_friction_forces

    act_forces_vis = mlab.quiver3d(
        robot_points[0, :, 0],
        robot_points[0, :, 1],
        robot_points[0, :, 2],
        act_forces[0, :, 0],
        act_forces[0, :, 1],
        act_forces[0, :, 2],
        line_width=3.0,
        scale_factor=0.1,
        color=(1, 0.5, 0),
    )
    act_forces_vis.visible = show_act_forces

    # optionally set camera
    if cam_cfg is not None:
        mlab.view(
            azimuth=cam_cfg.get("azimuth", 0),
            elevation=cam_cfg.get("elevation", 90),
            distance=cam_cfg.get("distance", 10),
            focalpoint=cam_cfg.get("focalpoint", (0, 0, 0)),
        )

    # animate
    for i in range(xs.shape[0]):
        # separate contact from non-contact
        robot_point = robot_points[i]
        fsp = fspring[i]
        tvec = thrust_vectors[i]
        ffr = ffriction[i]
        contact_mask = contact_masks[i]
        act_fs = act_forces[i]
        non_contact_points = robot_point.copy()
        contact_points = robot_point.copy()
        non_contact_points[contact_mask] = 0
        contact_points[~contact_mask] = 0
        fsp[~contact_mask] = 0

        # update
        if show_non_contacts:
            non_contact_point_vis.mlab_source.set(x=non_contact_points[:, 0], y=non_contact_points[:, 1], z=non_contact_points[:, 2])
        if show_contacts:
            contact_point_vis.mlab_source.set(x=contact_points[:, 0], y=contact_points[:, 1], z=contact_points[:, 2])
        if show_spring_forces:
            spring_forces_vis.mlab_source.set(
                x=contact_points[:, 0],
                y=contact_points[:, 1],
                z=contact_points[:, 2],
                u=fsp[:, 0],
                v=fsp[:, 1],
                w=fsp[:, 2],
            )
        if show_thrust_vectors:
            thrust_vectors_vis.mlab_source.set(
                x=robot_point[:, 0], y=robot_point[:, 1], z=robot_point[:, 2], u=tvec[:, 0], v=tvec[:, 1], w=tvec[:, 2]
            )

        if show_friction_forces:
            friction_forces_vis.mlab_source.set(x=robot_point[:, 0], y=robot_point[:, 1], z=robot_point[:, 2], u=ffr[:, 0], v=ffr[:, 1], w=ffr[:, 2])

        if show_act_forces:
            act_forces_vis.mlab_source.set(
                x=robot_point[:, 0], y=robot_point[:, 1], z=robot_point[:, 2], u=act_fs[:, 0], v=act_fs[:, 1], w=act_fs[:, 2]
            )

        traj_vis.mlab_source.reset(x=xs[: i + 1, 0], y=xs[: i + 1, 1], z=xs[: i + 1, 2])

        mlab.process_ui_events()
        if freq is not None:
            if freq == "realtime":
                time.sleep(engine_config.dt)
            else:
                time.sleep(1 / freq)

    mlab.show()
