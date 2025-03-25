from ..configs.base_config import BaseConfig
from dataclasses import dataclass


@dataclass
class PhysicsEngineConfig(BaseConfig):
    """
    Physics Engine configuration. Contains the physical constants of the simulation, such as gravity, time step, etc.

    Attributes:
        num_robots (int): number of robots in the simulation.
        dt (float): time step for numerical integration. Default is 0.01. (100 Hz)
        gravity (float): acceleration due to gravity. Default is 9.81 m/s^2.
        torque_limit (float): torque limit that can be generated on CoG. The physics engine clips it to this value. Default is 500.0 Nm.
        damping_alpha (float): damping coefficient modifier, should be between larger than 1.0. Applies damping alpha times higher
        than critical damping.
        soft_contact_sigma (float): soft contact sigma. Default is 0.5. This is the width of the soft contact force function.
    """

    num_robots: int
    dt: float = 0.01
    gravity: float = 9.81
    torque_limit: float = 200.0
    damping_alpha: float = 1.0
    soft_contact_sigma: float = 0.01

    def __post_init__(self):
        if self.num_robots <= 0:
            raise ValueError("num_robots must be greater than 0")
        if self.dt <= 0:
            raise ValueError("dt must be greater than 0")
        if self.gravity <= 0:
            raise ValueError("gravity must be greater than 0")
        if self.torque_limit <= 0:
            raise ValueError("torque_limit must be greater than 0")
        if self.damping_alpha < 1.0:
            raise ValueError("damping_alpha must be greater than or equal to 1.0")
