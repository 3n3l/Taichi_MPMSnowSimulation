import taichi as ti
import numpy as np


@ti.data_oriented
class Configuration:
    """This class represents a starting configuration for the MLS-MPM algorithm."""

    def __init__(
        self,
        velocity: np.ndarray,
        position: np.ndarray,
        name: str,
        nu=0.2,  # Poisson's ratio (0.2)
        E=1.4e5,  # Young's modulus (1.4e5)
        zeta=10,  # Hardening coefficient (10)
        stickiness=1,  # Higher value means a stickier border
        friction=1,  # Higher value means the border has more friction
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=7.5e-3,  # Critical stretch (7.5e-3)
    ):
        n = position.shape[0]
        m = velocity.shape[0]
        assert n == m, "Positions and velocities shape not matching!"

        # Parameters starting points for MPM
        self.group_size = position.shape[0]
        self.velocity = velocity
        self.position = position
        self.E = E
        self.nu = nu
        self.name = name
        self.zeta = zeta
        self.theta_c = theta_c
        self.theta_s = theta_s
        self.friction = friction
        self.stickiness = stickiness
        self.mu_0 = self.E / (2 * (1 + self.nu))
        self.lambda_0 = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
