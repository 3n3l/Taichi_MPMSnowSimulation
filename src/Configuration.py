import numpy as np


class Configuration:
    """This class represents a starting configuration for the MLS-MPM algorithm."""

    def __init__(
        self,
        velocity: np.ndarray,
        position: np.ndarray,
        n_particles: int,
        quality: int,
        name: str,
        nu=0.2,  # Poisson's ratio (0.2)
        E=1.4e5,  # Young's modulus (1.4e5)
        zeta=10,  # Hardening coefficient (10)
        rho_0=4e2,  # Initial density (4e2)
        sticky=0.5,  # The lower, the stickier the border
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=7.5e-3,  # Critical stretch (7.5e-3)
    ):
        n = position.shape[0]
        m = velocity.shape[0]
        assert n == m, "Positions and velocities shape not matching!"

        self.group_size = position.shape[0]
        self.velocity = velocity
        self.position = position
        self.name = name
        self.quality = quality
        self.n_particles = n_particles
        self.nu = nu
        self.E = E
        self.zeta = zeta
        self.rho_0 = rho_0
        self.sticky = sticky
        self.theta_c = theta_c
        self.theta_s = theta_s
