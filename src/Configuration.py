import numpy as np


class Configuration:
    def __init__(
        self,
        velocity: np.ndarray,
        position: np.ndarray,
        n_particles: int,
        quality: int,
        name: str,
    ):
        n = position.shape[0]
        m = velocity.shape[0]
        assert n == m, "Positions and velocities shape not matching!"

        self.name = name
        self.quality = quality
        self.n_particles = n_particles
        self.group_size = position.shape[0]
        self.velocity = velocity
        self.position = position
