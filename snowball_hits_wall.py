from MLS_MPM import MPM
import taichi as ti
import numpy as np


ti.init(arch=ti.gpu)  # Try to run on GPU


def main():
    mpm = MPM(
        quality=3,
        initial_gravity=[0, -9.8],
        initial_positions=np.array([[0.5, 0.5]], dtype=np.float32),
        initial_velocities=np.array([[5, 0]], dtype=np.float32),
        initial_radii=np.array([0.04], dtype=np.float32),
    )
    mpm.run()


if __name__ == "__main__":
    main()
