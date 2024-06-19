from MLS_MPM import MPM
import taichi as ti
import numpy as np


ti.init(arch=ti.gpu)  # Try to run on GPU


def main():
    print("[Hint] Use left/right mouse buttons to attract/repel and start the simulation. Press R to reset.")
    window = ti.ui.Window(name="Attractor", res=(512, 512))
    mpm = MPM(
        window=window,
        quality=3,
        initial_gravity=[0, -9.8],
        attractor_active=True,
        initial_positions=np.array([[0.5, 0.5]], dtype=np.float32),
        initial_velocities=np.array([[0, 0]], dtype=np.float32),
        initial_radii=np.array([0.04], dtype=np.float32),
    )
    mpm.run()


if __name__ == "__main__":
    main()
