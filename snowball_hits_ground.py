from MLS_MPM import MPM
import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)  # Try to run on GPU


def main():
    print("[Hint] Press R to reset.")
    gui = ti.GUI("Snowball hits ground", res=512, background_color=0x0E1018)
    mpm = MPM(
        gui=gui,
        quality=3,
        initial_gravity=[0, -9.8],
        initial_positions=np.array([[0.5, 0.5]], dtype=np.float32),
        initial_velocities=np.array([[0, 0]], dtype=np.float32),
        initial_radii=np.array([0.04], dtype=np.float32),
    )
    mpm.run()


if __name__ == "__main__":
    main()
