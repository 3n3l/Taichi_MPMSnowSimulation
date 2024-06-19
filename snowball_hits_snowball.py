from MLS_MPM import MPM
import taichi as ti
import numpy as np


def main():
    print("[Hint] Press R to reset.")
    window = ti.ui.Window(name="Snowball hits snowball", res=(512, 512))
    mpm = MPM(
        window=window,
        E=1.4e5,  # Young's modulus (1.4e5)
        nu=0.2,  # Poisson's ratio (0.2)
        zeta=10,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=7.5e-3,  # Critical stretch (7.5e-3)
        rho_0=4e2,  # Initial density (4e2)
        sticky=0.5,  # The lower, the stickier the border
        quality=3,  # Use a larger value for higher-res simulations
        initial_gravity=[0, -9.8],
        initial_positions=np.array([[0.05, 0.595], [0.95, 0.615]], dtype=np.float32),
        initial_velocities=np.array([[3, 0], [-3, 0]], dtype=np.float32),
        initial_radii=np.array([0.1, 0.1], dtype=np.float32),
    )
    mpm.run()


if __name__ == "__main__":
    main()
