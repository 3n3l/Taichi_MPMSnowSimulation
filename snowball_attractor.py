from MLS_MPM import MPM
import taichi as ti


ti.init(arch=ti.gpu)  # Try to run on GPU


def main():
    mpm = MPM(
        E = 1.4e5,          # Young's modulus (1.4e5)
        nu = 0.2,           # Poisson's ratio (0.2)
        zeta = 10,          # Hardening coefficient (10)
        theta_c = 2.5e-2,   # Critical compression (2.5e-2)
        theta_s = 7.5e-3,   # Critical stretch (7.5e-3)
        rho_0 = 4e2 ,       # Initial density (4e2)
        sticky = 0.5,       # The lower, the stickier the border
        quality = 3,        # Use a larger value for higher-res simulations
        initial_velocity = [0, 0],
        initial_gravity = [0, -9.8],
        radius = 0.04,
        attractor_active = True,
    )
    mpm.run()


if __name__ == "__main__":
    main()
