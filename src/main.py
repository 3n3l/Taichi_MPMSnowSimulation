from Configuration import Configuration
from MLS_MPM import Simulation
import taichi as ti
import numpy as np


def snowball_positions(positions=[[0, 0]], radii=[0.5], n_particles=1000):
    n_snowballs = len(positions)
    group_size = n_particles // n_snowballs
    p = np.zeros(shape=(n_particles, 2), dtype=np.float32)
    thetas = np.linspace(0, 2 * np.pi, group_size + 2, dtype=np.float32)[1:-1]
    for i in range(n_particles):
        j = i // group_size
        r = radii[j] * np.sqrt(np.random.rand())
        p[i, 0] = (r * np.sin(thetas[i % group_size])) + positions[j][0]
        p[i, 1] = (r * np.cos(thetas[i % group_size])) + positions[j][1]
    return p


def map_to_snowball(to_map=[[0, 0]], n_particles=1000):
    n_snowballs = len(to_map)
    group_size = n_particles // n_snowballs
    v = np.zeros(shape=(n_particles, len(to_map[0])), dtype=np.float32)
    for i in range(n_particles):
        j = i // group_size
        v[i] = to_map[j]
    return v


def main():
    ti.init(arch=ti.gpu)
    quality = 3
    n_particles = 2_000 * (quality**2)
    configurations = [
        Configuration(
            name="Snowball hits wall (sticky) [1]",
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=2.5e-2,  # Critical compression (2.5e-2)
            theta_s=5.0e-3,  # Critical stretch (7.5e-3)
            stickiness=2,  # Higher value means a stickier border
            friction=2,  # Higher value means the border has more friction
            position=snowball_positions([[0.5, 0.5]], radii=[0.05], n_particles=n_particles),
            color=map_to_snowball([[0.9, 0.9, 0.9]], n_particles=n_particles),
            velocity=map_to_snowball([[5, 0]], n_particles=n_particles),
        ),
        Configuration(
            name="Snowball hits wall (slippery) [2]",
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=2.5e-2,  # Critical compression (2.5e-2)
            theta_s=7.5e-3,  # Critical stretch (7.5e-3)
            stickiness=1,  # Higher value means a stickier border
            friction=1,  # Higher value means the border has more friction
            position=snowball_positions([[0.5, 0.5]], radii=[0.05], n_particles=n_particles),
            color=map_to_snowball([[0.9, 0.9, 0.9]], n_particles=n_particles),
            velocity=map_to_snowball([[5, 0]], n_particles=n_particles),
        ),
        Configuration(
            name="Snowball hits ground (sticky) [1]",
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=2.5e-2,  # Critical compression (2.5e-2)
            theta_s=4.5e-3,  # Critical stretch (7.5e-3)
            stickiness=2,  # Higher value means a stickier border
            friction=2,  # Higher value means the border has more friction
            position=snowball_positions([[0.5, 0.5]], radii=[0.05], n_particles=n_particles),
            color=map_to_snowball([[0.9, 0.9, 0.9]], n_particles=n_particles),
            velocity=map_to_snowball([[0, 0]], n_particles=n_particles),
        ),
        Configuration(
            name="Snowball hits ground (slippery) [2]",
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=2.5e-2,  # Critical compression (2.5e-2)
            theta_s=4.5e-3,  # Critical stretch (7.5e-3)
            stickiness=1,  # Higher value means a stickier border
            friction=1,  # Higher value means the border has more friction
            position=snowball_positions([[0.5, 0.5]], radii=[0.05], n_particles=n_particles),
            color=map_to_snowball([[0.9, 0.9, 0.9]], n_particles=n_particles),
            velocity=map_to_snowball([[0, 0]], n_particles=n_particles),
        ),
        Configuration(
            name="Snowball hits snowball [1]",
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=8,  # Hardening coefficient (10)
            theta_c=1.5e-2,  # Critical compression (2.5e-2)
            theta_s=5.5e-3,  # Critical stretch (7.5e-3)
            stickiness=2,  # Higher value means a stickier border
            friction=2,  # Higher value means the border has more friction
            position=snowball_positions([[0.07, 0.595], [0.91, 0.615]], radii=[0.04, 0.06], n_particles=n_particles),
            color=map_to_snowball([[0.9, 0.9, 0.9], [0.4, 0.4, 0.4]], n_particles=n_particles),
            velocity=map_to_snowball([[6, 0], [-3, 0]], n_particles=n_particles),
        ),
        Configuration(
            name="Snowball hits snowball [2]",
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=5,  # Hardening coefficient (10)
            theta_c=2.5e-2,  # Critical compression (2.5e-2)
            theta_s=4.0e-3,  # Critical stretch (7.5e-3)
            stickiness=2,  # Higher value means a stickier border
            friction=2,  # Higher value means the border has more friction
            position=snowball_positions([[0.06, 0.5], [0.94, 0.53]], radii=[0.05, 0.05], n_particles=n_particles),
            color=map_to_snowball([[0.9, 0.9, 0.9], [0.9, 0.9, 0.9]], n_particles=n_particles),
            velocity=map_to_snowball([[4, 0], [-4, 0]], n_particles=n_particles),
        ),
        Configuration(
            name="Snowball hits snowball (colored) [3]",
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=5,  # Hardening coefficient (10)
            theta_c=2.5e-2,  # Critical compression (2.5e-2)
            theta_s=4.0e-3,  # Critical stretch (7.5e-3)
            stickiness=2,  # Higher value means a stickier border
            friction=2,  # Higher value means the border has more friction
            position=snowball_positions([[0.06, 0.5], [0.94, 0.53]], radii=[0.05, 0.05], n_particles=n_particles),
            color=map_to_snowball([[1, 0.5, 0.5], [0.5, 0.5, 1]], n_particles=n_particles),
            velocity=map_to_snowball([[4, 0], [-4, 0]], n_particles=n_particles),
        ),
    ]

    print("-" * 150)
    print("[Hint] Press R to [R]eset, P|SPACE to [P]ause/un[P]ause and S|BACKSPACE to [S]tart recording!")
    print("-" * 150)

    simulation = Simulation(quality=quality, n_particles=n_particles, configurations=configurations)
    simulation.run()


if __name__ == "__main__":
    main()
