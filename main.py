from src.Configuration import Configuration
from src.MLS_MPM import MPM
import taichi as ti
import numpy as np
from argparse import ArgumentParser, RawTextHelpFormatter


ti.init(arch=ti.vulkan)


def snowball_positions(position=[[0, 0]], n_particles=1000, radius=1.0):
    n_snowballs = len(position)
    group_size = n_particles // n_snowballs
    p = np.zeros(shape=(n_particles, 2), dtype=np.float32)
    thetas = np.linspace(0, 2 * np.pi, group_size + 2, dtype=np.float32)[1:-1]
    r = radius * np.sqrt(np.random.rand(n_particles))
    for i in range(n_particles):
        j = i // group_size
        p[i, 0] = (r[i] * np.sin(thetas[i % group_size])) + position[j][0]
        p[i, 1] = (r[i] * np.cos(thetas[i % group_size])) + position[j][1]
    return p


def snowball_velocities(velocity=[[0, 0]], n_particles=1000):
    n_snowballs = len(velocity)
    group_size = n_particles // n_snowballs
    v = np.zeros(shape=(n_particles, 2), dtype=np.float32)
    for i in range(n_particles):
        j = i // group_size
        v[i, 0] = velocity[j][0]
        v[i, 1] = velocity[j][1]
    return v


def main():
    quality = 3
    radius = 0.05
    n_particles = 2_000 * (quality**2)
    configurations = [
        Configuration(
            name="Snowball hits wall",
            quality=quality,
            n_particles=n_particles,
            position=snowball_positions([[0.5, 0.5]], radius=radius, n_particles=n_particles),
            velocity=snowball_velocities([[5, 0]], n_particles=n_particles),
        ),
        Configuration(
            name="Snowball hits ground",
            quality=quality,
            n_particles=n_particles,
            position=snowball_positions([[0.5, 0.5]], radius=radius, n_particles=n_particles),
            velocity=snowball_velocities([[0, 0]], n_particles=n_particles),
        ),
        Configuration(
            name="Snowball hits snowball",
            quality=quality,
            n_particles=n_particles,
            position=snowball_positions([[0.06, 0.595], [0.94, 0.615]], radius=radius, n_particles=n_particles),
            velocity=snowball_velocities([[3, 0], [-3, 0]], n_particles=n_particles),
        ),
    ]

    print()
    p_epilog = "[Hint] Press R to reset, SPACE to pause/unpause the simulation!"
    parser = ArgumentParser(prog="main.py", epilog=p_epilog, formatter_class=RawTextHelpFormatter)
    s_help = "Show settings in subwindow."
    parser.add_argument("--showSettings", const=True, default=False, nargs="?", help=s_help)
    w_help = "Write frames to disk."
    parser.add_argument("--writeToDisk", const=True, default=False, nargs="?", help=w_help)
    p_help = "\n".join([f"{i}: {c.name}" for i, c in enumerate(configurations)])
    parser.add_argument("--configuration", default=0, nargs="?", help=p_help, type=int)
    args = parser.parse_args()
    parser.print_help()
    print()

    mpm = MPM(
        quality=quality,
        n_particles=n_particles,
        initial_gravity=[0, -9.8],
        should_write_to_disk=args.writeToDisk,
        should_show_settings=args.showSettings,
        configuration_id=args.configuration,
        configurations=configurations,
    )
    mpm.run()


if __name__ == "__main__":
    main()
