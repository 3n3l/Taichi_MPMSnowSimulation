from MLS_MPM import MPM
import taichi as ti


ti.init(arch=ti.gpu)  # Try to run on GPU


def main():
    mpm = MPM(
        quality=3,
        initial_gravity=[0, -9.8],
        initial_velocity=[5, 0],
    )
    mpm.run()


if __name__ == "__main__":
    main()
