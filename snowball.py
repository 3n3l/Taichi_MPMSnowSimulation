import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)  # Try to run on GPU


quality = 2 # Use a larger value for higher-res simulations
n_particles, n_grid = 5_000 * quality**2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5) ** 2, 1
p_mass = p_vol * p_rho
E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters


position = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
velocity = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # deformation gradient
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass
gravity = ti.Vector.field(2, dtype=float, shape=())
attractor_strength = ti.field(dtype=float, shape=())
attractor_pos = ti.Vector.field(2, dtype=float, shape=())


R = 0.5 # initial radius of the snowball
GRAVITY = 9.81
THETAS = np.linspace(0, 2 * np.pi, n_particles + 2, dtype=np.float32)[1:-1] # = (0, 2pi)

@ti.kernel
def substep():
    # Reset the grids
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0

    # Particle state update and scatter to grid (P2G)
    for p in position:
        base = (position[p] * inv_dx - 0.5).cast(int)
        fx = position[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # deformation gradient update
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        # Hardening coefficient: snow gets harder when compressed
        h = ti.max(0.1, ti.min(5, ti.exp(10 * (1.0 - Jp[p]))))
        # if material[p] == 1:  # jelly, make it softer
            # h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        # if material[p] == 0:  # liquid
            # mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            # if material[p] == 2:  # Snow
            new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        # if material[p] == 0:
            # Reset deformation gradient to avoid numerical instability
            # F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        # elif material[p] == 2:
            # Reconstruct elastic deformation gradient after plasticity
        F[p] = U @ sig @ V.transpose()
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * velocity[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    # Momentum to velocity
    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]
            grid_v[i, j] += dt * gravity[None] * 30  # gravity
            dist = attractor_pos[None] - dx * ti.Vector([i, j])
            grid_v[i, j] += dist / (0.01 + dist.norm()) * attractor_strength[None] * dt * 100
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0  # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j][0] > 0:
                grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0:
                grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0:
                grid_v[i, j][1] = 0

    # Grid to particle (G2P)
    for p in position:
        base = (position[p] * inv_dx - 0.5).cast(int)
        fx = position[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        velocity[p], C[p] = new_v, new_C
        position[p] += dt * velocity[p]  # advection


@ti.kernel
def reset(thetas:ti.types.ndarray()):
    for i in range(n_particles):
        radius = R * ti.sqrt(ti.random())
        position[i] = [
            radius * (ti.sin(thetas[i]) * 0.1) + 0.5,
            radius * (ti.cos(thetas[i]) * 0.1) + 0.5,
        ]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        C[i] = ti.Matrix.zero(float, 2, 2)
        velocity[i] = [0, 0]
        Jp[i] = 1


def parametrize_circle(pos) -> np.ndarray:
    thetas = np.linspace(0, 2 * np.pi, n_particles + 2)[1:-1] # = (0, 2pi)
    disk_pos = np.zeros((n_particles, 2))
    disk_pos[..., 0] = np.sin(thetas)
    disk_pos[..., 1] = np.cos(thetas)
    return disk_pos


def main():
    print("[Hint] Use WSAD/arrow keys to control gravity. Use left/right mouse buttons to attract/repel. Press R to reset.")
    gui = ti.GUI("Taichi MLS-MPM-128", res=512, background_color=0x112F41)
    gravity[None] = [0, -GRAVITY]
    reset(THETAS)

    for _ in range(20_000):
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == "r":
                reset(THETAS)
            elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                break
        # if gui.event is not None:
            # gravity[None] = [0, -GRAVITY]  # if had any event
        # if gui.is_pressed(ti.GUI.LEFT, "a"):
        #     gravity[None][0] = -1
        # if gui.is_pressed(ti.GUI.RIGHT, "d"):
        #     gravity[None][0] = 1
        # if gui.is_pressed(ti.GUI.UP, "w"):
        #     gravity[None][1] = 1
        # if gui.is_pressed(ti.GUI.DOWN, "s"):
        #     gravity[None][1] = -1
        # mouse = gui.get_cursor_pos()
        # gui.circle((mouse[0], mouse[1]), color=0x336699, radius=15)
        # attractor_pos[None] = [mouse[0], mouse[1]]
        # attractor_strength[None] = 0
        # if gui.is_pressed(ti.GUI.LMB):
            # attractor_strength[None] = 1
        # if gui.is_pressed(ti.GUI.RMB):
            # attractor_strength[None] = -1
        for _ in range(int(2e-3 // dt)):
            substep()

        gui.circles(
            position.to_numpy(),
            radius=1,
        )

        # Change to gui.show(f'{frame:06d}.png') to write images to disk
        gui.show()

if __name__ == "__main__":
    main()
