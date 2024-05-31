import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)  # Try to run on GPU


# Parameter starting points for MPM
E = 1.4e5  # Young's modulus
nu = 0.2  # Poisson's ratio
zeta = 10 # Hardening coefficient
theta_c = 2.5e-2 # Critical compression
theta_s = 7.5e-3 # Critical stretch
rho_0 = 4e2  # Initial density
mu_0 = E / (2 * (1 + nu)) # Lame parameters
lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
 

# Parameter to control the simulation
quality = 4 # Use a larger value for higher-res simulations
n_particles, n_grid = 1_000 * quality**2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol = (dx * 0.5) ** 2
p_mass = p_vol * rho_0
sticky = 0.5  # The lower, the stickier the border


position = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
velocity = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # deformation gradient
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
grid_velo = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))  # grid node momentum/velocity
grid_mass = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass
gravity = ti.Vector.field(2, dtype=float, shape=())
attractor_strength = ti.field(dtype=float, shape=())
attractor_pos = ti.Vector.field(2, dtype=float, shape=())


# Control gravity, construct snowball
t = np.linspace(0, 2 * np.pi, n_particles + 2, dtype=np.float32)[1:-1] # in (0, 2pi)
thetas = ti.field(dtype=float, shape=n_particles)  # used to parametrize the snowball
thetas.from_numpy(t)
GRAVITY = 9.81
R = 0.05 # initial radius of the snowball


@ti.kernel
def substep():
    # Reset the grids
    for i, j in grid_mass:
        grid_velo[i, j] = [0, 0]
        grid_mass[i, j] = 0

    # Particle state update and scatter to grid (P2G)
    for p in position:
        base = (position[p] * inv_dx - 0.5).cast(int)
        fx = position[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # deformation gradient update
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        # Hardening coefficient: snow gets harder when compressed
        h = ti.exp(zeta * (1.0 - Jp[p]))
        mu, la = mu_0 * h, lambda_0 * h
        U, sigma, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            singular_value = float(sigma[d, d])
            singular_value = max(singular_value, 1 - theta_c)
            singular_value = min(singular_value, 1 + theta_s)  # Plasticity
            Jp[p] *= sigma[d, d] / singular_value
            sigma[d, d] = singular_value
            J *= singular_value
        # Reconstruct elastic deformation gradient after plasticity
        F[p] = U @ sigma @ V.transpose()
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_velo[base + offset] += weight * (p_mass * velocity[p] + affine @ dpos)
            grid_mass[base + offset] += weight * p_mass

    # Momentum to velocity
    for i, j in grid_mass:
        if grid_mass[i, j] > 0:  # No need for epsilon here
            grid_velo[i, j] = (1 / grid_mass[i, j]) * grid_velo[i, j]
            grid_velo[i, j] += dt * gravity[None]  # gravitty
            # dist = attractor_pos[None] - dx * ti.Vector([i, j])
            # grid_velo[i, j] += dist / (0.01 + dist.norm()) * attractor_strength[None] * dt * 100
            # Boundary conditions
            if i < 3 and grid_velo[i, j][0] < 0:
                grid_velo[i, j][0] = 0
                grid_velo[i, j][1] *= sticky
            if i > n_grid - 3 and grid_velo[i, j][0] > 0:
                grid_velo[i, j][0] = 0
                grid_velo[i, j][1] *= sticky
            if j < 3 and grid_velo[i, j][1] < 0:
                grid_velo[i, j][0] *= sticky
                grid_velo[i, j][1] = 0
            if j > n_grid - 3 and grid_velo[i, j][1] > 0:
                grid_velo[i, j][0] *= sticky
                grid_velo[i, j][1] = 0

    # Grid to particle (G2P)
    for p in position:
        base = (position[p] * inv_dx - 0.5).cast(int)
        fx = position[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            # Loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_velo[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        velocity[p], C[p] = new_v, new_C
        position[p] += dt * velocity[p]  # advection


@ti.kernel
def reset():
    for i in range(n_particles):
        radius = R * ti.sqrt(ti.random())
        position[i] = [
            radius * (ti.sin(thetas[i])) + 0.5,
            radius * (ti.cos(thetas[i])) + 0.5,
            # radius * (ti.cos(thetas[i])) + 1.0 - (R + 0.01),
        ]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        C[i] = ti.Matrix.zero(float, 2, 2)
        velocity[i] = [0, 0]
        Jp[i] = 1


def main():
    # print("[Hint] Use WSAD/arrow keys to control gravity. Use left/right mouse buttons to attract/repel. Press R to reset.")
    gui = ti.GUI("Taichi MLS-MPM-128", res=512, background_color=0x112F41)
    gravity[None] = [0, -GRAVITY]
    reset()

    for _ in range(20_000):
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == "r":
                reset()
            elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                break
        # if gui.event is not None:
        #     gravity[None] = [0, 0]  # if had any event
        # if gui.is_pressed(ti.GUI.LEFT, "a"):
        #     gravity[None][0] = -GRAVITY
        # if gui.is_pressed(ti.GUI.RIGHT, "d"):
        #     gravity[None][0] = GRAVITY
        # if gui.is_pressed(ti.GUI.UP, "w"):
        #     gravity[None][1] = GRAVITY
        # if gui.is_pressed(ti.GUI.DOWN, "s"):
        #     gravity[None][1] = -GRAVITY

        for _ in range(int(2e-3 // dt)):
            substep()

        gui.circles(position.to_numpy(), radius=1)
        gui.show() # change to gui.show(f'{frame:06d}.png') to write images to disk

if __name__ == "__main__":
    main()
