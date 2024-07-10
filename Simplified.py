import taichi as ti
import numpy as np


ti.init(arch=ti.gpu)


# Parameters to control the simulation
quality = 3
n_grid = 128 * quality
n_particles = 1_000 * (quality**2)
friction = 5  # Border friction
dx = 1 / n_grid  #
inv_dx = float(n_grid)  #
dt = 1e-4 / quality  # Timestep


# Build the GGUI
window = ti.ui.Window(name="MLS-MPM", res=(720, 720), fps_limit=60)
canvas = window.get_canvas()
gui = window.get_gui()


# MPM Parameters
E = 1.4e5  # Young's modulus (1.4e5)
nu = 0.2  # Poisson's ratio (0.2)
zeta = 10  # Hardening coefficient (10)
theta_c = 3.5e-2  # Critical compression (2.5e-2)
theta_s = 5.5e-3  # Critical stretch (7.5e-3)
mu_0 = E / (2 * (1 + nu))  # Lamé parameter
lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lamé parameter
density = 4e2  # Initial density
volume = (dx * 0.5) ** 2  # Initial volume
mass = volume * density  # Initial mass


# Fields
gravity = ti.Vector.field(2, dtype=float, shape=())
g_velo = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))
g_mass = ti.field(dtype=float, shape=(n_grid, n_grid))
p_position = ti.Vector.field(2, dtype=float, shape=n_particles)
p_velocity = ti.Vector.field(2, dtype=float, shape=n_particles)
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
JP = ti.field(dtype=float, shape=n_particles)


# Build snowball, have it thrown against the wall with realistic gravity
t = np.linspace(0, 2 * np.pi, n_particles + 2, dtype=np.float32)[1:-1]
r = 0.05 * np.sqrt(np.random.rand(n_particles))
initial_position = ti.Vector.field(2, dtype=float, shape=n_particles)
initial_position.from_numpy(np.array([(r * np.sin(t)) + 0.5, (r * np.cos(t)) + 0.5], dtype=np.float32).T)
initial_velocity = ti.Vector.field(2, dtype=float, shape=n_particles)
initial_velocity.from_numpy(np.full(fill_value=[8, 0], shape=(n_particles, 2), dtype=np.float32))
gravity[None] = [0, -9.8]


@ti.kernel
def reset_grids():
    for i, j in g_mass:
        g_velo[i, j] = [0, 0]
        g_mass[i, j] = 0


@ti.kernel
def particle_to_grid():
    for p in p_position:
        base = (p_position[p] * inv_dx - 0.5).cast(int)
        fx = p_position[p] * inv_dx - base.cast(float)
        # Quadratic kernels
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # Deformation gradient update
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        # Hardening coefficient
        h = ti.max(0.1, ti.min(5, ti.exp(zeta * (1.0 - JP[p]))))
        mu, la = mu_0 * h, lambda_0 * h
        U, sigma, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            singular_value = float(sigma[d, d])
            singular_value = max(singular_value, 1 - theta_c)
            singular_value = min(singular_value, 1 + theta_s)  # Plasticity
            JP[p] *= sigma[d, d] / singular_value
            sigma[d, d] = singular_value
            J *= singular_value
        # Reconstruct elastic deformation gradient after plasticity
        F[p] = U @ sigma @ V.transpose()
        piola_kirchoff = 2 * mu * (F[p] - U @ V.transpose())
        piola_kirchoff = piola_kirchoff @ F[p].transpose()
        piola_kirchoff += ti.Matrix.identity(float, 2) * la * J * (J - 1)
        piola_kirchoff *= -dt * volume * 4 * inv_dx * inv_dx
        affine = piola_kirchoff + (mass * C[p])
        for i, j in ti.static(ti.ndrange(3, 3)):
            # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            v = mass * p_velocity[p] + affine @ dpos
            g_velo[base + offset] += weight * v
            g_mass[base + offset] += weight * mass


@ti.kernel
def momentum_to_velocity():
    for i, j in g_mass:
        if g_mass[i, j] > 0:  # No need for epsilon here
            g_velo[i, j] = g_velo[i, j] / g_mass[i, j]
            g_velo[i, j] += dt * gravity[None]
            if i < 3 or i > (n_grid - 3):  # Vertical collision
                g_velo[i, j][0] = 0
                g_velo[i, j][1] *= 1 / friction
            if j < 3 or j > (n_grid - 3):  # Horizontal collision
                g_velo[i, j][0] *= 1 / friction
                g_velo[i, j][1] = 0


@ti.kernel
def grid_to_particle():
    for p in p_position:
        base = (p_position[p] * inv_dx - 0.5).cast(int)
        fx = p_position[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        n_velocity = ti.Vector.zero(float, 2)
        n_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = g_velo[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            n_velocity += weight * g_v
            n_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        p_velocity[p], C[p] = n_velocity, n_C
        p_position[p] += dt * n_velocity  # advection


@ti.kernel
def reset_particles():
    for i in range(n_particles):
        p_position[i] = initial_position[i]
        p_velocity[i] = initial_velocity[i]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        C[i] = ti.Matrix.zero(float, 2, 2)
        JP[i] = 1


def handle_events():
    if window.get_event(ti.ui.PRESS):
        if window.event.key == "r":
            reset_particles()
        elif window.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            window.running = False  # Stop the simulation


def substep():
    for _ in range(int(2e-3 // dt)):
        reset_grids()
        particle_to_grid()
        momentum_to_velocity()
        grid_to_particle()


def render():
    canvas.set_background_color((0.054, 0.06, 0.09))
    canvas.circles(centers=p_position, radius=0.0012, color=(0.8, 0.8, 0.8))
    window.show()


def run():
    reset_particles()
    while window.running:
        handle_events()
        substep()
        render()


if __name__ == "__main__":
    run()
