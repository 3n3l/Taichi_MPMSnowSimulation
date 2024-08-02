from Configuration import Configuration
from datetime import datetime
import taichi as ti
import os


@ti.data_oriented
class Simulation:
    def __init__(
        self,
        quality: int,
        n_particles: int,
        configurations: list[Configuration],
    ):
        # MPM Parameters that are configuration independent
        self.quality = quality
        self.n_particles = n_particles
        self.n_grid = 128 * quality
        self.dx = 1 / self.n_grid
        self.inv_dx = float(self.n_grid)
        self.dt = 1e-4 / self.quality
        self.rho_0 = 4e2
        self.p_vol = (self.dx * 0.5) ** 2
        self.p_mass = self.p_vol * self.rho_0

        # Parameters to control the simulation
        self.window = ti.ui.Window(name="MLS-MPM", res=(720, 720), fps_limit=60)
        self.gui = self.window.get_gui()
        self.canvas = self.window.get_canvas()
        self.frame = 0
        self.is_paused = True
        self.should_write_to_disk = False
        self.is_showing_settings = not self.is_paused
        # Create folders to dump the frames
        self.directory = datetime.now().strftime("%d%m%Y_%H%M")
        if not os.path.exists(".output"):
            os.makedirs(".output")
        if not os.path.exists(f".output/{self.directory}"):
            os.makedirs(f".output/{self.directory}")

        # Fields
        self.g_velo = ti.Vector.field(2, dtype=float, shape=(self.n_grid, self.n_grid))
        self.g_mass = ti.field(dtype=float, shape=(self.n_grid, self.n_grid))
        self.initial_position = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.initial_velocity = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.p_position = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.p_velocity = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.p_color = ti.Vector.field(3, dtype=float, shape=self.n_particles)
        self.C = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles)  # affine velocity field
        self.F = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles)  # deformation gradient
        self.JP = ti.field(dtype=float, shape=self.n_particles)  # plastic deformation
        self.gravity = ti.Vector.field(2, dtype=float, shape=())
        self.stickiness = ti.field(dtype=float, shape=())
        self.friction = ti.field(dtype=float, shape=())
        self.theta_c = ti.field(dtype=float, shape=())
        self.theta_s = ti.field(dtype=float, shape=())
        self.zeta = ti.field(dtype=int, shape=())
        self.nu = ti.field(dtype=float, shape=())
        self.E = ti.field(dtype=float, shape=())
        self.lambda_0 = ti.field(dtype=float, shape=())
        self.mu_0 = ti.field(dtype=float, shape=())

        # Load the initial configuration
        self.configuration_id = 0
        self.configurations = configurations
        self.model = configurations[self.configuration_id]
        self.load_configuration()

    @ti.kernel
    def reset_grids(self):
        for i, j in self.g_mass:
            self.g_velo[i, j] = [0, 0]
            self.g_mass[i, j] = 0

    @ti.kernel
    def particle_to_grid(self):
        for p in self.p_position:
            base = (self.p_position[p] * self.inv_dx - 0.5).cast(int)
            fx = self.p_position[p] * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            # Deformation gradient update
            self.F[p] = (ti.Matrix.identity(float, 2) + self.dt * self.C[p]) @ self.F[p]
            # Apply snow hardening by adjusting Lame parameters
            h = ti.max(0.1, ti.min(5, ti.exp(self.zeta[None] * (1.0 - self.JP[p]))))
            mu, la = self.mu_0[None] * h, self.lambda_0[None] * h
            U, sigma, V = ti.svd(self.F[p])
            J = 1.0
            for d in ti.static(range(2)):
                singular_value = float(sigma[d, d])
                singular_value = max(singular_value, 1 - self.theta_c[None])
                singular_value = min(singular_value, 1 + self.theta_s[None])
                self.JP[p] *= sigma[d, d] / singular_value
                sigma[d, d] = singular_value
                J *= singular_value
            # Reconstruct elastic deformation gradient after plasticity
            self.F[p] = U @ sigma @ V.transpose()
            piola_kirchoff = 2 * mu * (self.F[p] - U @ V.transpose()) @ self.F[p].transpose()
            piola_kirchoff += ti.Matrix.identity(float, 2) * la * J * (J - 1)
            piola_kirchoff *= -self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx
            affine = piola_kirchoff + self.p_mass * self.C[p]
            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1]
                v = self.p_mass * self.p_velocity[p] + affine @ dpos
                self.g_velo[base + offset] += weight * v
                self.g_mass[base + offset] += weight * self.p_mass

    @ti.kernel
    def momentum_to_velocity(self):
        for i, j in self.g_mass:
            if self.g_mass[i, j] > 0:  # No need for epsilon here
                self.g_velo[i, j] = (1 / self.g_mass[i, j]) * self.g_velo[i, j]
                self.g_velo[i, j] += self.dt * self.gravity[None]  # gravity
                # Boundary conditions for the grid velocities, this implements sticky collisions
                if i < 3 or i > (self.n_grid - 3):  # Vertical collision
                    self.g_velo[i, j][0] = 0
                    self.g_velo[i, j][1] *= 1 / self.friction[None]
                if j < 3 or j > (self.n_grid - 3):  # Horizontal collision
                    self.g_velo[i, j][0] *= 1 / self.friction[None]
                    self.g_velo[i, j][1] = 0

    @ti.kernel
    def grid_to_particle(self):
        for p in self.p_position:
            base = (self.p_position[p] * self.inv_dx - 0.5).cast(int)
            fx = self.p_position[p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            n_velocity = ti.Vector.zero(float, 2)
            n_C = ti.Matrix.zero(float, 2, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                dpos = ti.Vector([i, j]).cast(float) - fx
                g_v = self.g_velo[base + ti.Vector([i, j])]
                weight = w[i][0] * w[j][1]
                n_velocity += weight * g_v
                n_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
            self.p_velocity[p], self.C[p] = n_velocity, n_C
            self.p_position[p] += self.dt * n_velocity  # Advection

    @ti.kernel
    def reset_particles(self):
        self.gravity[None] = [0, -9.8]
        for i in range(self.n_particles):
            self.p_position[i] = self.initial_position[i]
            self.p_velocity[i] = self.initial_velocity[i]
            # self.p_color[i] = self.initial_velocity[i]
            self.F[i] = ti.Matrix([[1, 0], [0, 1]])
            self.C[i] = ti.Matrix.zero(float, 2, 2)
            self.JP[i] = 1

    def reset(self):
        self.frame = 0
        self.directory = datetime.now().strftime("%d%m%Y_%H%M")
        os.makedirs(f".output/{self.directory}")
        self.reset_particles()

    def load_configuration(self):
        self.initial_position.from_numpy(self.model.position)
        self.initial_velocity.from_numpy(self.model.velocity)
        self.p_color.from_numpy(self.model.color)
        self.stickiness[None] = self.model.stickiness
        self.friction[None] = self.model.friction
        self.lambda_0[None] = self.model.lambda_0
        self.theta_c[None] = self.model.theta_c
        self.theta_s[None] = self.model.theta_s
        self.zeta[None] = self.model.zeta
        self.mu_0[None] = self.model.mu_0
        self.nu[None] = self.model.nu
        self.E[None] = self.model.E

    def handle_events(self):
        if self.window.get_event(ti.ui.PRESS):
            if self.window.event.key == "r":
                self.reset_particles()
            elif self.window.event.key in [ti.GUI.BACKSPACE, "s"]:
                self.should_write_to_disk = not self.should_write_to_disk
            elif self.window.event.key in [ti.GUI.SPACE, "p"]:
                self.is_paused = not self.is_paused
            elif self.window.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                self.window.running = False  # Stop the simulation

    def substep(self):
        if not self.is_paused:
            for _ in range(int(2e-3 // self.dt)):
                self.reset_grids()
                self.particle_to_grid()
                self.momentum_to_velocity()
                self.grid_to_particle()

    def show_configurations(self, subwindow):
        prev_configuration_id = self.configuration_id
        for i in range(len(self.configurations)):
            name = self.configurations[i].name
            if subwindow.checkbox(name, self.configuration_id == i):
                self.configuration_id = i
        if self.configuration_id != prev_configuration_id:
            _id = self.configuration_id
            self.model = self.configurations[_id]
            self.load_configuration()
            self.reset_particles()
            self.is_paused = True

    def show_parameters(self, subwindow):
        self.stickiness[None] = subwindow.slider_float("stickiness", self.stickiness[None], 1.0, 5.0)
        self.friction[None] = subwindow.slider_float("friction", self.friction[None], 1.0, 5.0)
        self.theta_c[None] = subwindow.slider_float("theta_c", self.theta_c[None], 1e-2, 3.5e-2)
        self.theta_s[None] = subwindow.slider_float("theta_s", self.theta_s[None], 5.0e-3, 10e-3)
        self.zeta[None] = subwindow.slider_int("zeta", self.zeta[None], 3, 10)
        self.nu[None] = subwindow.slider_float("nu", self.nu[None], 0.1, 0.4)
        self.E[None] = subwindow.slider_float("E", self.E[None], 4.8e4, 2.8e5)
        self.lambda_0[None] = self.E[None] * self.nu[None] / ((1 + self.nu[None]) * (1 - 2 * self.nu[None]))
        self.mu_0[None] = self.E[None] / (2 * (1 + self.nu[None]))

    def show_buttons(self, subwindow):
        if subwindow.button(" Stop recording  " if self.should_write_to_disk else " Start recording "):
            self.should_write_to_disk = not self.should_write_to_disk
        if subwindow.button(" Reset Particles "):
            self.reset()
        if subwindow.button(" Start Simulation"):
            self.is_paused = False

    def show_settings(self):
        if not self.is_paused:
            self.is_showing_settings = False
            return  # don't bother
        self.is_showing_settings = True
        with self.gui.sub_window("Settings", 0.01, 0.01, 0.98, 0.98) as subwindow:
            self.show_parameters(subwindow)
            self.show_configurations(subwindow)
            self.show_buttons(subwindow)

    def render(self):
        self.canvas.set_background_color((0.054, 0.06, 0.09))
        self.canvas.circles(centers=self.p_position, radius=0.0015, per_vertex_color=self.p_color)
        if self.should_write_to_disk and not self.is_paused and not self.is_showing_settings:
            self.window.save_image(f".output/{self.directory}/{self.frame:06d}.png")
            self.frame += 1
        self.window.show()

    def run(self):
        self.reset_particles()
        while self.window.running:
            self.handle_events()
            self.show_settings()
            self.substep()
            self.render()
