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
        should_show_settings=False,
        should_write_to_disk=False,
        initial_gravity=[0, 0],  # Gravity of the simulation ([0, 0])
        configuration_id=0,
        is_paused=False,
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
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        self.is_paused = is_paused
        self.configurations = configurations
        self.initial_gravity = initial_gravity
        self.configuration_id = configuration_id
        self.should_show_settings = should_show_settings
        self.should_write_to_disk = should_write_to_disk
        self.frame = 0  # for writing this to disk
        # Create folders to dump the frames
        self.directory = datetime.now().strftime("%d%m%Y_%H%M")
        if not os.path.exists(".output"):
            os.makedirs(".output")
        if not os.path.exists(f".output/{self.directory}"):
            os.makedirs(f".output/{self.directory}")

        # Fields
        self.grid_velo = ti.Vector.field(2, dtype=float, shape=(self.n_grid, self.n_grid))
        self.grid_mass = ti.field(dtype=float, shape=(self.n_grid, self.n_grid))
        self.initial_position = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.initial_velocity = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.position = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.velocity = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.C = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles)  # affine velocity field
        self.F = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles)  # deformation gradient
        self.Jp = ti.field(dtype=float, shape=self.n_particles)  # plastic deformation
        self.gravity = ti.Vector.field(2, dtype=float, shape=())

        # Load the initial configuration
        self.configuration = configurations[configuration_id]
        self.load_configuration()

    @ti.kernel
    def reset_grids(self):
        for i, j in self.grid_mass:
            self.grid_velo[i, j] = [0, 0]
            self.grid_mass[i, j] = 0

    @ti.kernel
    def particle_to_grid(self, lambda_0: float, mu_0: float, zeta: float, theta_c: float, theta_s: float):
        for p in self.position:
            base = (self.position[p] * self.inv_dx - 0.5).cast(int)
            fx = self.position[p] * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            # Deformation gradient update
            self.F[p] = (ti.Matrix.identity(float, 2) + self.dt * self.C[p]) @ self.F[p]
            # Hardening coefficient: snow gets harder when compressed,
            # clamp this to stop the rebound from compressed snow
            h = ti.max(0.1, ti.min(zeta, ti.exp(zeta * (1.0 - self.Jp[p]))))
            mu, la = mu_0 * h, lambda_0 * h
            U, sigma, V = ti.svd(self.F[p])
            J = 1.0
            for d in ti.static(range(2)):
                singular_value = float(sigma[d, d])
                singular_value = max(singular_value, 1 - theta_c)
                singular_value = min(singular_value, 1 + theta_s)  # Plasticity
                self.Jp[p] *= sigma[d, d] / singular_value
                sigma[d, d] = singular_value
                J *= singular_value
            # Reconstruct elastic deformation gradient after plasticity
            self.F[p] = U @ sigma @ V.transpose()
            stress = 2 * mu * (self.F[p] - U @ V.transpose()) @ self.F[p].transpose()
            stress = stress + ti.Matrix.identity(float, 2) * la * J * (J - 1)
            stress = stress * (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx)
            affine = stress + self.p_mass * self.C[p]
            for i, j in ti.static(ti.ndrange(3, 3)):
                # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1]
                v = self.p_mass * self.velocity[p] + affine @ dpos
                self.grid_velo[base + offset] += weight * v
                self.grid_mass[base + offset] += weight * self.p_mass

    @ti.kernel
    def momentum_to_velocity(self, friction: float):
        for i, j in self.grid_mass:
            if self.grid_mass[i, j] > 0:  # No need for epsilon here
                self.grid_velo[i, j] = (1 / self.grid_mass[i, j]) * self.grid_velo[i, j]
                self.grid_velo[i, j] += self.dt * self.gravity[None]  # gravity
                # Boundary conditions for the grid velocities
                collision_left = i < 3 and self.grid_velo[i, j][0] < 0
                collision_right = i > (self.n_grid - 3) and self.grid_velo[i, j][0] > 0
                if collision_left or collision_right:
                    self.grid_velo[i, j][0] = 0
                    self.grid_velo[i, j][1] *= 1 / friction
                collision_top = j < 3 and self.grid_velo[i, j][1] < 0
                collision_bottom = j > (self.n_grid - 3) and self.grid_velo[i, j][1] > 0
                if collision_top or collision_bottom:
                    self.grid_velo[i, j][0] *= 1 / friction
                    self.grid_velo[i, j][1] = 0

    @ti.kernel
    def grid_to_particle(self, stickiness: float, friction: float):
        for p in self.position:
            base = (self.position[p] * self.inv_dx - 0.5).cast(int)
            fx = self.position[p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(float, 2)
            new_C = ti.Matrix.zero(float, 2, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):
                # Loop over 3x3 grid node neighborhood
                dpos = ti.Vector([i, j]).cast(float) - fx
                g_v = self.grid_velo[base + ti.Vector([i, j])]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
            # Dampen velocity if the particle is close to a boundary
            position = self.position[p]
            collision_horizont = position[0] < 0.01 or position[0] > 0.99
            collision_vertical = position[1] < 0.01 or position[1] > 0.99
            if collision_horizont:
                new_v[0] *= 1 / stickiness
                new_v[1] *= 1 / friction
            if collision_vertical:
                new_v[0] *= 1 / friction
                new_v[1] *= 1 / stickiness
            self.velocity[p], self.C[p] = new_v, new_C
            self.position[p] += self.dt * new_v  # advection

    @ti.kernel
    def reset_particles(self):
        self.gravity[None] = self.initial_gravity
        for i in range(self.n_particles):
            self.position[i] = self.initial_position[i]
            self.velocity[i] = self.initial_velocity[i]
            self.F[i] = ti.Matrix([[1, 0], [0, 1]])
            self.C[i] = ti.Matrix.zero(float, 2, 2)
            self.Jp[i] = 1

    def load_configuration(self):
        self.initial_position.from_numpy(self.configuration.position)
        self.initial_velocity.from_numpy(self.configuration.velocity)
        # Save configuration variables, so these won't be overriden
        self.stickiness = self.configuration.stickiness
        self.friction = self.configuration.friction
        self.lambda_0 = self.configuration.lambda_0
        self.theta_c = self.configuration.theta_c
        self.theta_s = self.configuration.theta_s
        self.zeta = self.configuration.zeta
        self.mu_0 = self.configuration.mu_0
        self.nu = self.configuration.nu
        self.E = self.configuration.E

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
                self.particle_to_grid(self.lambda_0, self.mu_0, self.zeta, self.theta_c, self.theta_s)
                self.momentum_to_velocity(self.friction)
                self.grid_to_particle(self.stickiness, self.friction)

    def show_settings(self):
        if not self.should_show_settings or not self.is_paused:
            return  # don't bother
        with self.gui.sub_window("Settings", 0.01, 0.01, 0.98, 0.98) as w:
            # Parameters
            self.stickiness = w.slider_int(text="stickiness", old_value=self.stickiness, minimum=1, maximum=10)
            self.friction = w.slider_int(text="friction", old_value=self.friction, minimum=1, maximum=10)
            self.theta_c = w.slider_float(text="theta_c", old_value=self.theta_c, minimum=0, maximum=5e-2)
            self.theta_s = w.slider_float(text="theta_s", old_value=self.theta_s, minimum=0, maximum=15e-3)
            self.zeta = w.slider_int(text="zeta", old_value=self.zeta, minimum=1, maximum=20)
            self.nu = w.slider_float(text="nu", old_value=self.nu, minimum=0, maximum=1)
            self.E = w.slider_float(text="E", old_value=self.E, minimum=4.8e4, maximum=4.8e5)
            self.mu_0 = self.E / (2 * (1 + self.nu))
            self.lambda_0 = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
            # Configurations
            prev_configuration_id = self.configuration_id
            for i in range(len(self.configurations)):
                name = self.configurations[i].name
                if w.checkbox(name, self.configuration_id == i):
                    self.configuration_id = i
            if self.configuration_id != prev_configuration_id:
                _id = self.configuration_id
                self.configuration = self.configurations[_id]
                self.load_configuration()
                self.reset_particles()
                self.is_paused = True
            # Write to disk
            if self.should_write_to_disk:
                if w.button(" Stop recording  "):
                    self.should_write_to_disk = False
            else:
                if w.button(" Start recording "):
                    self.should_write_to_disk = True
            # Reset
            if w.button(" Reset Particles "):
                self.reset_particles()
            # Pause/Unpause
            if w.button(" Start Simulation"):
                self.is_paused = False

    def render(self):
        self.canvas.set_background_color((0.054, 0.06, 0.09))
        self.canvas.circles(centers=self.position, radius=0.0016, color=(0.8, 0.8, 0.8))
        if self.should_write_to_disk and not self.is_paused:
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
