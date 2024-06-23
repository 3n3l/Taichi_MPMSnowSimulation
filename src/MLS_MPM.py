from src.Configuration import Configuration
import taichi as ti


@ti.data_oriented
class MPM:
    def __init__(
        self,
        configurations: list[Configuration],
        E=1.4e5,  # Young's modulus (1.4e5)
        nu=0.2,  # Poisson's ratio (0.2)
        zeta=10,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=7.5e-3,  # Critical stretch (7.5e-3)
        rho_0=4e2,  # Initial density (4e2)
        sticky=0.5,  # The lower, the stickier the border
        quality=1,  # Use a larger value for higher-res simulations
        n_particles=10_000,
        initial_gravity=[0, 0],  # Gravity of the simulation ([0, 0])
    ):
        # Parameters starting points for MPM
        self.E = E
        self.nu = nu
        self.zeta = zeta
        self.theta_c = theta_c
        self.theta_s = theta_s
        self.rho_0 = rho_0
        self.mu_0 = E / (2 * (1 + nu))  # Lame parameters
        self.lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

        self.write_to_disk = False
        self.is_paused = True
        self.frame = 0  # for writing this to disk

        self.configuration_id = 0
        self.configurations = configurations
        self.configuration = configurations[self.configuration_id]
        self.initial_position = ti.Vector.field(2, dtype=float, shape=self.configuration.n_particles)  # position
        self.initial_velocity = ti.Vector.field(2, dtype=float, shape=self.configuration.n_particles)  # velocity
        self.initial_position.from_numpy(self.configuration.position)
        self.initial_velocity.from_numpy(self.configuration.velocity)

        # Parameters to control the simulation
        self.window = ti.ui.Window(name="MLS-MPM", res=(720, 720), fps_limit=60)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        self.quality = quality
        self.n_particles = n_particles
        self.n_grid = 128 * self.quality
        self.dx = 1 / self.n_grid
        self.inv_dx = float(self.n_grid)
        self.dt = 1e-4 / self.configuration.quality
        self.p_vol = (self.dx * 0.5) ** 2
        self.p_mass = self.p_vol * rho_0
        self.sticky = sticky
        self.initial_gravity = initial_gravity

        # Fields
        self.position = ti.Vector.field(2, dtype=float, shape=self.configuration.n_particles)  # position
        self.velocity = ti.Vector.field(2, dtype=float, shape=self.configuration.n_particles)  # velocity
        self.C = ti.Matrix.field(2, 2, dtype=float, shape=self.configuration.n_particles)  # affine velocity field
        self.F = ti.Matrix.field(2, 2, dtype=float, shape=self.configuration.n_particles)  # deformation gradient
        self.Jp = ti.field(dtype=float, shape=self.configuration.n_particles)  # plastic deformation
        self.grid_velo = ti.Vector.field(2, dtype=float, shape=(self.n_grid, self.n_grid))  # grid node momentum
        self.grid_mass = ti.field(dtype=float, shape=(self.n_grid, self.n_grid))  # grid node mass
        self.gravity = ti.Vector.field(2, dtype=float, shape=())

    @ti.kernel
    def reset_grids(self):
        for i, j in self.grid_mass:
            self.grid_velo[i, j] = [0, 0]
            self.grid_mass[i, j] = 0

    @ti.kernel
    def particle_to_grid(self):
        for p in self.position:
            base = (self.position[p] * self.inv_dx - 0.5).cast(int)
            fx = self.position[p] * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            # Deformation gradient update
            self.F[p] = (ti.Matrix.identity(float, 2) + self.dt * self.C[p]) @ self.F[p]
            # Hardening coefficient: snow gets harder when compressed,
            # clamp this to stop the rebound from compressed snow
            h = ti.max(0.1, ti.min(5, ti.exp(self.zeta * (1.0 - self.Jp[p]))))
            mu, la = self.mu_0 * h, self.lambda_0 * h
            U, sigma, V = ti.svd(self.F[p])
            J = 1.0
            for d in ti.static(range(2)):
                singular_value = float(sigma[d, d])
                singular_value = max(singular_value, 1 - self.theta_c)
                singular_value = min(singular_value, 1 + self.theta_s)  # Plasticity
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
    def momentum_to_velocity(self):
        for i, j in self.grid_mass:
            if self.grid_mass[i, j] > 0:  # No need for epsilon here
                self.grid_velo[i, j] = (1 / self.grid_mass[i, j]) * self.grid_velo[i, j]
                self.grid_velo[i, j] += self.dt * self.gravity[None]  # gravity
                # Boundary conditions for the grid velocities
                collision_left = i < 3 and self.grid_velo[i, j][0] < 0
                collision_right = i > (self.n_grid - 3) and self.grid_velo[i, j][0] > 0
                if collision_left or collision_right:
                    self.grid_velo[i, j][0] = 0
                    self.grid_velo[i, j][1] *= self.sticky
                collision_top = j < 3 and self.grid_velo[i, j][1] < 0
                collision_bottom = j > (self.n_grid - 3) and self.grid_velo[i, j][1] > 0
                if collision_top or collision_bottom:
                    self.grid_velo[i, j][0] *= self.sticky
                    self.grid_velo[i, j][1] = 0

    @ti.kernel
    def grid_to_particle(self):
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
            self.velocity[p], self.C[p] = new_v, new_C
            self.position[p] += self.dt * new_v  # advection

    @ti.kernel
    def reset_fields(self):
        self.gravity[None] = self.initial_gravity
        for i in range(self.configuration.n_particles):
            self.position[i] = self.initial_position[i]
            self.velocity[i] = self.initial_velocity[i]
            self.F[i] = ti.Matrix([[1, 0], [0, 1]])
            self.C[i] = ti.Matrix.zero(float, 2, 2)
            self.Jp[i] = 1

    def initialize_simulation(self):
        configuration = self.configurations[self.configuration_id]
        self.initial_position.from_numpy(configuration.position)
        self.initial_velocity.from_numpy(configuration.velocity)

    def handle_events(self):
        if self.window.get_event(ti.ui.PRESS):
            if self.window.event.key == "r":
                self.reset_fields()
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

    def show_options(self):
        with self.gui.sub_window("Settings", 0.01, 0.01, 0.98, 0.98) as w:
            # Parameters
            self.E = w.slider_float(text="E", old_value=self.E, minimum=4.8e4, maximum=4.8e5)
            self.nu = w.slider_float(text="nu", old_value=self.nu, minimum=0, maximum=1)
            self.zeta = w.slider_float(text="zeta", old_value=self.zeta, minimum=0, maximum=20)
            self.theta_c = w.slider_float(text="theta_c", old_value=self.theta_c, minimum=0, maximum=5e-2)
            self.theta_s = w.slider_float(text="theta_s", old_value=self.theta_s, minimum=0, maximum=15e-3)
            self.rho_0 = w.slider_float(text="rho_0", old_value=self.rho_0, minimum=0, maximum=4e2)
            self.sticky = w.slider_float(text="sticky", old_value=self.sticky, minimum=0, maximum=1)
            # Presets
            prev_configuration_id = self.configuration_id
            for i in range(len(self.configurations)):
                name = self.configurations[i].name
                if w.checkbox(name, self.configuration_id == i):
                    self.configuration_id = i
            if self.configuration_id != prev_configuration_id:
                self.initialize_simulation()
                self.reset_fields()
                self.is_paused = True
            # Write to disk
            if self.write_to_disk:
                if w.button("Stop recording"):
                    self.write_to_disk = False
            else:
                if w.button("Start recording"):
                    self.write_to_disk = True
            # Reset
            if w.button("Reset"):
                self.reset_fields()
                self.is_paused = True
            # Pause/Unpause
            if self.is_paused:
                if w.button("Play"):
                    self.is_paused = False
            else:
                if w.button("Stop"):
                    self.is_paused = True

    def render(self):
        self.canvas.set_background_color((0.054, 0.06, 0.09))
        self.canvas.circles(centers=self.position, radius=0.0016, color=(0.8, 0.8, 0.8))
        if self.write_to_disk:
            self.window.save_image(f"{self.frame:06d}.png")
            self.frame += 1
        self.window.show()

    def run(self):
        self.initialize_simulation()
        self.reset_fields()
        while self.window.running:
            self.handle_events()
            self.substep()
            self.show_options()
            self.render()