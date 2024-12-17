"""
Testing a hydrodynamic model, roughly based on
'A two-way coupling method for simulating wave-induced breakup of ice floes based on SPH'
"""

from datetime import datetime
import taichi as ti
import numpy as np
import os

ICE_COLOR = [0.8, 0.8, 1]
WATER_COLOR = [0.4, 0.4, 1]


class Phase:
    Ice = 0
    Water = 1


@ti.data_oriented
class HydrodynamicModel:
    def __init__(
        self,
        quality: int,
        n_particles: int,
        initial_position: np.ndarray,
        initial_velocity: np.ndarray,
        initial_phase: np.ndarray,
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=5.5e-3,  # Critical stretch (7.5e-3)
        zeta=20,  # Hardening coefficient (10)
        E=3.8e5,  # Young's modulus (1.4e5)
        nu=0.33,  # Poisson's ratio (0.2)
    ):
        # MPM Parameters that are configuration independent
        self.quality = quality
        self.n_particles = n_particles
        self.n_grid = 128 * quality
        self.dx = 1 / self.n_grid
        self.inv_dx = float(self.n_grid)
        self.dt = 1e-4 / self.quality
        self.rho_0 = 4e2
        self.p_vol = (self.dx * 0.01) ** 2
        # self.p_mass = self.p_vol * self.rho_0

        # Parameters to control the simulation
        self.window = ti.ui.Window(name="MLS-MPM", res=(1080, 1080), fps_limit=60)
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

        # Create fields
        self.g_velo = ti.Vector.field(2, dtype=float, shape=(self.n_grid, self.n_grid))
        self.g_mass = ti.field(dtype=float, shape=(self.n_grid, self.n_grid))

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
        self.attractor_strength = ti.field(dtype=float, shape=())
        self.attractor_pos = ti.Vector.field(2, dtype=float, shape=())

        # Initialize fields
        self.lambda_0[None] = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu_0[None] = E / (2 * (1 + nu))
        self.gravity[None] = [0, -9.8]
        self.theta_c[None] = theta_c
        self.theta_s[None] = theta_s
        self.zeta[None] = zeta
        self.nu[None] = nu
        self.E[None] = E

        ### New parameters ====================================================
        self.p_initial_position = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.p_initial_velocity = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        # self.p_initial_phase = ti.field(dtype=float, shape=self.n_particles)
        self.p_phase = ti.field(dtype=float, shape=self.n_particles)
        self.p_mass = ti.field(dtype=float, shape=self.n_particles)

        self.p_initial_position.from_numpy(initial_position)
        self.p_initial_velocity.from_numpy(initial_velocity)
        self.p_phase.from_numpy(initial_phase)
        ### ===================================================================

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
            h = ti.max(0.1, ti.min(20, ti.exp(self.zeta[None] * (1.0 - self.JP[p]))))
            mu, la = self.mu_0[None] * h, self.lambda_0[None] * h
            if self.p_phase[p] == Phase.Water:
                mu = 0
            U, sigma, V = ti.svd(self.F[p])
            J = 1.0
            for d in ti.static(range(2)):
                # Clamp singular values to [1 - theta_c, 1 + theta_s]
                singular_value = float(sigma[d, d])
                singular_value = max(singular_value, 1 - self.theta_c[None])
                singular_value = min(singular_value, 1 + self.theta_s[None])
                self.JP[p] *= sigma[d, d] / singular_value
                sigma[d, d] = singular_value
                J *= singular_value
            # Deciding on the phase we reset or reconstruct the deformation gradient
            if self.p_phase[p] == Phase.Water:
                # Reset deformation gradient to avoid numerical instability
                self.F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
            elif self.p_phase[p] == Phase.Ice:  # ICE
                # Reconstruct elastic deformation gradient after plasticity
                self.F[p] = U @ sigma @ V.transpose()
            ### ================================================================
            stress = 2 * mu * (self.F[p] - U @ V.transpose()) @ self.F[p].transpose()
            stress += ti.Matrix.identity(float, 2) * la * J * (J - 1)
            stress *= -self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx
            affine = stress + self.p_mass[p] * self.C[p]
            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1]
                v = self.p_mass[p] * self.p_velocity[p] + affine @ dpos
                self.g_velo[base + offset] += weight * v
                self.g_mass[base + offset] += weight * self.p_mass[p]

    @ti.kernel
    def momentum_to_velocity(self):
        for i, j in self.g_mass:
            if self.g_mass[i, j] > 0:  # No need for epsilon here
                self.g_velo[i, j] = (1 / self.g_mass[i, j]) * self.g_velo[i, j]
                self.g_velo[i, j] += self.dt * self.gravity[None]  # gravity

                dist = self.attractor_pos[None] - self.dx * ti.Vector([i, j])
                self.g_velo[i, j] += dist / (0.01 + dist.norm()) * self.attractor_strength[None] * self.dt * 100

                vertical_collision = i < 3 and self.g_velo[i, j][0] < 0
                vertical_collision |= i > (self.n_grid - 3) and self.g_velo[i, j][0] > 0
                horizontal_collision = j < 3 and self.g_velo[i, j][1] < 0
                horizontal_collision |= j > (self.n_grid - 3) and self.g_velo[i, j][1] > 0
                if vertical_collision:
                    self.g_velo[i, j][0] = 0
                if horizontal_collision:
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
    def reset(self):
        for i in range(self.n_particles):
            self.p_position[i] = self.p_initial_position[i]
            # material[i] = i // group_size  # 0: fluid 1: jelly 2: snow
            self.F[i] = ti.Matrix([[1, 0], [0, 1]])
            self.C[i] = ti.Matrix.zero(float, 2, 2)
            ### New parameters =================================================
            # self.p_phase[i] = self.p_initial_phase[i]
            ### ================================================================
            self.p_color[i] = WATER_COLOR if self.p_phase[i] == Phase.Water else ICE_COLOR

            # TODO: Make ice a bit less dense???
            # TODO: Research the correct density for ice
            rho = self.rho_0 if self.p_phase[i] == Phase.Water else self.rho_0 * 0.15
            self.p_mass[i] = self.p_vol * rho

            self.p_velocity[i] = self.p_initial_velocity[i]
            self.JP[i] = 1

    def handle_events(self):
        self.attractor_strength[None] = 0
        if self.window.get_event(ti.ui.PRESS):
            if self.window.event.key == "r":
                self.reset()
            elif self.window.event.key in [ti.GUI.BACKSPACE, "s"]:
                self.should_write_to_disk = not self.should_write_to_disk
            elif self.window.event.key in [ti.GUI.SPACE, "p"]:
                self.is_paused = not self.is_paused
            elif self.window.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                self.window.running = False  # Stop the simulation
            elif self.window.event.key in [ti.GUI.LMB, ti.GUI.RMB]:
                mouse = self.window.get_cursor_pos()
                # self.gui.circle((mouse[0], mouse[1]), color=0x336699, radius=15)
                self.attractor_pos[None] = [mouse[0], mouse[1]]
                if self.window.is_pressed(ti.GUI.LMB):
                    self.attractor_strength[None] = 1
                if self.window.is_pressed(ti.GUI.RMB):
                    self.attractor_strength[None] = -1

    def substep(self):
        if not self.is_paused:
            for _ in range(int(2e-3 // self.dt)):
                self.reset_grids()
                self.particle_to_grid()
                self.momentum_to_velocity()
                self.grid_to_particle()

    def show_parameters(self, subwindow):
        self.theta_c[None] = subwindow.slider_float("theta_c", self.theta_c[None], 1e-2, 3.5e-2)
        self.theta_s[None] = subwindow.slider_float("theta_s", self.theta_s[None], 5.0e-3, 10e-3)
        self.zeta[None] = subwindow.slider_int("zeta", self.zeta[None], 3, 20)
        self.nu[None] = subwindow.slider_float("nu", self.nu[None], 0.1, 0.4)
        self.E[None] = subwindow.slider_float("E", self.E[None], 4.8e4, 4.8e5)
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
            self.show_buttons(subwindow)

    def render(self):
        self.canvas.set_background_color((0.054, 0.06, 0.09))
        self.canvas.circles(centers=self.p_position, radius=0.0015, per_vertex_color=self.p_color)
        if self.should_write_to_disk and not self.is_paused and not self.is_showing_settings:
            self.window.save_image(f".output/{self.directory}/{self.frame:06d}.png")
            self.frame += 1
        self.window.show()

    def run(self):
        self.reset()
        while self.window.running:
            self.handle_events()
            self.show_settings()
            self.substep()
            self.render()


def main():
    ti.init(arch=ti.gpu)

    quality = 3
    n_particles = 3_000 * (quality**2)

    print("-" * 150)
    print("[Hint] Press R to [R]eset, P|SPACE to [P]ause/un[P]ause and S|BACKSPACE to [S]tart recording!")
    print("-" * 150)

    def create_square(positions=[[0.5, 0.5]], heights=[0.5], widths=[0.5]):
        n_snowballs = len(positions)
        group_size = n_particles // n_snowballs
        p = np.zeros(shape=(n_particles, 2), dtype=np.float32)
        for i in range(n_particles):
            j = i // group_size
            p[i, 0] = (np.random.rand() * widths[j]) + positions[j][0]
            p[i, 1] = (np.random.rand() * heights[j]) + positions[j][1]
        return p

    def property_to_object(to_map=[[0, 0]]):
        n_objects = len(to_map)
        group_size = n_particles // n_objects
        if isinstance(to_map[0], list):
            m = np.zeros(shape=(n_particles, len(to_map[0])), dtype=np.float32)
        else:
            m = np.zeros(shape=n_particles, dtype=np.float32)
        for i in range(n_particles):
            j = i // group_size
            m[i] = to_map[j]
        return m

    simulation = HydrodynamicModel(
        quality=quality,
        n_particles=n_particles,
        initial_position=create_square(
            positions=[[0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.001, 0.001], [0.33, 0.05]],
            heights=[0.05, 0.05, 0.05, 0.05, 0.01],
            widths=[0.98, 0.98, 0.98, 0.98, 0.33],
        ),
        initial_velocity=property_to_object([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
        initial_phase=property_to_object([Phase.Water, Phase.Water, Phase.Water, Phase.Water, Phase.Ice]),
    )
    simulation.run()


if __name__ == "__main__":
    main()
