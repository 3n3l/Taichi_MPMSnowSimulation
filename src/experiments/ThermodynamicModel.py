from datetime import datetime
from numpy import dtype, shape
import taichi.math as tm
import taichi as ti
import os

ICE_COLOR = [0.8, 0.8, 1]
WATER_COLOR = [0.4, 0.4, 1]


class Phase:
    Ice = 0
    Water = 1


@ti.data_oriented
class ThermodynamicModel:
    def __init__(
        self,
        quality: int,
        n_particles: int,
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        zeta=10,  # Hardening coefficient (10)
        E=1.4e5,  # Young's modulus (1.4e5)
        nu=0.2,  # Poisson's ratio (0.2)
    ):
        # MPM Parameters that are configuration independent
        self.quality = quality
        self.n_particles = n_particles
        self.n_grid = 128 * quality
        self.dx = 1 / self.n_grid
        self.inv_dx = float(self.n_grid)
        self.dt = 1e-4 / self.quality
        self.rho_0 = 4e2
        self.p_vol = (self.dx * 0.25) ** 2
        self.p_mass = self.p_vol * self.rho_0

        ### New parameters ====================================================
        # self.should_show_grid = True
        # self.should_show_cells = False

        # self.nx = 719 # Grid width
        # self.dy = self.dx
        # self.g_position = ti.Vector.field(2, dtype=float, shape=(self.n_grid**2))

        # Number of dimensions
        self.d = 2

        # MAC cells
        self.x_faces = ti.field(dtype=float, shape=(self.n_grid + 1, self.n_grid))
        self.y_faces = ti.field(dtype=float, shape=(self.n_grid, self.n_grid + 1))
        self.pressure = ti.field(dtype=float, shape=(self.n_grid, self.n_grid))

        # Track the phase, this will either be Phase.Ice or Phase.Water
        self.p_phase = ti.field(dtype=float, shape=self.n_particles)

        # Track elastic and plastic parts of the deformation gradient
        # NOTE: This might not be needed, as all of the computations might be done in p2g
        self.FE = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles)
        self.FP = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles)
        ### ===================================================================

        # Parameters to control the simulation
        # self.window = ti.ui.Window(name="MLS-MPM", res=((self.nx + 1), (self.ny + 1)), fps_limit=60)
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

        # Initialize fields
        self.lambda_0[None] = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu_0[None] = E / (2 * (1 + nu))
        self.gravity[None] = [0, -9.8]
        self.theta_c[None] = theta_c
        self.theta_s[None] = theta_s
        self.zeta[None] = zeta
        self.nu[None] = nu
        self.E[None] = E

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
            w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1) ** 2,
                0.5 * (fx - 0.5) ** 2,
            ]
            # Deformation gradient update
            self.F[p] = (ti.Matrix.identity(float, 2) + self.dt * self.C[p]) @ self.F[p]
            # Apply snow hardening by adjusting Lame parameters
            h = ti.max(0.1, ti.min(5, ti.exp(self.zeta[None] * (1.0 - self.JP[p]))))
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
            ### New ============================================================
            FE = U @ sigma @ V.transpose()
            FP = tm.inverse(self.F[p]) @ FE

            if self.p_phase[p] == Phase.Water:  # Apply correction for dilational/deviatoric stresses
                FE *= J ** (1 / self.d)
                FP *= J ** -(1 / self.d)
                self.F[p] = FE @ FP
            elif self.p_phase[p] == Phase.Ice:  # Reconstruct elastic deformation gradient after plasticity
                self.F[p] = U @ sigma @ V.transpose()
            ### ================================================================
            stress = 2 * mu * (self.F[p] - U @ V.transpose()) @ self.F[p].transpose()
            stress += ti.Matrix.identity(float, 2) * la * J * (J - 1)
            stress *= -self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx
            affine = stress + self.p_mass * self.C[p]
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
            # TODO: set the color somewhere here after computing temperature
            self.p_color[p] = WATER_COLOR

    @ti.kernel
    def reset(self):
        for i in range(self.n_particles):
            self.p_position[i] = [(ti.random() * 0.1) + 0.45, (ti.random() * 0.1) + 0.001]
            # material[i] = i // group_size  # 0: fluid 1: jelly 2: snow
            self.F[i] = ti.Matrix([[1, 0], [0, 1]])
            self.C[i] = ti.Matrix.zero(float, 2, 2)
            ### New parameters =================================================
            self.p_phase[i] = Phase.Water
            self.FE[i] = ti.Matrix([[1, 0], [0, 1]])
            self.FP[i] = ti.Matrix([[1, 0], [0, 1]])
            ### ================================================================
            self.p_color[i] = WATER_COLOR
            self.p_velocity[i] = [0, 0]
            self.JP[i] = 1

    def handle_events(self):
        if self.window.get_event(ti.ui.PRESS):
            if self.window.event.key == "r":
                self.reset()
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

    def show_parameters(self, subwindow):
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
        # if subwindow.button(" Toggle Grid     "):
        # self.should_show_grid = not self.should_show_grid
        # if subwindow.button(" Toggle Cells    "):
        # self.should_show_cells = not self.should_show_cells

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
        self.canvas.circles(centers=self.p_position, radius=0.001, per_vertex_color=self.p_color)
        if self.should_write_to_disk and not self.is_paused and not self.is_showing_settings:
            self.window.save_image(f".output/{self.directory}/{self.frame:06d}.png")
            self.frame += 1
        self.window.show()

    def run(self):
        # self.init()
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

    simulation = ThermodynamicModel(quality=quality, n_particles=n_particles)
    simulation.run()


if __name__ == "__main__":
    main()