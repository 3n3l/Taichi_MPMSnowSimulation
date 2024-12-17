from datetime import datetime
from taichi.lang.matrix_ops import Vector, determinant
import taichi.math as tm
import taichi as ti
import os

WATER_CONDUCTIVITY = 0.55  # Water: 0.55, Ice: 2.33
ICE_CONDUCTIVITY = 2.33
WATER_HEAT_CAPACITY = 4.186  # j/dC
ICE_HEAT_CAPACITY = 2.093  # j/dC
LATEN_HEAT = 0.334  # J/kg


class Classification:
    Empty = 0
    Colliding = 1
    Interior = 2


class Phase:
    Ice = 0
    Water = 1


class Color:
    Ice = [0.8, 0.8, 1]
    Water = [0.4, 0.4, 1]


@ti.data_oriented
class ThermodynamicModel:
    def __init__(
        self,
        quality: int,
        n_particles: int,
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=7.5e-3,  # Critical stretch (7.5e-3)
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
        ### New parameters ====================================================
        ### ===================================================================
        self.dt = 1e-4 / self.quality
        self.rho_0 = 4e2
        self.p_vol = (self.dx * 0.5) ** 2

        ### New parameters ====================================================
        # Number of dimensions
        self.n_dimensions = 2

        # Parameters to control melting/freezing
        # TODO: these are variables and need to be put into fields
        # TODO: these depend not only on phase, but also on temperature,
        #       so ideally they are functions of these two variables
        # self.heat_conductivity = 0.55 # Water: 0.55, Ice: 2.33
        # self.heat_capacity = 4.186 # Water: 4.186, Ice: 2.093 (j/dC)
        # self.latent_heat = 0.334 # in J/kg

        # Properties on MAC-faces.
        self.face_mass_x = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid))
        self.face_mass_y = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid + 1))
        self.face_velocity_x = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid))
        self.face_velocity_y = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid + 1))
        self.face_conductivity_x = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid))
        self.face_conductivity_y = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid + 1))

        # Properties on MAC-cells.
        self.cell_mass = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.cell_pressure = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid + 1))
        self.cell_capacity = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.cell_inv_lambda = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.cell_temperature = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.cell_classification = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))

        # Properties on particles.
        self.p_mass = ti.field(dtype=ti.float32, shape=self.n_particles)
        self.p_phase = ti.field(dtype=ti.float32, shape=self.n_particles)
        self.p_capacity = ti.field(dtype=ti.float32, shape=self.n_particles)
        self.p_inv_lambda = ti.field(dtype=ti.float32, shape=self.n_particles)
        self.p_temperature = ti.field(dtype=ti.float32, shape=self.n_particles)
        self.p_conductivity = ti.field(dtype=ti.float32, shape=self.n_particles)
        self.cp_x = ti.Vector.field(2, dtype=ti.float32, shape=self.n_particles)
        self.cp_y = ti.Vector.field(2, dtype=ti.float32, shape=self.n_particles)

        # Track elastic and plastic parts of the deformation gradient
        # NOTE: This might not be needed, as all of the computations might be done in p2g
        # self.FE = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles)
        # self.FP = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles)
        ### ===================================================================

        # Parameters to control the simulation
        # self.window = ti.ui.Window(name="MLS-MPM", res=((self.nx + 1), (self.ny + 1)), fps_limit=60)
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

        # Create fields
        self.p_position = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.p_velocity = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.p_color = ti.Vector.field(3, dtype=float, shape=self.n_particles)
        self.C = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles)  # affine velocity field
        self.F = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles)  # deformation gradient
        # TODO: not all of the Js need to be saved
        self.J = ti.field(dtype=float, shape=self.n_particles)
        self.JP = ti.field(dtype=float, shape=self.n_particles)
        self.JE = ti.field(dtype=float, shape=self.n_particles)
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
        for i, j in self.face_mass_x:
            self.face_velocity_x[i, j] = 0
            self.face_mass_x[i, j] = 0
        for i, j in self.face_mass_y:
            self.face_velocity_y[i, j] = 0
            self.face_mass_y[i, j] = 0
        for i, j in self.cell_mass:
            self.cell_classification[i, j] = Classification.Empty
            self.cell_mass[i, j] = 0

    @ti.kernel
    def particle_to_grid(self):
        for p in self.p_position:
            # Deformation gradient update
            self.F[p] = (ti.Matrix.identity(float, 2) + self.dt * self.C[p]) @ self.F[p]
            # Apply snow hardening by adjusting Lame parameters
            h = ti.max(0.1, ti.min(5, ti.exp(self.zeta[None] * (1.0 - self.JP[p]))))
            mu, la = self.mu_0[None] * h, self.lambda_0[None] * h
            if self.p_phase[p] == Phase.Water:
                mu = 0
            U, sigma, V = ti.svd(self.F[p])
            J = 1.0
            for d in ti.static(range(self.n_dimensions)):  # Clamp singular values to [1 - theta_c, 1 + theta_s]
                singular_value = float(sigma[d, d])
                singular_value = max(singular_value, 1 - self.theta_c[None])
                singular_value = min(singular_value, 1 + self.theta_s[None])
                self.J[p] *= sigma[d, d] / singular_value
                sigma[d, d] = singular_value
                J *= singular_value

            ### New ============================================================
            FE = U @ sigma @ V.transpose()
            FP = tm.inverse(self.F[p]) @ FE
            if self.p_phase[p] == Phase.Water:  # Apply correction for dilational/deviatoric stresses
                FE *= J ** (1 / self.n_dimensions)
                FP *= J ** -(1 / self.n_dimensions)
                self.F[p] = FE @ FP
            elif self.p_phase[p] == Phase.Ice:  # Reconstruct elastic deformation gradient after plasticity
                self.F[p] = FE

            self.JE[p] = determinant(FE)
            self.JP[p] = determinant(FP)

            # Compute pressure.
            # self.cell_temperature[c_base + offset] += c_weight * self.p_temperature[p]
            # pressure = (-1 / self.JP[p]) * self.lambda_0 * (self.JE[p] - 1)
            ### ================================================================

            # Compute Piola-Kirchhoff stress.
            # TODO: This might only need the elastic part? Or something else?
            stress = 2 * mu * (self.F[p] - U @ V.transpose()) @ self.F[p].transpose()
            stress += ti.Matrix.identity(float, 2) * la * J * (J - 1)
            stress *= -self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx
            x_affine = (stress + self.p_mass[p] * self.C[p]) @ ti.Vector([1, 0])
            y_affine = (stress + self.p_mass[p] * self.C[p]) @ ti.Vector([0, 1])

            # NOTE: Quadratic kernels (JST16, Eqn. 123, with x=fx, fx-1,fx-2)
            # TODO: this might need cubic weights from the augmented mpm paper
            c_stagger = ti.Vector([0.5, 0.5])  # -(0.0, 0.0) + (0.5, 0.5)
            x_stagger = ti.Vector([0.5, 1.0])  # -(0.5, 0.0) + (0.5, 0.5)
            y_stagger = ti.Vector([1.0, 0.5])  # -(0.0, 0.5) + (0.5, 0.5)
            c_base = (self.p_position[p] * self.inv_dx - c_stagger).cast(int)  # pyright: ignore
            x_base = (self.p_position[p] * self.inv_dx - x_stagger).cast(int)  # pyright: ignore
            y_base = (self.p_position[p] * self.inv_dx - y_stagger).cast(int)  # pyright: ignore
            c_fx = self.p_position[p] * self.inv_dx - c_base.cast(float)
            x_fx = self.p_position[p] * self.inv_dx - (x_base.cast(float) + ti.Vector([0, 0.5]))
            y_fx = self.p_position[p] * self.inv_dx - (y_base.cast(float) + ti.Vector([0.5, 0]))
            c_w = [0.5 * (1.5 - c_fx) ** 2, 0.75 - (c_fx - 1) ** 2, 0.5 * (c_fx - 0.5) ** 2]
            x_w = [0.5 * (1.5 - x_fx) ** 2, 0.75 - (x_fx - 1) ** 2, 0.5 * (x_fx - 0.5) ** 2]
            y_w = [0.5 * (1.5 - y_fx) ** 2, 0.75 - (y_fx - 1) ** 2, 0.5 * (y_fx - 0.5) ** 2]

            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                c_weight = c_w[i][0] * c_w[j][1]
                x_weight = x_w[i][0] * x_w[j][1]
                y_weight = y_w[i][0] * y_w[j][1]
                # c_dpos = (offset.cast(float) - c_fx) * self.dx
                x_dpos = (offset.cast(float) - x_fx) * self.dx
                y_dpos = (offset.cast(float) - y_fx) * self.dx

                # print("=" * 200)
                # print("position -> ", self.p_position[p])
                # print("x_base   -> ", x_base)
                # print("x_fx     -> ", x_fx)
                # print("x_weight -> ", x_weight)
                # print("offset   -> ", offset)
                # print("b + o    -> ", x_base + offset)
                # print()

                # Rasterize mass to grid faces.
                self.face_mass_x[x_base + offset] += x_weight * self.p_mass[p]
                self.face_mass_y[y_base + offset] += y_weight * self.p_mass[p]

                # Rasterize velocity to grid faces.
                # x_velocity = self.p_mass[p] * (self.p_velocity[p][0] + self.cp_x[p] @ x_dpos)
                # y_velocity = self.p_mass[p] * (self.p_velocity[p][1] + self.cp_y[p] @ y_dpos)
                x_velocity = self.p_mass[p] * (self.p_velocity[p][0] + x_affine @ x_dpos)
                y_velocity = self.p_mass[p] * (self.p_velocity[p][1] + y_affine @ y_dpos)
                self.face_velocity_x[x_base + offset] += x_weight * x_velocity  # + x_stress
                self.face_velocity_y[y_base + offset] += y_weight * y_velocity  # + y_stress

                # Rasterize conductivity to grid faces.
                self.face_conductivity_x[x_base + offset] += x_weight * self.p_mass[p] * self.p_conductivity[p]
                self.face_conductivity_y[y_base + offset] += y_weight * self.p_mass[p] * self.p_conductivity[p]

                # Rasterize to cell centers.
                # TODO: JE, JP, J and others might need to be computed here???
                self.cell_mass[c_base + offset] += c_weight * self.p_mass[p]
                self.cell_capacity[c_base + offset] += c_weight * self.p_capacity[p]
                self.cell_temperature[c_base + offset] += c_weight * self.p_temperature[p]
                self.cell_inv_lambda[c_base + offset] += c_weight * self.p_inv_lambda[p]

    @ti.kernel
    def classify_cells(self):
        # We can extract the offset coordinates from the faces by adding one to the respective axis,
        # e.g. we get the two x-faces with [i, j] and [i + 1, j], where each cell looks like:
        # -  ^  -
        # >  *  >
        # -  ^  -
        for i, j in self.cell_classification:
            # TODO: A cell is marked as colliding if all of its surrounding faces are colliding.
            # TODO: A MAC face is colliding if the level set computed by any collision object is negative at the face center.
            # TODO: Colliding cells are either assigned the temperature of the object it collides with or a user-defined
            # spatially-varying value depending on the setup. If the free surface is being enforced as a Dirichlet
            # temperature condition, the ambient air temperature is recorded for empty cells. No other cells
            # require temperatures to be recorded at this stage.
            is_colliding = False

            # A cell is interior if the cell and all of its surrounding faces have mass.
            is_interior = self.face_mass_x[i, j] > 0
            is_interior &= self.face_mass_y[i, j] > 0
            is_interior &= self.face_mass_x[i + 1, j] > 0
            is_interior &= self.face_mass_y[i, j + 1] > 0

            if is_colliding:
                self.cell_classification[i, j] = Classification.Colliding
            elif is_interior:
                self.cell_classification[i, j] = Classification.Interior
            else:
                self.cell_classification[i, j] = Classification.Empty

    @ti.kernel
    def momentum_to_velocity(self):
        for i, j in self.face_mass_x:
            if self.face_mass_x[i, j] > 0:  # No need for epsilon here
                self.face_velocity_x[i, j] *= 1 / self.face_mass_x[i, j]
                self.face_velocity_x[i, j] += self.dt * self.gravity[None][0]
                collision_left = i < 3 and self.face_velocity_x[i, j] < 0
                collision_right = i > (self.n_grid - 3) and self.face_velocity_x[i, j] > 0
                if collision_left or collision_right:
                    self.face_velocity_x[i, j] = 0
        for i, j in self.face_mass_y:
            if self.face_mass_y[i, j] > 0:  # No need for epsilon here
                self.face_velocity_y[i, j] *= 1 / self.face_mass_y[i, j]
                self.face_velocity_y[i, j] += self.dt * self.gravity[None][1]
                collision_top = j > (self.n_grid - 3) and self.face_velocity_y[i, j] > 0
                collision_bottom = j < 3 and self.face_velocity_y[i, j] < 0
                if collision_top or collision_bottom:
                    self.face_velocity_y[i, j] = 0
        for i, j in self.cell_mass:
            if self.cell_mass[i, j] > 0:  # No need for epsilon here
                self.cell_mass[i, j] *= 1 / self.cell_mass[i, j]
                self.cell_capacity[i, j] *= 1 / self.cell_mass[i, j]
                self.cell_temperature[i, j] *= 1 / self.cell_mass[i, j]
                self.cell_inv_lambda[i, j] *= 1 / self.cell_mass[i, j]

    @ti.kernel
    def N(x):
        x = ti.abs(x)

    @ti.kernel
    def grid_to_particle(self):
        for p in self.p_position:
            x_stagger = ti.Vector([0.5, 1.0])  # -(0.5, 0.0) + (0.5, 0.5)
            y_stagger = ti.Vector([1.0, 0.5])  # -(0.0, 0.5) + (0.5, 0.5)
            x_base = (self.p_position[p] * self.inv_dx - x_stagger).cast(int)  # pyright: ignore
            y_base = (self.p_position[p] * self.inv_dx - y_stagger).cast(int)  # pyright: ignore
            x_fx = self.p_position[p] * self.inv_dx - (x_base.cast(float) + ti.Vector([0, 0.5]))
            y_fx = self.p_position[p] * self.inv_dx - (y_base.cast(float) + ti.Vector([0.5, 0]))
            x_w = [0.5 * (1.5 - x_fx) ** 2, 0.75 - (x_fx - 1) ** 2, 0.5 * (x_fx - 0.5) ** 2]
            y_w = [0.5 * (1.5 - y_fx) ** 2, 0.75 - (y_fx - 1) ** 2, 0.5 * (y_fx - 0.5) ** 2]

            # TODO: the cell values must be transferred back to the particles?!

            bx = ti.Vector.zero(float, 2)
            by = ti.Vector.zero(float, 2)
            nv = ti.Vector.zero(float, 2)

            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                x_weight = x_w[i][0] * x_w[j][1]
                y_weight = y_w[i][0] * y_w[j][1]
                x_dpos = offset.cast(float) - x_fx
                y_dpos = offset.cast(float) - y_fx
                x_velocity = x_weight * self.face_velocity_x[x_base + offset]
                y_velocity = y_weight * self.face_velocity_y[y_base + offset]
                nv += ti.Vector([x_velocity, y_velocity])
                bx += x_weight * x_velocity * x_dpos
                by += y_weight * y_velocity * y_dpos

            cx = 4 * self.inv_dx * self.inv_dx * bx # C = B @ (D^(-1))
            cy = 4 * self.inv_dx * self.inv_dx * by # C = B @ (D^(-1))
            self.cp_x[p], self.cp_y[p] = cx, cy
            self.C[p] = ti.Matrix([[cx[0], cy[0]], [cx[1], cy[1]]])  # pyright: ignore

            self.p_velocity[p] = nv
            self.p_position[p] += self.dt * nv
            self.p_color[p] = Color.Water if self.p_phase[p] == Phase.Water else Color.Ice
            # TODO: set the color somewhere after setting the new phase

    @ti.kernel
    def reset(self):
        for i in range(self.n_particles):
            # self.p_position[i] = [(ti.random() * 0.1) + 0.45, (ti.random() * 0.1) + 0.001]
            self.p_position[i] = [(ti.random() * 0.1) + 0.45, (ti.random() * 0.1) + 0.25]
            # material[i] = i // group_size  # 0: fluid 1: jelly 2: snow
            self.F[i] = ti.Matrix([[1, 0], [0, 1]])
            self.C[i] = ti.Matrix.zero(float, 2, 2)
            ### New ============================================================
            self.p_phase[i] = Phase.Ice
            self.p_color[i] = Color.Ice
            self.p_mass[i] = self.p_vol * self.rho_0  # TODO: ???
            self.cp_x[i] = [0, 0]
            self.cp_y[i] = [0, 0]
            ### ================================================================
            self.p_velocity[i] = [0, 0]
            self.JP[i] = 1
            self.JE[i] = 1
            self.J[i] = 1

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
                self.classify_cells()
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
    # ti.init(arch=ti.cpu, debug=True)
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
