import taichi as ti
import numpy as np


@ti.data_oriented
class MPM:
    def __init__(
        self,
        E = 1.4e5,                  # Young's modulus (1.4e5)
        nu = 0.2,                   # Poisson's ratio (0.2)
        zeta = 10,                  # Hardening coefficient (10)
        theta_c = 2.5e-2,           # Critical compression (2.5e-2)
        theta_s = 7.5e-3,           # Critical stretch (7.5e-3)
        rho_0 = 4e2 ,               # Initial density (4e2)
        sticky = 0.5,               # The lower, the stickier the border
        quality = 1,                # Use a larger value for higher-res simulations
        initial_gravity = [0, 0],   # Gravity of the simulation ([0, 0])
        attractor_active = False,   # Enables mouse controlled attractor (False)
        initial_velocities = np.array([[0, 0]], dtype=np.float32),
        initial_positions = np.array([[0, 0]], dtype=np.float32),
        initial_radii = np.array([0.5], dtype=np.float32),
    ):
        # Parameters starting points for MPM
        self.E = E
        self.nu = nu
        self.zeta = zeta
        self.theta_c = theta_c
        self.theta_s = theta_s
        self.rho_0 = rho_0
        self.mu_0 = E / (2 * (1 + nu))                      # Lame parameters
        self.lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
 

        # Parameters to control the simulation
        self.quality = quality
        self.n_particles = 1_000 * (quality ** 2)
        self.n_grid = 128 * quality
        self.dx = 1 / self.n_grid
        self.inv_dx = float(self.n_grid)
        self.dt = 1e-4 / self.quality
        self.p_vol = (self.dx * 0.5) ** 2
        self.p_mass = self.p_vol * rho_0
        self.sticky = sticky
        self.initial_gravity = initial_gravity
        self.attractor_is_active = attractor_active
        self.group_size = self.n_particles // initial_radii.shape[0]
        self.thetas = ti.field(dtype=float, shape=self.group_size)  # used to parametrize the snowball
        self.thetas.from_numpy(np.linspace(0, 2 * np.pi, self.group_size + 2, dtype=np.float32)[1:-1])
        self.initial_velocities = ti.Vector.field(n=2, dtype=float, shape=initial_velocities.shape[0])
        self.initial_positions = ti.Vector.field(n=2, dtype=float, shape=initial_positions.shape[0])
        self.initial_radii = ti.field(dtype=float, shape=initial_radii.shape[0])
        self.initial_velocities.from_numpy(initial_velocities)
        self.initial_positions.from_numpy(initial_positions)
        self.initial_radii.from_numpy(initial_radii)

        # Fields
        self.position = ti.Vector.field(2, dtype=float, shape=self.n_particles)             # position
        self.velocity = ti.Vector.field(2, dtype=float, shape=self.n_particles)             # velocity
        self.C = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles)                 # affine velocity field
        self.F = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles)                 # deformation gradient
        self.Jp = ti.field(dtype=float, shape=self.n_particles)                             # plastic deformation
        self.grid_velo = ti.Vector.field(2, dtype=float, shape=(self.n_grid, self.n_grid))  # grid node momentum/velocity
        self.grid_mass = ti.field(dtype=float, shape=(self.n_grid, self.n_grid))            # grid node mass
        self.gravity = ti.Vector.field(2, dtype=float, shape=())
        self.attractor_strength = ti.field(dtype=float, shape=())
        self.attractor_pos = ti.Vector.field(2, dtype=float, shape=())


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
            stress = 2 * mu * (self.F[p] - U @ V.transpose()) @ self.F[p].transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1)
            stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * stress
            affine = stress + self.p_mass * self.C[p]
            for i, j in ti.static(ti.ndrange(3, 3)):
                # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1]
                self.grid_velo[base + offset] += weight * (self.p_mass * self.velocity[p] + affine @ dpos)
                self.grid_mass[base + offset] += weight * self.p_mass

    @ti.kernel
    def momentum_to_velocity(self):
        for i, j in self.grid_mass:
            if self.grid_mass[i, j] > 0:  # No need for epsilon here
                self.grid_velo[i, j] = (1 / self.grid_mass[i, j]) * self.grid_velo[i, j]
                self.grid_velo[i, j] += self.dt * self.gravity[None]  # gravity
                dist = self.attractor_pos[None] - self.dx * ti.Vector([i, j])
                self.grid_velo[i, j] += dist / (0.01 + dist.norm()) * self.attractor_strength[None] * self.dt * 100
                # Boundary conditions for the grid velocities
                collision_left = i < 3 and self.grid_velo[i, j][0] < 0
                collision_right = i > (self.n_grid - 3) and self.grid_velo[i, j][0] > 0
                if collision_left or collision_right:
                    self.grid_velo[i, j][1] = 0
                    self.grid_velo[i, j][0] *= self.sticky
                collision_top = j < 3 and self.grid_velo[i, j][1] < 0
                collision_bottom = j > (self.n_grid - 3) and self.grid_velo[i, j][1] > 0
                if collision_top or collision_bottom:
                    self.grid_velo[i, j][1] *= self.sticky
                    self.grid_velo[i, j][0] = 0
                # ^ this should be the other way round, but works better this way?!


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
    def reset(self):
        self.gravity[None] = self.initial_gravity
        for i in range(self.n_particles):
            index = i // self.group_size
            position = self.initial_positions[index]
            radius = self.initial_radii[index] * ti.sqrt(ti.random())
            x = (radius * (ti.sin(self.thetas[i % self.group_size]))) + position[0]
            y = (radius * (ti.cos(self.thetas[i % self.group_size]))) + position[1]
            self.position[i] = [x, y]
            self.velocity[i] = self.initial_velocities[index]
            self.F[i] = ti.Matrix([[1, 0], [0, 1]])
            self.C[i] = ti.Matrix.zero(float, 2, 2)
            self.Jp[i] = 1


    def run(self):
        # print("[Hint] Use left/right mouse buttons to attract/repel and start the simulation. Press R to reset.")
        gui = ti.GUI("Snowball MLS-MPM", res=512, background_color=0x0E1018)
        self.reset()

        for _ in range(20_000):
            if gui.get_event(ti.GUI.PRESS):
                if gui.event.key == "r":
                    self.reset()
                elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                    break

            if self.attractor_is_active:
                # Control attractor
                mouse = gui.get_cursor_pos()
                self.attractor_strength[None] = 0
                if gui.is_pressed(ti.GUI.LMB):
                    gui.circle((mouse[0], mouse[1]), color=0x336699, radius=10)
                    self.attractor_pos[None] = [mouse[0], mouse[1]]
                    self.gravity[None] = self.initial_gravity
                    self.attractor_strength[None] = 1
                if gui.is_pressed(ti.GUI.RMB):
                    gui.circle((mouse[0], mouse[1]), color=0x336699, radius=10)
                    self.attractor_pos[None] = [mouse[0], mouse[1]]
                    self.gravity[None] = self.initial_gravity
                    self.attractor_strength[None] = -1

            for _ in range(int(2e-3 // self.dt)):
                self.reset_grids()
                self.particle_to_grid()
                self.momentum_to_velocity()
                self.grid_to_particle()

            gui.circles(self.position.to_numpy(), radius=1)
            gui.show() # change to gui.show(f'{frame:06d}.png') to write images to disk
