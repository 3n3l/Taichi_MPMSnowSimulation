# x_gradient = ti.Vector([x_g[i][0] * x_w[j][1], x_w[i][0] * x_g[j][1]])
# y_gradient = ti.Vector([y_g[i][0] * y_w[j][1], y_w[i][0] * y_g[j][1]])

# x_base = (self.p_position[p] * self.inv_dx - ti.Vector([1.0, 0.5])).cast(int)
# y_base = (self.p_position[p] * self.inv_dx - ti.Vector([0.5, 1.0])).cast(int)

# x_fx = self.p_position[p] * self.inv_dx - (x_base.cast(float) + ti.Vector([0, 0.5]))
# y_fx = self.p_position[p] * self.inv_dx - (y_base.cast(float) + ti.Vector([0.5, 0]))

# x_g = [x_fx - 1.5, (-2) * (x_fx - 1), x_fx - 3.5]
# y_g = [y_fx - 1.5, (-2) * (y_fx - 1), y_fx - 3.5]

# n_C = ti.Matrix.zero(float, 2, 2)
# D_x = ti.Matrix.zero(float, 2, 2)
# D_y = ti.Matrix.zero(float, 2, 2)

# x_gradient = ti.Vector([x_g[i][0] * x_w[j][1], x_w[i][0] * x_g[j][1]])
# y_gradient = ti.Vector([y_g[i][0] * y_w[j][1], y_w[i][0] * y_g[j][1]])

# self.cp_x[p] = ti.Vector([n_C[0, 0], n_C[1, 0]])
# self.cp_y[p] = ti.Vector([n_C[0, 1], n_C[1, 1]])
# cp_x = 4 * self.inv_dx * self.inv_dx * bp_x
# cp_y = 4 * self.inv_dx * self.inv_dx * bp_y
# cp_x = D_x.inverse() @ bp_x  # pyright: ignore
# cp_y = D_y.inverse() @ bp_y  # pyright: ignore
# print(D_x.inverse())

# x_stress = self.dt * ti.Vector([1, 0]) @ (stress @ x_gradient)
# y_stress = self.dt * ti.Vector([0, 1]) @ (stress @ y_gradient)
# self.face_velocity_x[x_base + offset] += x_velocity + x_stress  # + x_gravity
# self.face_velocity_y[y_base + offset] += y_velocity + y_stress  # + y_gravity
# x_velocity = self.p_mass[p] * (self.p_velocity[p][0] + x_affine @ x_dpos)
# y_velocity = self.p_mass[p] * (self.p_velocity[p][1] + y_affine @ y_dpos)

# x_gradient = ti.Vector([x_g[i][0] * x_w[j][1], x_w[i][0] * x_g[j][1]])
# y_gradient = ti.Vector([y_g[i][0] * y_w[j][1], y_w[i][0] * y_g[j][1]])
