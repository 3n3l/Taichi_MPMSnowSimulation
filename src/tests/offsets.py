import taichi as ti

ti.init(arch=ti.cpu)


@ti.kernel
def main():
    xp = ti.Vector([0.5, 0.5])
    inv_dx = 4

    c_base = (xp * inv_dx - 0.5).cast(ti.i32)  # pyright: ignore
    c_fx = xp * inv_dx - c_base.cast(ti.f32)
    print("c_base: ", c_base)
    print("c_fx: ", c_fx)

    x_stagger = ti.Vector([0.0, 0.5])
    x_base = (xp * inv_dx - x_stagger).cast(ti.i32)  # pyright: ignore
    x_fx = xp * inv_dx - x_base.cast(ti.f32)
    # x_fx = xp * inv_dx - (x_base.cast(ti.f32) + ti.Vector([0.5, 0]))
    print("x_base: ", x_base)
    print("x_fx: ", x_fx)

    y_stagger = ti.Vector([0.5, 0.0])
    y_base = (xp * inv_dx - y_stagger).cast(ti.i32)  # pyright: ignore
    y_fx = xp * inv_dx - y_base.cast(ti.f32)
    # y_fx = xp * inv_dx - (y_base.cast(ti.f32) + ti.Vector([0, 0.5]))
    print("y_base: ", y_base)
    print("y_fx: ", y_fx)


if __name__ == "__main__":
    main()
