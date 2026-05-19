module Operators

using KernelAbstractions
using ..Utils

export velocity_boundary!, compute_explicit_terms!, build_pressure_rhs!, correct_velocity!

@kernel function velocity_boundary_kernel!(ux, uy, u_in, solid, ibm_x, ibm_y, nx, ny, dx, dy, dt, um, T)
    ix, iy = @index(Global, NTuple)
    ii = ix + 3
    jj = iy + 3


    if ix == 1
        ux[1, jj] = u_in[iy]
        ux[2, jj] = u_in[iy]
        ux[3, jj] = u_in[iy]
        uy[1, jj] = T(0.0)
        uy[2, jj] = T(0.0)
        uy[3, jj] = T(0.0)
    elseif ix == nx
        ux[nx+4, jj] = ux[nx+4, jj] - um * dt * (-ux[nx+3, jj] + ux[nx+4, jj]) / dx
        ux[nx+5, jj] = ux[nx+5, jj] - um * dt * (-ux[nx+4, jj] + ux[nx+5, jj]) / dx
        ux[nx+6, jj] = ux[nx+6, jj] - um * dt * (-ux[nx+5, jj] + ux[nx+6, jj]) / dx
        uy[nx+4, jj] = uy[nx+4, jj] - um * dt * (-uy[nx+3, jj] + uy[nx+4, jj]) / dx
        uy[nx+5, jj] = uy[nx+5, jj] - um * dt * (-uy[nx+4, jj] + uy[nx+5, jj]) / dx
        uy[nx+6, jj] = uy[nx+6, jj] - um * dt * (-uy[nx+5, jj] + uy[nx+6, jj]) / dx
    end

    if iy == 1
        ux[ii, 1] = -ux[ii, 6]
        ux[ii, 2] = -ux[ii, 5]
        ux[ii, 3] = -ux[ii, 4]
        uy[ii, 1] = -uy[ii, 5]
        uy[ii, 2] = -uy[ii, 4]
        uy[ii, 3] = T(0.0)
    elseif iy == ny
        ux[ii, ny+4] = -ux[ii, ny+3]
        ux[ii, ny+5] = -ux[ii, ny+2]
        ux[ii, ny+6] = -ux[ii, ny+1]
        uy[ii, ny+3] = T(0.0)
        uy[ii, ny+4] = -uy[ii, ny+2]
        uy[ii, ny+5] = -uy[ii, ny+1]
        uy[ii, ny+6] = -uy[ii, ny]
    end

    if ix == 1 && iy == 1
        ux[1:3, 1:3] .= T(0.0)
        uy[1:3, 1:3] .= T(0.0)
    elseif ix == 1 && iy == ny
        ux[1:3, ny+4:ny+6] .= T(0.0)
        uy[1:3, ny+4:ny+6] .= T(0.0)
    elseif ix == nx && iy == 1
        ux[nx+4:nx+6, 1:3] .= T(0.0)
        uy[nx+4:nx+6, 1:3] .= T(0.0)
    elseif ix == nx && iy == ny
        ux[nx+4:nx+6, ny+4:ny+6] .= T(0.0)
        uy[nx+4:nx+6, ny+4:ny+6] .= T(0.0)
    end

    if solid[ii, jj] == 1 || solid[ii+1, jj] == 1
        ibm_x[ii, jj] += ux[ii, jj] * dx * dy / dt
        ux[ii, jj] = T(0.0)
    end
    if solid[ii, jj] == 1 || solid[ii, jj+1] == 1
        ibm_y[ii, jj] += uy[ii, jj] * dx * dy / dt
        uy[ii, jj] = T(0.0)
    end

end

function velocity_boundary!(param::Parameters, ux, uy, field)
    nx, ny = param.space.num_grids
    dx, dy = param.space.dx, param.space.dy
    dt = param.time.dt
    T = param.Ttype
    um = sum(ux[nx+3, 4:ny+3]) / ny

    velocity_boundary_kernel!(param.dev, Int.(param.groupsize))(ux, uy, field.u_in, field.solid, field.ibm_x, field.ibm_y, nx, ny, dx, dy, dt, um, T; ndrange=(nx, ny))
    KernelAbstractions.synchronize(param.dev)
end

# ============================================================================================

@kernel function kernel_explicit_terms!(ux_s, uy_s, ux, uy, ν, dx, dy, dt, nx, ny, T)

    i, j = @index(Global, NTuple)
    ii = i + 3
    jj = j + 3
    dx2 = dx^2
    dy2 = dy^2

    # x-direction advection term using central difference
    adux_1 = (
        (T(9) * (ux[ii-1, jj] + ux[ii, jj]) - (ux[ii-2, jj] + ux[ii+1, jj])) * (-ux[ii-1, jj] + ux[ii, jj]) / (T(32) * dx)
        +
        (T(9) * (ux[ii, jj] + ux[ii+1, jj]) - (ux[ii-1, jj] + ux[ii+2, jj])) * (-ux[ii, jj] + ux[ii+1, jj]) / (T(32) * dx)
    )
    adux_2 = (
        (T(9) * (ux[ii-2, jj] + ux[ii-1, jj]) - (ux[ii-3, jj] + ux[ii, jj])) * (-ux[ii-3, jj] + ux[ii, jj]) / (T(96) * dx)
        +
        (T(9) * (ux[ii+1, jj] + ux[ii+2, jj]) - (ux[ii, jj] + ux[ii+3, jj])) * (-ux[ii, jj] + ux[ii+3, jj]) / (T(96) * dx)
    )
    adux = (T(9) * adux_1 - adux_2) / T(8)

    aduy_1 = (
        (T(9) * (uy[ii, jj-1] + uy[ii+1, jj-1]) - (uy[ii-1, jj-1] + uy[ii+2, jj-1])) * (-ux[ii, jj-1] + ux[ii, jj]) / (T(32) * dy)
        +
        (T(9) * (uy[ii, jj] + uy[ii+1, jj]) - (uy[ii-1, jj] + uy[ii+2, jj])) * (-ux[ii, jj] + ux[ii, jj+1]) / (T(32) * dy)
    )
    aduy_2 = (
        (T(9) * (uy[ii, jj-2] + uy[ii+1, jj-2]) - (uy[ii-1, jj-2] + uy[ii+2, jj-2])) * (-ux[ii, jj-3] + ux[ii, jj]) / (T(96) * dy)
        +
        (T(9) * (uy[ii, jj+1] + uy[ii+1, jj+1]) - (uy[ii-1, jj+1] + uy[ii+2, jj+1])) * (-ux[ii, jj] + ux[ii, jj+3]) / (T(96) * dy)
    )
    aduy = (T(9) * aduy_1 - aduy_2) / T(8)
    adv = adux + aduy

    difx = (-ux[ii+2, jj] + T(16) * ux[ii+1, jj] - T(30.0) * ux[ii, jj] + T(16) * ux[ii-1, jj] - ux[ii-2, jj]) / (T(12) * dx2)
    dify = (-ux[ii, jj+2] + T(16) * ux[ii, jj+1] - T(30.0) * ux[ii, jj] + T(16) * ux[ii, jj-1] - ux[ii, jj-2]) / (T(12) * dy2)
    lap = difx + dify

    ux_s[ii, jj] = ux[ii, jj] + dt * (-adv + ν * lap)

    # y-direction advection term using central difference
    advx_1 = (
        (T(9) * (ux[ii-1, jj] + ux[ii-1, jj+1]) - (ux[ii-1, jj-1] + ux[ii-1, jj+2])) * (-uy[ii-1, jj] + uy[ii, jj]) / (T(32) * dx)
        +
        (T(9) * (ux[ii, jj] + ux[ii, jj+1]) - (ux[ii, jj-1] + ux[ii, jj+2])) * (-uy[ii, jj] + uy[ii+1, jj]) / (T(32) * dx)
    )
    advx_2 = (
        (T(9) * (ux[ii-2, jj] + ux[ii-2, jj+1]) - (ux[ii-2, jj-1] + ux[ii-2, jj+2])) * (-uy[ii-3, jj] + uy[ii, jj]) / (T(96) * dx)
        +
        (T(9) * (ux[ii+1, jj] + ux[ii+1, jj+1]) - (ux[ii+1, jj-1] + ux[ii+1, jj+2])) * (-uy[ii, jj] + uy[ii+3, jj]) / (T(96) * dx)
    )
    advx = (T(9) * advx_1 - advx_2) / T(8)

    advy_1 = (
        (T(9) * (uy[ii, jj-1] + uy[ii, jj]) - (uy[ii, jj-2] + uy[ii, jj+1])) * (-uy[ii, jj-1] + uy[ii, jj]) / (T(32) * dy)
        +
        (T(9) * (uy[ii, jj] + uy[ii, jj+1]) - (uy[ii, jj-1] + uy[ii, jj+2])) * (-uy[ii, jj] + uy[ii, jj+1]) / (T(32) * dy)
    )
    advy_2 = (
        (T(9) * (uy[ii, jj-2] + uy[ii, jj-1]) - (uy[ii, jj-3] + uy[ii, jj])) * (-uy[ii, jj-3] + uy[ii, jj]) / (T(96) * dy)
        +
        (T(9) * (uy[ii, jj+1] + uy[ii, jj+2]) - (uy[ii, jj] + uy[ii, jj+3])) * (-uy[ii, jj] + uy[ii, jj+3]) / (T(96) * dy)
    )
    advy = (T(9) * advy_1 - advy_2) / T(8)
    adv = advx + advy

    difx = (-uy[ii+2, jj] + T(16) * uy[ii+1, jj] - T(30.0) * uy[ii, jj] + T(16) * uy[ii-1, jj] - uy[ii-2, jj]) / (T(12) * dx2)
    dify = (-uy[ii, jj+2] + T(16) * uy[ii, jj+1] - T(30.0) * uy[ii, jj] + T(16) * uy[ii, jj-1] - uy[ii, jj-2]) / (T(12) * dy2)
    lap = difx + dify

    uy_s[ii, jj] = uy[ii, jj] + dt * (-adv + ν * lap)

end

function compute_explicit_terms!(param::Parameters, field::Fields)
    nx, ny = param.space.num_grids
    groupsize = Int.(param.groupsize)
    kernel_explicit_terms!(param.dev, groupsize)(
        field.ux_s, field.uy_s, field.ux, field.uy,
        param.fdm.ν, param.space.dx, param.space.dy, param.time.dt,
        nx, ny, param.Ttype;
        ndrange=(nx, ny),
    )
    KernelAbstractions.synchronize(param.dev)
end

# ============================================================================================

@kernel function kernel_build_pressure_rhs!(rhs, ux, uy, scale, dx, dy, nx, ny, T)
    i, j = @index(Global, NTuple)
    ii = i + 3
    jj = j + 3

    rhs[i, j] = scale * (
        (ux[ii-2, jj] - T(27) * ux[ii-1, jj] + T(27) * ux[ii, jj] - ux[ii+1, jj]) / (T(24) * dx)
        +
        (uy[ii, jj-2] - T(27) * uy[ii, jj-1] + T(27) * uy[ii, jj] - uy[ii, jj+1]) / (T(24) * dy)
    )
end

"""
    build_pressure_rhs!(param, field)

Assemble the pressure Poisson right-hand side from the discrete divergence of
the staggered velocity field.
"""
function build_pressure_rhs!(param::Parameters, field::Fields)
    nx, ny = param.space.num_grids
    scale = param.Ttype(1.0) / param.time.dt
    kernel_build_pressure_rhs!(param.dev, Int.(param.groupsize))(
        field.rhs, field.ux_s, field.uy_s, scale, param.space.dx, param.space.dy, nx, ny, param.Ttype; ndrange=(nx, ny)
    )
    KernelAbstractions.synchronize(param.dev)
end

# ============================================================================================

@kernel function kernel_correct_ux!(ux, ux_s, p, scale, dx, nx, ny, T)
    i, j = @index(Global, NTuple)
    ii = i + 3
    jj = j + 3

    ux[ii, jj] = ux_s[ii, jj] - scale * (p[ii-1, jj] - T(27) * p[ii, jj] + T(27) * p[ii+1, jj] - p[ii+2, jj]) / (T(24) * dx)
end

@kernel function kernel_correct_uy!(uy, uy_s, p, scale, dy, nx, ny, T)
    i, j = @index(Global, NTuple)
    ii = i + 3
    jj = j + 3

    uy[ii, jj] = uy_s[ii, jj] - scale * (p[ii, jj-1] - T(27) * p[ii, jj] + T(27) * p[ii, jj+1] - p[ii, jj+2]) / (T(24) * dy)
end

"""
    correct_velocity!(param, field)

Apply the pressure-gradient correction and refill periodic ghost cells for the
updated velocity fields.
"""
function correct_velocity!(param::Parameters, field::Fields)
    nx, ny = param.space.num_grids
    groupsize = Int.(param.groupsize)
    scale = param.time.dt / param.Ttype(1.0)
    kernel_correct_ux!(param.dev, groupsize)(
        field.ux, field.ux_s, field.p, scale, param.space.dx, nx, ny, param.Ttype; ndrange=(nx, ny)
    )
    kernel_correct_uy!(param.dev, groupsize)(
        field.uy, field.uy_s, field.p, scale, param.space.dy, nx, ny, param.Ttype; ndrange=(nx, ny)
    )
    KernelAbstractions.synchronize(param.dev)
    velocity_boundary!(param, field.ux, field.uy, field)
    KernelAbstractions.synchronize(param.dev)
end


end
