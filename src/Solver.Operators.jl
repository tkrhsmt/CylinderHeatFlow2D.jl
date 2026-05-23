module Operators

using KernelAbstractions
using ..Utils

export velocity_boundary!, compute_explicit_terms!, build_pressure_rhs!, correct_velocity!

function velocity_boundary!(param::Parameters, ux, uy, temp, field)
    nx, ny = param.space.num_grids
    dx, dy = param.space.dx, param.space.dy
    dt = param.time.dt
    um = sum(ux[nx+3, 4:ny+3]) / ny
    temp_wall = param.cylinder.temp_wall
    temp_cylinder = param.cylinder.temp_cylinder
    u_in = field.u_in
    temp_in = field.temp_in
    solid = field.solid
    ibm_x = field.ibm_x
    ibm_y = field.ibm_y
    z = zero(dx)
    c2 = oftype(dx, 2)
    solid_marker = one(eltype(solid))

    Utils.Parallel.foraxes(param.dev, Int.(param.groupsize), (nx, ny)) do ix, iy

        ii = ix + 3
        jj = iy + 3

        if ix == 1
            ux[1, jj] = u_in[iy]
            ux[2, jj] = u_in[iy]
            ux[3, jj] = u_in[iy]
            uy[1, jj] = z
            uy[2, jj] = z
            uy[3, jj] = z
            temp[1, jj] = temp_in[iy]
            temp[2, jj] = temp_in[iy]
            temp[3, jj] = temp_in[iy]
        elseif ix == nx
            u4 = ux[nx+4, jj] - um * dt * (-ux[nx+3, jj] + ux[nx+4, jj]) / dx
            u5 = ux[nx+5, jj] - um * dt * (-ux[nx+4, jj] + ux[nx+5, jj]) / dx
            u6 = ux[nx+6, jj] - um * dt * (-ux[nx+5, jj] + ux[nx+6, jj]) / dx
            ux[nx+4, jj] = u4
            ux[nx+5, jj] = u5
            ux[nx+6, jj] = u6
            u4 = uy[nx+4, jj] - um * dt * (-uy[nx+3, jj] + uy[nx+4, jj]) / dx
            u5 = uy[nx+5, jj] - um * dt * (-uy[nx+4, jj] + uy[nx+5, jj]) / dx
            u6 = uy[nx+6, jj] - um * dt * (-uy[nx+5, jj] + uy[nx+6, jj]) / dx
            uy[nx+4, jj] = u4
            uy[nx+5, jj] = u5
            uy[nx+6, jj] = u6
            t4 = temp[nx+4, jj] - um * dt * (-temp[nx+3, jj] + temp[nx+4, jj]) / dx
            t5 = temp[nx+5, jj] - um * dt * (-temp[nx+4, jj] + temp[nx+5, jj]) / dx
            t6 = temp[nx+6, jj] - um * dt * (-temp[nx+5, jj] + temp[nx+6, jj]) / dx
            temp[nx+4, jj] = t4
            temp[nx+5, jj] = t5
            temp[nx+6, jj] = t6
        end

        if iy == 1
            ux[ii, 1] = -ux[ii, 6]
            ux[ii, 2] = -ux[ii, 5]
            ux[ii, 3] = -ux[ii, 4]
            uy[ii, 1] = -uy[ii, 5]
            uy[ii, 2] = -uy[ii, 4]
            uy[ii, 3] = z
            temp[ii, 1] = c2 * temp_wall - temp[ii, 6]
            temp[ii, 2] = c2 * temp_wall - temp[ii, 5]
            temp[ii, 3] = c2 * temp_wall - temp[ii, 4]
        elseif iy == ny
            ux[ii, ny+4] = -ux[ii, ny+3]
            ux[ii, ny+5] = -ux[ii, ny+2]
            ux[ii, ny+6] = -ux[ii, ny+1]
            uy[ii, ny+3] = z
            uy[ii, ny+4] = -uy[ii, ny+2]
            uy[ii, ny+5] = -uy[ii, ny+1]
            uy[ii, ny+6] = -uy[ii, ny]
            temp[ii, ny+4] = c2 * temp_wall - temp[ii, ny+3]
            temp[ii, ny+5] = c2 * temp_wall - temp[ii, ny+2]
            temp[ii, ny+6] = c2 * temp_wall - temp[ii, ny+1]
        end

        if ix == 1 && iy == 1
            corner_temp = c2 * temp_wall - temp[4, 4]
            ux[1, 1] = z
            ux[2, 1] = z
            ux[3, 1] = z
            ux[1, 2] = z
            ux[2, 2] = z
            ux[3, 2] = z
            ux[1, 3] = z
            ux[2, 3] = z
            ux[3, 3] = z
            uy[1, 1] = z
            uy[2, 1] = z
            uy[3, 1] = z
            uy[1, 2] = z
            uy[2, 2] = z
            uy[3, 2] = z
            uy[1, 3] = z
            uy[2, 3] = z
            uy[3, 3] = z
            temp[1, 1] = corner_temp
            temp[2, 1] = corner_temp
            temp[3, 1] = corner_temp
            temp[1, 2] = corner_temp
            temp[2, 2] = corner_temp
            temp[3, 2] = corner_temp
            temp[1, 3] = corner_temp
            temp[2, 3] = corner_temp
            temp[3, 3] = corner_temp
        elseif ix == 1 && iy == ny
            corner_temp = c2 * temp_wall - temp[4, ny+3]
            ux[1, ny+4] = z
            ux[2, ny+4] = z
            ux[3, ny+4] = z
            ux[1, ny+5] = z
            ux[2, ny+5] = z
            ux[3, ny+5] = z
            ux[1, ny+6] = z
            ux[2, ny+6] = z
            ux[3, ny+6] = z
            uy[1, ny+4] = z
            uy[2, ny+4] = z
            uy[3, ny+4] = z
            uy[1, ny+5] = z
            uy[2, ny+5] = z
            uy[3, ny+5] = z
            uy[1, ny+6] = z
            uy[2, ny+6] = z
            uy[3, ny+6] = z
            temp[1, ny+4] = corner_temp
            temp[2, ny+4] = corner_temp
            temp[3, ny+4] = corner_temp
            temp[1, ny+5] = corner_temp
            temp[2, ny+5] = corner_temp
            temp[3, ny+5] = corner_temp
            temp[1, ny+6] = corner_temp
            temp[2, ny+6] = corner_temp
            temp[3, ny+6] = corner_temp
        elseif ix == nx && iy == 1
            corner_temp = c2 * temp_wall - temp[nx+3, 4]
            ux[nx+4, 1] = z
            ux[nx+5, 1] = z
            ux[nx+6, 1] = z
            ux[nx+4, 2] = z
            ux[nx+5, 2] = z
            ux[nx+6, 2] = z
            ux[nx+4, 3] = z
            ux[nx+5, 3] = z
            ux[nx+6, 3] = z
            uy[nx+4, 1] = z
            uy[nx+5, 1] = z
            uy[nx+6, 1] = z
            uy[nx+4, 2] = z
            uy[nx+5, 2] = z
            uy[nx+6, 2] = z
            uy[nx+4, 3] = z
            uy[nx+5, 3] = z
            uy[nx+6, 3] = z
            temp[nx+4, 1] = corner_temp
            temp[nx+5, 1] = corner_temp
            temp[nx+6, 1] = corner_temp
            temp[nx+4, 2] = corner_temp
            temp[nx+5, 2] = corner_temp
            temp[nx+6, 2] = corner_temp
            temp[nx+4, 3] = corner_temp
            temp[nx+5, 3] = corner_temp
            temp[nx+6, 3] = corner_temp
        elseif ix == nx && iy == ny
            corner_temp = c2 * temp_wall - temp[nx+3, ny+3]
            ux[nx+4, ny+4] = z
            ux[nx+5, ny+4] = z
            ux[nx+6, ny+4] = z
            ux[nx+4, ny+5] = z
            ux[nx+5, ny+5] = z
            ux[nx+6, ny+5] = z
            ux[nx+4, ny+6] = z
            ux[nx+5, ny+6] = z
            ux[nx+6, ny+6] = z
            uy[nx+4, ny+4] = z
            uy[nx+5, ny+4] = z
            uy[nx+6, ny+4] = z
            uy[nx+4, ny+5] = z
            uy[nx+5, ny+5] = z
            uy[nx+6, ny+5] = z
            uy[nx+4, ny+6] = z
            uy[nx+5, ny+6] = z
            uy[nx+6, ny+6] = z
            temp[nx+4, ny+4] = corner_temp
            temp[nx+5, ny+4] = corner_temp
            temp[nx+6, ny+4] = corner_temp
            temp[nx+4, ny+5] = corner_temp
            temp[nx+5, ny+5] = corner_temp
            temp[nx+6, ny+5] = corner_temp
            temp[nx+4, ny+6] = corner_temp
            temp[nx+5, ny+6] = corner_temp
            temp[nx+6, ny+6] = corner_temp
        end

        if solid[ii, jj] == solid_marker || solid[ii+1, jj] == solid_marker
            ibm_x[ii, jj] += ux[ii, jj] * dx * dy / dt
            ux[ii, jj] = z
        end
        if solid[ii, jj] == solid_marker || solid[ii, jj+1] == solid_marker
            ibm_y[ii, jj] += uy[ii, jj] * dx * dy / dt
            uy[ii, jj] = z
        end
        if solid[ii, jj] == solid_marker
            temp[ii, jj] = temp_cylinder
        end

    end
end

# ============================================================================================

function compute_explicit_terms!(param::Parameters, field::Fields)
    nx, ny = param.space.num_grids
    μ = param.fdm.μ
    α = param.fdm.α
    β = param.fdm.β
    g = param.fdm.g
    dx = param.space.dx
    dy = param.space.dy
    dt = param.time.dt
    T_in = param.cylinder.temp_wall
    ρ0 = param.fdm.ρ0
    dx2 = dx^2
    dy2 = dy^2
    ux = field.ux
    uy = field.uy
    temp = field.temp
    ux_s = field.ux_s
    uy_s = field.uy_s
    temp_s = field.temp_s
    c2 = oftype(dx, 2)
    c6 = oftype(dx, 6)
    c8 = oftype(dx, 8)
    c9 = oftype(dx, 9)
    c12 = oftype(dx, 12)
    c16 = oftype(dx, 16)
    c30 = oftype(dx, 30)
    c32 = oftype(dx, 32)
    c96 = oftype(dx, 96)
    c192 = oftype(dx, 192)

    Utils.Parallel.foraxes(param.dev, Int.(param.groupsize), (nx, ny)) do i, j
        ii = i + 3
        jj = j + 3

        adux_1 = (
            (c9 * (ux[ii-1, jj] + ux[ii, jj]) - (ux[ii-2, jj] + ux[ii+1, jj])) * (-ux[ii-1, jj] + ux[ii, jj]) / (c32 * dx)
            +
            (c9 * (ux[ii, jj] + ux[ii+1, jj]) - (ux[ii-1, jj] + ux[ii+2, jj])) * (-ux[ii, jj] + ux[ii+1, jj]) / (c32 * dx)
        )
        adux_2 = (
            (c9 * (ux[ii-2, jj] + ux[ii-1, jj]) - (ux[ii-3, jj] + ux[ii, jj])) * (-ux[ii-3, jj] + ux[ii, jj]) / (c96 * dx)
            +
            (c9 * (ux[ii+1, jj] + ux[ii+2, jj]) - (ux[ii, jj] + ux[ii+3, jj])) * (-ux[ii, jj] + ux[ii+3, jj]) / (c96 * dx)
        )
        adux = (c9 * adux_1 - adux_2) / c8

        aduy_1 = (
            (c9 * (uy[ii, jj-1] + uy[ii+1, jj-1]) - (uy[ii-1, jj-1] + uy[ii+2, jj-1])) * (-ux[ii, jj-1] + ux[ii, jj]) / (c32 * dy)
            +
            (c9 * (uy[ii, jj] + uy[ii+1, jj]) - (uy[ii-1, jj] + uy[ii+2, jj])) * (-ux[ii, jj] + ux[ii, jj+1]) / (c32 * dy)
        )
        aduy_2 = (
            (c9 * (uy[ii, jj-2] + uy[ii+1, jj-2]) - (uy[ii-1, jj-2] + uy[ii+2, jj-2])) * (-ux[ii, jj-3] + ux[ii, jj]) / (c96 * dy)
            +
            (c9 * (uy[ii, jj+1] + uy[ii+1, jj+1]) - (uy[ii-1, jj+1] + uy[ii+2, jj+1])) * (-ux[ii, jj] + ux[ii, jj+3]) / (c96 * dy)
        )
        aduy = (c9 * aduy_1 - aduy_2) / c8
        adv = adux + aduy

        difx = (-ux[ii+2, jj] + c16 * ux[ii+1, jj] - c30 * ux[ii, jj] + c16 * ux[ii-1, jj] - ux[ii-2, jj]) / (c12 * dx2)
        dify = (-ux[ii, jj+2] + c16 * ux[ii, jj+1] - c30 * ux[ii, jj] + c16 * ux[ii, jj-1] - ux[ii, jj-2]) / (c12 * dy2)
        lap = difx + dify

        temp_ij = temp[ii, jj]
        temp_x = (-temp[ii-1, jj] + c9 * temp[ii, jj] + c9 * temp[ii+1, jj] - temp[ii+2, jj]) / c16
        temp_y = (-temp[ii, jj-1] + c9 * temp[ii, jj] + c9 * temp[ii, jj+1] - temp[ii, jj+2]) / c16

        ux_s[ii, jj] = ux[ii, jj] + dt * (-adv + μ(temp_x) / ρ0 * lap + β(temp_x) * g * (temp_x - T_in))

        advx_1 = (
            (c9 * (ux[ii-1, jj] + ux[ii-1, jj+1]) - (ux[ii-1, jj-1] + ux[ii-1, jj+2])) * (-uy[ii-1, jj] + uy[ii, jj]) / (c32 * dx)
            +
            (c9 * (ux[ii, jj] + ux[ii, jj+1]) - (ux[ii, jj-1] + ux[ii, jj+2])) * (-uy[ii, jj] + uy[ii+1, jj]) / (c32 * dx)
        )
        advx_2 = (
            (c9 * (ux[ii-2, jj] + ux[ii-2, jj+1]) - (ux[ii-2, jj-1] + ux[ii-2, jj+2])) * (-uy[ii-3, jj] + uy[ii, jj]) / (c96 * dx)
            +
            (c9 * (ux[ii+1, jj] + ux[ii+1, jj+1]) - (ux[ii+1, jj-1] + ux[ii+1, jj+2])) * (-uy[ii, jj] + uy[ii+3, jj]) / (c96 * dx)
        )
        advx = (c9 * advx_1 - advx_2) / c8

        advy_1 = (
            (c9 * (uy[ii, jj-1] + uy[ii, jj]) - (uy[ii, jj-2] + uy[ii, jj+1])) * (-uy[ii, jj-1] + uy[ii, jj]) / (c32 * dy)
            +
            (c9 * (uy[ii, jj] + uy[ii, jj+1]) - (uy[ii, jj-1] + uy[ii, jj+2])) * (-uy[ii, jj] + uy[ii, jj+1]) / (c32 * dy)
        )
        advy_2 = (
            (c9 * (uy[ii, jj-2] + uy[ii, jj-1]) - (uy[ii, jj-3] + uy[ii, jj])) * (-uy[ii, jj-3] + uy[ii, jj]) / (c96 * dy)
            +
            (c9 * (uy[ii, jj+1] + uy[ii, jj+2]) - (uy[ii, jj] + uy[ii, jj+3])) * (-uy[ii, jj] + uy[ii, jj+3]) / (c96 * dy)
        )
        advy = (c9 * advy_1 - advy_2) / c8
        adv = advx + advy

        difx = (-uy[ii+2, jj] + c16 * uy[ii+1, jj] - c30 * uy[ii, jj] + c16 * uy[ii-1, jj] - uy[ii-2, jj]) / (c12 * dx2)
        dify = (-uy[ii, jj+2] + c16 * uy[ii, jj+1] - c30 * uy[ii, jj] + c16 * uy[ii, jj-1] - uy[ii, jj-2]) / (c12 * dy2)
        lap = difx + dify
        uy_s[ii, jj] = uy[ii, jj] + dt * (-adv + μ(temp_y) / ρ0 * lap)

        adtx_1 = (
            ux[ii, jj] * (temp[ii+1, jj] - temp[ii, jj]) / (c2 * dx)
            +
            ux[ii-1, jj] * (temp[ii, jj] - temp[ii-1, jj]) / (c2 * dx)
        )
        adtx_2 = (
            ux[ii+1, jj] * (temp[ii+3, jj] - temp[ii, jj]) / (c6 * dx)
            +
            ux[ii-2, jj] * (temp[ii, jj] - temp[ii-3, jj]) / (c6 * dx)
        )
        adtx = (c9 * adtx_1 - adtx_2) / c8
        adty_1 = (
            uy[ii, jj] * (temp[ii, jj+1] - temp[ii, jj]) / (c2 * dy)
            +
            uy[ii, jj-1] * (temp[ii, jj] - temp[ii, jj-1]) / (c2 * dy)
        )
        adty_2 = (
            uy[ii, jj+1] * (temp[ii, jj+3] - temp[ii, jj]) / (c6 * dy)
            +
            uy[ii, jj-2] * (temp[ii, jj] - temp[ii, jj-3]) / (c6 * dy)
        )
        adty = (c9 * adty_1 - adty_2) / c8
        adt = adtx + adty

        difx = (-temp[ii+2, jj] + c16 * temp[ii+1, jj] - c30 * temp[ii, jj] + c16 * temp[ii-1, jj] - temp[ii-2, jj]) / (c12 * dx2)
        dify = (-temp[ii, jj+2] + c16 * temp[ii, jj+1] - c30 * temp[ii, jj] + c16 * temp[ii, jj-1] - temp[ii, jj-2]) / (c12 * dy2)
        lap = difx + dify
        temp_s[ii, jj] = temp_ij + dt * (-adt + α * lap)
    end
end

# ============================================================================================

"""
    build_pressure_rhs!(param, field)

Assemble the pressure Poisson right-hand side from the discrete divergence of
the staggered velocity field.
"""
function build_pressure_rhs!(param::Parameters, field::Fields)
    nx, ny = param.space.num_grids
    scale = param.fdm.ρ0 / param.time.dt
    dx = param.space.dx
    dy = param.space.dy
    rhs = field.rhs
    ux_s = field.ux_s
    uy_s = field.uy_s
    c24 = oftype(dx, 24)
    c27 = oftype(dx, 27)
    Utils.Parallel.foraxes(param.dev, Int.(param.groupsize), (nx, ny)) do i, j
        ii = i + 3
        jj = j + 3
        rhs[i, j] = scale * (
            (ux_s[ii-2, jj] - c27 * ux_s[ii-1, jj] + c27 * ux_s[ii, jj] - ux_s[ii+1, jj]) / (c24 * dx)
            +
            (uy_s[ii, jj-2] - c27 * uy_s[ii, jj-1] + c27 * uy_s[ii, jj] - uy_s[ii, jj+1]) / (c24 * dy)
        )
    end
end

# ============================================================================================

"""
    correct_velocity!(param, field)

Apply the pressure-gradient correction and refill periodic ghost cells for the
updated velocity fields.
"""
function correct_velocity!(param::Parameters, field::Fields)
    nx, ny = param.space.num_grids
    scale = param.time.dt / param.fdm.ρ0
    dx = param.space.dx
    dy = param.space.dy
    ux = field.ux
    uy = field.uy
    ux_s = field.ux_s
    uy_s = field.uy_s
    p = field.p
    c24 = oftype(dx, 24)
    c27 = oftype(dx, 27)
    Utils.Parallel.foraxes(param.dev, Int.(param.groupsize), (nx, ny)) do i, j
        ii = i + 3
        jj = j + 3
        ux[ii, jj] = ux_s[ii, jj] - scale * (p[ii-1, jj] - c27 * p[ii, jj] + c27 * p[ii+1, jj] - p[ii+2, jj]) / (c24 * dx)
        uy[ii, jj] = uy_s[ii, jj] - scale * (p[ii, jj-1] - c27 * p[ii, jj] + c27 * p[ii, jj+1] - p[ii, jj+2]) / (c24 * dy)
    end
    field.temp .= field.temp_s
    velocity_boundary!(param, field.ux, field.uy, field.temp, field)
end


end
