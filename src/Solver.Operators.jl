module Operators

using KernelAbstractions
using ..Utils

export velocity_boundary!, compute_explicit_terms!, build_pressure_rhs!, correct_velocity!

function velocity_boundary!(param::Parameters, ux, uy, temp, field)
    nx, ny = param.space.num_grids
    dx, dy = param.space.dx, param.space.dy
    dt = param.time.dt
    T = param.Ttype
    um = sum(ux[nx+3, 4:ny+3]) / ny
    temp_wall = param.cylinder.temp_wall
    temp_cylinder = param.cylinder.temp_cylinder

    Utils.Parallel.foraxes(param.dev, Int.(param.groupsize), (nx, ny)) do ix, iy

        ii = ix + 3
        jj = iy + 3

        if ix == 1
            ux[1, jj] = field.u_in[iy]
            ux[2, jj] = field.u_in[iy]
            ux[3, jj] = field.u_in[iy]
            uy[1, jj] = T(0.0)
            uy[2, jj] = T(0.0)
            uy[3, jj] = T(0.0)
            temp[1, jj] = field.temp_in[iy]
            temp[2, jj] = field.temp_in[iy]
            temp[3, jj] = field.temp_in[iy]
        elseif ix == nx
            ux[nx+4, jj] = ux[nx+4, jj] - um * dt * (-ux[nx+3, jj] + ux[nx+4, jj]) / dx
            ux[nx+5, jj] = ux[nx+5, jj] - um * dt * (-ux[nx+4, jj] + ux[nx+5, jj]) / dx
            ux[nx+6, jj] = ux[nx+6, jj] - um * dt * (-ux[nx+5, jj] + ux[nx+6, jj]) / dx
            uy[nx+4, jj] = uy[nx+4, jj] - um * dt * (-uy[nx+3, jj] + uy[nx+4, jj]) / dx
            uy[nx+5, jj] = uy[nx+5, jj] - um * dt * (-uy[nx+4, jj] + uy[nx+5, jj]) / dx
            uy[nx+6, jj] = uy[nx+6, jj] - um * dt * (-uy[nx+5, jj] + uy[nx+6, jj]) / dx
            temp[nx+4, jj] = temp[nx+4, jj] - um * dt * (-temp[nx+3, jj] + temp[nx+4, jj]) / dx
            temp[nx+5, jj] = temp[nx+5, jj] - um * dt * (-temp[nx+4, jj] + temp[nx+5, jj]) / dx
            temp[nx+6, jj] = temp[nx+6, jj] - um * dt * (-temp[nx+5, jj] + temp[nx+6, jj]) / dx
        end

        if iy == 1
            ux[ii, 1] = -ux[ii, 6]
            ux[ii, 2] = -ux[ii, 5]
            ux[ii, 3] = -ux[ii, 4]
            uy[ii, 1] = -uy[ii, 5]
            uy[ii, 2] = -uy[ii, 4]
            uy[ii, 3] = T(0.0)
            temp[ii, 1] = T(2) * temp_wall - temp[ii, 6]
            temp[ii, 2] = T(2) * temp_wall - temp[ii, 5]
            temp[ii, 3] = T(2) * temp_wall - temp[ii, 4]
        elseif iy == ny
            ux[ii, ny+4] = -ux[ii, ny+3]
            ux[ii, ny+5] = -ux[ii, ny+2]
            ux[ii, ny+6] = -ux[ii, ny+1]
            uy[ii, ny+3] = T(0.0)
            uy[ii, ny+4] = -uy[ii, ny+2]
            uy[ii, ny+5] = -uy[ii, ny+1]
            uy[ii, ny+6] = -uy[ii, ny]
            temp[ii, ny+4] = T(2) * temp_wall - temp[ii, ny+3]
            temp[ii, ny+5] = T(2) * temp_wall - temp[ii, ny+2]
            temp[ii, ny+6] = T(2) * temp_wall - temp[ii, ny+1]
        end

        if ix == 1 && iy == 1
            corner_temp = T(2) * temp_wall - temp[4, 4]
            ux[1, 1] = T(0.0); ux[2, 1] = T(0.0); ux[3, 1] = T(0.0)
            ux[1, 2] = T(0.0); ux[2, 2] = T(0.0); ux[3, 2] = T(0.0)
            ux[1, 3] = T(0.0); ux[2, 3] = T(0.0); ux[3, 3] = T(0.0)
            uy[1, 1] = T(0.0); uy[2, 1] = T(0.0); uy[3, 1] = T(0.0)
            uy[1, 2] = T(0.0); uy[2, 2] = T(0.0); uy[3, 2] = T(0.0)
            uy[1, 3] = T(0.0); uy[2, 3] = T(0.0); uy[3, 3] = T(0.0)
            temp[1, 1] = corner_temp; temp[2, 1] = corner_temp; temp[3, 1] = corner_temp
            temp[1, 2] = corner_temp; temp[2, 2] = corner_temp; temp[3, 2] = corner_temp
            temp[1, 3] = corner_temp; temp[2, 3] = corner_temp; temp[3, 3] = corner_temp
        elseif ix == 1 && iy == ny
            corner_temp = T(2) * temp_wall - temp[4, ny+3]
            ux[1, ny+4] = T(0.0); ux[2, ny+4] = T(0.0); ux[3, ny+4] = T(0.0)
            ux[1, ny+5] = T(0.0); ux[2, ny+5] = T(0.0); ux[3, ny+5] = T(0.0)
            ux[1, ny+6] = T(0.0); ux[2, ny+6] = T(0.0); ux[3, ny+6] = T(0.0)
            uy[1, ny+4] = T(0.0); uy[2, ny+4] = T(0.0); uy[3, ny+4] = T(0.0)
            uy[1, ny+5] = T(0.0); uy[2, ny+5] = T(0.0); uy[3, ny+5] = T(0.0)
            uy[1, ny+6] = T(0.0); uy[2, ny+6] = T(0.0); uy[3, ny+6] = T(0.0)
            temp[1, ny+4] = corner_temp; temp[2, ny+4] = corner_temp; temp[3, ny+4] = corner_temp
            temp[1, ny+5] = corner_temp; temp[2, ny+5] = corner_temp; temp[3, ny+5] = corner_temp
            temp[1, ny+6] = corner_temp; temp[2, ny+6] = corner_temp; temp[3, ny+6] = corner_temp
        elseif ix == nx && iy == 1
            corner_temp = T(2) * temp_wall - temp[nx+3, 4]
            ux[nx+4, 1] = T(0.0); ux[nx+5, 1] = T(0.0); ux[nx+6, 1] = T(0.0)
            ux[nx+4, 2] = T(0.0); ux[nx+5, 2] = T(0.0); ux[nx+6, 2] = T(0.0)
            ux[nx+4, 3] = T(0.0); ux[nx+5, 3] = T(0.0); ux[nx+6, 3] = T(0.0)
            uy[nx+4, 1] = T(0.0); uy[nx+5, 1] = T(0.0); uy[nx+6, 1] = T(0.0)
            uy[nx+4, 2] = T(0.0); uy[nx+5, 2] = T(0.0); uy[nx+6, 2] = T(0.0)
            uy[nx+4, 3] = T(0.0); uy[nx+5, 3] = T(0.0); uy[nx+6, 3] = T(0.0)
            temp[nx+4, 1] = corner_temp; temp[nx+5, 1] = corner_temp; temp[nx+6, 1] = corner_temp
            temp[nx+4, 2] = corner_temp; temp[nx+5, 2] = corner_temp; temp[nx+6, 2] = corner_temp
            temp[nx+4, 3] = corner_temp; temp[nx+5, 3] = corner_temp; temp[nx+6, 3] = corner_temp
        elseif ix == nx && iy == ny
            corner_temp = T(2) * temp_wall - temp[nx+3, ny+3]
            ux[nx+4, ny+4] = T(0.0); ux[nx+5, ny+4] = T(0.0); ux[nx+6, ny+4] = T(0.0)
            ux[nx+4, ny+5] = T(0.0); ux[nx+5, ny+5] = T(0.0); ux[nx+6, ny+5] = T(0.0)
            ux[nx+4, ny+6] = T(0.0); ux[nx+5, ny+6] = T(0.0); ux[nx+6, ny+6] = T(0.0)
            uy[nx+4, ny+4] = T(0.0); uy[nx+5, ny+4] = T(0.0); uy[nx+6, ny+4] = T(0.0)
            uy[nx+4, ny+5] = T(0.0); uy[nx+5, ny+5] = T(0.0); uy[nx+6, ny+5] = T(0.0)
            uy[nx+4, ny+6] = T(0.0); uy[nx+5, ny+6] = T(0.0); uy[nx+6, ny+6] = T(0.0)
            temp[nx+4, ny+4] = corner_temp; temp[nx+5, ny+4] = corner_temp; temp[nx+6, ny+4] = corner_temp
            temp[nx+4, ny+5] = corner_temp; temp[nx+5, ny+5] = corner_temp; temp[nx+6, ny+5] = corner_temp
            temp[nx+4, ny+6] = corner_temp; temp[nx+5, ny+6] = corner_temp; temp[nx+6, ny+6] = corner_temp
        end

        if field.solid[ii, jj] == 1 || field.solid[ii+1, jj] == 1
            field.ibm_x[ii, jj] += ux[ii, jj] * dx * dy / dt
            ux[ii, jj] = T(0.0)
        end
        if field.solid[ii, jj] == 1 || field.solid[ii, jj+1] == 1
            field.ibm_y[ii, jj] += uy[ii, jj] * dx * dy / dt
            uy[ii, jj] = T(0.0)
        end
        if field.solid[ii, jj] == 1
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
    T = param.Ttype
    T_in = param.cylinder.temp_wall
    ρ0 = param.fdm.ρ0
    dx2 = dx^2
    dy2 = dy^2

    Utils.Parallel.foraxes(param.dev, Int.(param.groupsize), (nx, ny)) do i, j
        ii = i + 3
        jj = j + 3

        adux_1 = (
            (T(9) * (field.ux[ii-1, jj] + field.ux[ii, jj]) - (field.ux[ii-2, jj] + field.ux[ii+1, jj])) * (-field.ux[ii-1, jj] + field.ux[ii, jj]) / (T(32) * dx)
            +
            (T(9) * (field.ux[ii, jj] + field.ux[ii+1, jj]) - (field.ux[ii-1, jj] + field.ux[ii+2, jj])) * (-field.ux[ii, jj] + field.ux[ii+1, jj]) / (T(32) * dx)
        )
        adux_2 = (
            (T(9) * (field.ux[ii-2, jj] + field.ux[ii-1, jj]) - (field.ux[ii-3, jj] + field.ux[ii, jj])) * (-field.ux[ii-3, jj] + field.ux[ii, jj]) / (T(96) * dx)
            +
            (T(9) * (field.ux[ii+1, jj] + field.ux[ii+2, jj]) - (field.ux[ii, jj] + field.ux[ii+3, jj])) * (-field.ux[ii, jj] + field.ux[ii+3, jj]) / (T(96) * dx)
        )
        adux = (T(9) * adux_1 - adux_2) / T(8)

        aduy_1 = (
            (T(9) * (field.uy[ii, jj-1] + field.uy[ii+1, jj-1]) - (field.uy[ii-1, jj-1] + field.uy[ii+2, jj-1])) * (-field.ux[ii, jj-1] + field.ux[ii, jj]) / (T(32) * dy)
            +
            (T(9) * (field.uy[ii, jj] + field.uy[ii+1, jj]) - (field.uy[ii-1, jj] + field.uy[ii+2, jj])) * (-field.ux[ii, jj] + field.ux[ii, jj+1]) / (T(32) * dy)
        )
        aduy_2 = (
            (T(9) * (field.uy[ii, jj-2] + field.uy[ii+1, jj-2]) - (field.uy[ii-1, jj-2] + field.uy[ii+2, jj-2])) * (-field.ux[ii, jj-3] + field.ux[ii, jj]) / (T(96) * dy)
            +
            (T(9) * (field.uy[ii, jj+1] + field.uy[ii+1, jj+1]) - (field.uy[ii-1, jj+1] + field.uy[ii+2, jj+1])) * (-field.ux[ii, jj] + field.ux[ii, jj+3]) / (T(96) * dy)
        )
        aduy = (T(9) * aduy_1 - aduy_2) / T(8)
        adv = adux + aduy

        difx = (-field.ux[ii+2, jj] + T(16) * field.ux[ii+1, jj] - T(30.0) * field.ux[ii, jj] + T(16) * field.ux[ii-1, jj] - field.ux[ii-2, jj]) / (T(12) * dx2)
        dify = (-field.ux[ii, jj+2] + T(16) * field.ux[ii, jj+1] - T(30.0) * field.ux[ii, jj] + T(16) * field.ux[ii, jj-1] - field.ux[ii, jj-2]) / (T(12) * dy2)
        lap = difx + dify

        temp_ij = field.temp[ii, jj]
        field.ux_s[ii, jj] = field.ux[ii, jj] + dt * (-adv + μ(temp_ij) / ρ0 * lap - β(temp_ij) * g * (temp_ij - T_in))

        advx_1 = (
            (T(9) * (field.ux[ii-1, jj] + field.ux[ii-1, jj+1]) - (field.ux[ii-1, jj-1] + field.ux[ii-1, jj+2])) * (-field.uy[ii-1, jj] + field.uy[ii, jj]) / (T(32) * dx)
            +
            (T(9) * (field.ux[ii, jj] + field.ux[ii, jj+1]) - (field.ux[ii, jj-1] + field.ux[ii, jj+2])) * (-field.uy[ii, jj] + field.uy[ii+1, jj]) / (T(32) * dx)
        )
        advx_2 = (
            (T(9) * (field.ux[ii-2, jj] + field.ux[ii-2, jj+1]) - (field.ux[ii-2, jj-1] + field.ux[ii-2, jj+2])) * (-field.uy[ii-3, jj] + field.uy[ii, jj]) / (T(96) * dx)
            +
            (T(9) * (field.ux[ii+1, jj] + field.ux[ii+1, jj+1]) - (field.ux[ii+1, jj-1] + field.ux[ii+1, jj+2])) * (-field.uy[ii, jj] + field.uy[ii+3, jj]) / (T(96) * dx)
        )
        advx = (T(9) * advx_1 - advx_2) / T(8)

        advy_1 = (
            (T(9) * (field.uy[ii, jj-1] + field.uy[ii, jj]) - (field.uy[ii, jj-2] + field.uy[ii, jj+1])) * (-field.uy[ii, jj-1] + field.uy[ii, jj]) / (T(32) * dy)
            +
            (T(9) * (field.uy[ii, jj] + field.uy[ii, jj+1]) - (field.uy[ii, jj-1] + field.uy[ii, jj+2])) * (-field.uy[ii, jj] + field.uy[ii, jj+1]) / (T(32) * dy)
        )
        advy_2 = (
            (T(9) * (field.uy[ii, jj-2] + field.uy[ii, jj-1]) - (field.uy[ii, jj-3] + field.uy[ii, jj])) * (-field.uy[ii, jj-3] + field.uy[ii, jj]) / (T(96) * dy)
            +
            (T(9) * (field.uy[ii, jj+1] + field.uy[ii, jj+2]) - (field.uy[ii, jj] + field.uy[ii, jj+3])) * (-field.uy[ii, jj] + field.uy[ii, jj+3]) / (T(96) * dy)
        )
        advy = (T(9) * advy_1 - advy_2) / T(8)
        adv = advx + advy

        difx = (-field.uy[ii+2, jj] + T(16) * field.uy[ii+1, jj] - T(30.0) * field.uy[ii, jj] + T(16) * field.uy[ii-1, jj] - field.uy[ii-2, jj]) / (T(12) * dx2)
        dify = (-field.uy[ii, jj+2] + T(16) * field.uy[ii, jj+1] - T(30.0) * field.uy[ii, jj] + T(16) * field.uy[ii, jj-1] - field.uy[ii, jj-2]) / (T(12) * dy2)
        lap = difx + dify
        field.uy_s[ii, jj] = field.uy[ii, jj] + dt * (-adv + μ(temp_ij) / ρ0 * lap)

        adtx = (
            (-field.ux[ii-2, jj] + T(9) * field.ux[ii-1, jj] + T(9) * field.ux[ii, jj] - field.ux[ii+1, jj])
            *
            (field.temp[ii-2, jj] - T(8) * field.temp[ii-1, jj] + T(8) * field.temp[ii+1, jj] - field.temp[ii+2, jj])
        ) / (T(192) * dx)
        adty = (
            (-field.uy[ii, jj-2] + T(9) * field.uy[ii, jj-1] + T(9) * field.uy[ii, jj] - field.uy[ii, jj+1])
            *
            (field.temp[ii, jj-2] - T(8) * field.temp[ii, jj-1] + T(8) * field.temp[ii, jj+1] - field.temp[ii, jj+2])
        ) / (T(192) * dy)
        adt = adtx + adty

        difx = (-field.temp[ii+2, jj] + T(16) * field.temp[ii+1, jj] - T(30.0) * field.temp[ii, jj] + T(16) * field.temp[ii-1, jj] - field.temp[ii-2, jj]) / (T(12) * dx2)
        dify = (-field.temp[ii, jj+2] + T(16) * field.temp[ii, jj+1] - T(30.0) * field.temp[ii, jj] + T(16) * field.temp[ii, jj-1] - field.temp[ii, jj-2]) / (T(12) * dy2)
        lap = difx + dify
        field.temp_s[ii, jj] = temp_ij + dt * (-adt + α * lap)
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
    T = param.Ttype
    Utils.Parallel.foraxes(param.dev, Int.(param.groupsize), (nx, ny)) do i, j
        ii = i + 3
        jj = j + 3
        field.rhs[i, j] = scale * (
            (field.ux_s[ii-2, jj] - T(27) * field.ux_s[ii-1, jj] + T(27) * field.ux_s[ii, jj] - field.ux_s[ii+1, jj]) / (T(24) * dx)
            +
            (field.uy_s[ii, jj-2] - T(27) * field.uy_s[ii, jj-1] + T(27) * field.uy_s[ii, jj] - field.uy_s[ii, jj+1]) / (T(24) * dy)
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
    T = param.Ttype
    Utils.Parallel.foraxes(param.dev, Int.(param.groupsize), (nx, ny)) do i, j
        ii = i + 3
        jj = j + 3
        field.ux[ii, jj] = field.ux_s[ii, jj] - scale * (field.p[ii-1, jj] - T(27) * field.p[ii, jj] + T(27) * field.p[ii+1, jj] - field.p[ii+2, jj]) / (T(24) * dx)
        field.uy[ii, jj] = field.uy_s[ii, jj] - scale * (field.p[ii, jj-1] - T(27) * field.p[ii, jj] + T(27) * field.p[ii, jj+1] - field.p[ii, jj+2]) / (T(24) * dy)
    end
    field.temp .= field.temp_s
    velocity_boundary!(param, field.ux, field.uy, field.temp, field)
end


end
