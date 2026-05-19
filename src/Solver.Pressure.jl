module Pressure

using KernelAbstractions
using ..Utils

export solve_pressure_poisson!, pressure_boundary!

@kernel function kernel_solve_pressure_poisson!(
    rhs_h, inv_laplacian
)

    i, j = @index(Global, NTuple)
    rhs_h[i, j] *= inv_laplacian[i, j]

end

"""
    solve_pressure_poisson!(solver, param, field)

Solve the periodic pressure Poisson equation in place.

The right-hand side stored in `field.rhs` is transformed to Fourier space,
scaled by the inverse discrete Laplacian eigenvalues, and copied back into the
interior of `field.p`.
"""
function solve_pressure_poisson!(
    solver::FFTPoissonSolver,
    param::Parameters,
    field::Fields,
)
    nx, ny = param.space.num_grids
    groupsize = Int.(param.groupsize)
    field.rhs .= param.fdm.pressure_solver.plan_f(field.rhs)

    kernel_solve_pressure_poisson!(param.dev, groupsize)(
        field.rhs,
        param.fdm.pressure_solver.inv_laplacian,
        ndrange=(nx, ny)
    )
    field.rhs .= param.fdm.pressure_solver.plan_f(field.rhs) * param.fdm.pressure_solver.inv_norm

    @views field.p[4:(nx+3), 4:(ny+3)] .= field.rhs
end

@kernel function pressure_boundary_kernel!(p, nx, ny)
    ix, iy = @index(Global, NTuple)
    ii = ix + 3
    jj = iy + 3

    if ix == 1
        p[1, jj] = p[6, jj]
        p[2, jj] = p[5, jj]
        p[3, jj] = p[4, jj]
    elseif ix == nx
        p[nx+4, jj] = p[nx+3, jj]
        p[nx+5, jj] = p[nx+2, jj]
        p[nx+6, jj] = p[nx+1, jj]
    end

    if iy == 1
        p[ii, 1] = p[ii, 6]
        p[ii, 2] = p[ii, 5]
        p[ii, 3] = p[ii, 4]
    elseif iy == ny
        p[ii, ny+4] = p[ii, ny+3]
        p[ii, ny+5] = p[ii, ny+2]
        p[ii, ny+6] = p[ii, ny+1]
    end

    if ix == 1 && iy == 1
        p[1:3, 1:3] .= p[2, 2]
    elseif ix == 1 && iy == ny
        p[1:3, ny+4:ny+6] .= p[2, ny+1]
    elseif ix == nx && iy == 1
        p[nx+4:nx+6, 1:3] .= p[nx+1, 2]
    elseif ix == nx && iy == ny
        p[nx+4:nx+6, ny+4:ny+6] .= p[nx+1, ny+1]
    end
end

function pressure_boundary!(param::Parameters, p)
    nx, ny = param.space.num_grids
    dx, dy = param.space.dx, param.space.dy
    dt = param.time.dt
    T = param.Ttype

    pressure_boundary_kernel!(param.dev, Int.(param.groupsize))(p, nx, ny, ndrange=(nx, ny))
    KernelAbstractions.synchronize(param.dev)
end

end
