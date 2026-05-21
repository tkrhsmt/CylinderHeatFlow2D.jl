module Pressure

using ..Utils

export solve_pressure_poisson!, pressure_boundary!

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
    field.rhs .= param.fdm.pressure_solver.plan_f(field.rhs)

    Utils.Parallel.foraxes(param.dev, Int.(param.groupsize), (nx, ny)) do i, j
        field.rhs[i, j] *= param.fdm.pressure_solver.inv_laplacian[i, j]
    end
    field.rhs .= param.fdm.pressure_solver.plan_f(field.rhs) * param.fdm.pressure_solver.inv_norm

    @views field.p[4:(nx+3), 4:(ny+3)] .= field.rhs
end

function pressure_boundary!(param::Parameters, p)
    nx, ny = param.space.num_grids
    Utils.Parallel.foraxes(param.dev, Int.(param.groupsize), (nx, ny)) do ix, iy
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
            corner_p = p[2, 2]
            p[1, 1] = corner_p; p[2, 1] = corner_p; p[3, 1] = corner_p
            p[1, 2] = corner_p; p[2, 2] = corner_p; p[3, 2] = corner_p
            p[1, 3] = corner_p; p[2, 3] = corner_p; p[3, 3] = corner_p
        elseif ix == 1 && iy == ny
            corner_p = p[2, ny+1]
            p[1, ny+4] = corner_p; p[2, ny+4] = corner_p; p[3, ny+4] = corner_p
            p[1, ny+5] = corner_p; p[2, ny+5] = corner_p; p[3, ny+5] = corner_p
            p[1, ny+6] = corner_p; p[2, ny+6] = corner_p; p[3, ny+6] = corner_p
        elseif ix == nx && iy == 1
            corner_p = p[nx+1, 2]
            p[nx+4, 1] = corner_p; p[nx+5, 1] = corner_p; p[nx+6, 1] = corner_p
            p[nx+4, 2] = corner_p; p[nx+5, 2] = corner_p; p[nx+6, 2] = corner_p
            p[nx+4, 3] = corner_p; p[nx+5, 3] = corner_p; p[nx+6, 3] = corner_p
        elseif ix == nx && iy == ny
            corner_p = p[nx+1, ny+1]
            p[nx+4, ny+4] = corner_p; p[nx+5, ny+4] = corner_p; p[nx+6, ny+4] = corner_p
            p[nx+4, ny+5] = corner_p; p[nx+5, ny+5] = corner_p; p[nx+6, ny+5] = corner_p
            p[nx+4, ny+6] = corner_p; p[nx+5, ny+6] = corner_p; p[nx+6, ny+6] = corner_p
        end
    end
end

end
