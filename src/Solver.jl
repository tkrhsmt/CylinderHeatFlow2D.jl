module Solver

using ..Utils

include("Solver.Operators.jl")
using .Operators
include("Solver.Pressure.jl")
using .Pressure

export time_march!

"""
    combine_rk3_ibm_force(stage1, stage2, stage3, scale, T)

Combine immersed-boundary force samples from the three SSPRK3 stages into a
single non-dimensional coefficient.
"""
function combine_rk3_ibm_force(stage1, stage2, stage3, scale, T)
    return (stage1 + stage2 + T(4) * stage3) / (T(6) * scale)
end

"""
    step!(param, fields)

Advance the solver by one internal time step.
"""
function step!(param::Parameters, fields::Fields)

    # Compute explicit terms and predict intermediate velocity
    compute_explicit_terms!(param, fields)

    # Fill ghost cells for predicted velocity
    velocity_boundary!(param, fields.ux_s, fields.uy_s, fields.temp_s, fields)

    # Build pressure Poisson equation RHS and solve for pressure
    build_pressure_rhs!(param, fields)
    solve_pressure_poisson!(param.fdm.pressure_solver, param, fields)
    pressure_boundary!(param, fields.p)

    # Correct velocity using pressure gradient
    correct_velocity!(param, fields)
end


"""
    time_march!(param, fields)

Advance the simulation by `param.time.num_time_interval` time steps in place.

Each step applies the explicit update, optional stochastic momentum terms,
pressure projection, and velocity correction on the staggered grid.
"""
function time_march!(param::Parameters, fields::Fields)

    T = param.Ttype
    u_in = maximum(abs, Array(fields.u_in))
    d = param.cylinder.d
    scale = T(0.5) * u_in^2 * d
    velocity_boundary!(param, fields.ux, fields.uy, fields.temp, fields)

    drag_stage1, lift_stage1 = zero(T), zero(T)
    drag_stage2, lift_stage2 = zero(T), zero(T)
    drag_stage3, lift_stage3 = zero(T), zero(T)

    for _ in 1:param.time.num_time_interval

        fields.ux_mem .= fields.ux
        fields.uy_mem .= fields.uy
        fields.temp_mem .= fields.temp

        fields.ibm_x .= T(0)
        fields.ibm_y .= T(0)

        step!(param, fields)

        drag_stage1 = sum(fields.ibm_x)
        lift_stage1 = sum(fields.ibm_y)
        fields.ibm_x .= T(0)
        fields.ibm_y .= T(0)

        step!(param, fields)

        # Update the fields using the RK3 formula
        fields.ux .= (T(3) * fields.ux_mem .+ fields.ux) / T(4)
        fields.uy .= (T(3) * fields.uy_mem .+ fields.uy) / T(4)
        fields.temp .= (T(3) * fields.temp_mem .+ fields.temp) / T(4)

        drag_stage2 = sum(fields.ibm_x)
        lift_stage2 = sum(fields.ibm_y)
        fields.ibm_x .= T(0)
        fields.ibm_y .= T(0)

        step!(param, fields)

        # Update the fields using the RK3 formula
        fields.ux .= (fields.ux_mem .+ T(2) * fields.ux) / T(3)
        fields.uy .= (fields.uy_mem .+ T(2) * fields.uy) / T(3)
        fields.temp .= (fields.temp_mem .+ T(2) * fields.temp) / T(3)

        drag_stage3 = sum(fields.ibm_x)
        lift_stage3 = sum(fields.ibm_y)

    end

    fields.C[1] = combine_rk3_ibm_force(drag_stage1, drag_stage2, drag_stage3, scale, T)
    fields.C[2] = combine_rk3_ibm_force(lift_stage1, lift_stage2, lift_stage3, scale, T)

end

end
