module ParamField

using KernelAbstractions
using FFTW
using CUDA
using ..Utils

export SpaceParameters, TimeParameters, FDMParameters
export FFTPoissonSolver
export Parameters, Fields
export AbstractFFTBackendPlan, FFTWPlanBackend, CUFFTPlanBackend

abstract type AbstractFFTBackendPlan end
"""Pressure-solver backend based on `FFTW` plans for CPU arrays."""
struct FFTWPlanBackend <: AbstractFFTBackendPlan end
struct CUFFTPlanBackend <: AbstractFFTBackendPlan end

# ============================================================================================

"""
    SpaceParameters(xy_range, num_grids)

Spatial discretization metadata for the 2D domain.

The constructor derives the domain lengths, total number of cells, and uniform
grid spacing from the coordinate range and the number of cells.
"""
struct SpaceParameters{T<:AbstractFloat,I<:Integer}
    xy_range::NTuple{2,NTuple{2,T}}
    num_grids::NTuple{2,I}
    n_total::I
    lengths::NTuple{2,T}
    dx::T
    dy::T

    function SpaceParameters(
        xy_range::NTuple{2,NTuple{2,T}},
        num_grids::NTuple{2,I},
    ) where {T<:AbstractFloat,I<:Integer}
        nx, ny = num_grids
        Lx = xy_range[1][2] - xy_range[1][1]
        Ly = xy_range[2][2] - xy_range[2][1]
        dx = Lx / T(nx)
        dy = Ly / T(ny)

        new{T,I}(xy_range, num_grids, nx * ny, (Lx, Ly), dx, dy)
    end
end

# ============================================================================================

"""
    TimeParameters(t_range, num_time_total, num_time_interval=num_time_total)

Time discretization metadata for a simulation run.

`num_time_total` determines the global time step `dt`, and
`num_time_interval` sets how many time steps are advanced by one call to
`time_march!`.
"""
struct TimeParameters{T<:AbstractFloat,I<:Integer}
    t_range::Tuple{T,T}
    num_time_total::I
    num_time_interval::I
    dt::T

    function TimeParameters(
        t_range::Tuple{T,T},
        num_time_total::I,
        num_time_interval::I=num_time_total,
    ) where {T<:AbstractFloat,I<:Integer}
        dt = (t_range[2] - t_range[1]) / T(num_time_total)

        new{T,I}(t_range, num_time_total, num_time_interval, dt)
    end
end

# ============================================================================================

"""
    FFTPoissonSolver(nx, ny, lx, ly, backend, ArrayType)

FFT-based Poisson solver used in the pressure projection step.
"""
struct FFTPoissonSolver{T<:AbstractFloat,I<:Integer}
    backend::AbstractFFTBackendPlan
    plan_f # Plan for pressure
    kx::AbstractArray
    ky::AbstractArray
    inv_laplacian::AbstractArray
    inv_norm::T
    fftw_num_threads::I
    kxn::I
    kyn::I

    function FFTPoissonSolver(
        nx::I,
        ny::I,
        lx::T,
        ly::T,
        backend::FFTWPlanBackend,
        fftw_num_threads::I=I(Threads.nthreads()),
    ) where {T<:AbstractFloat,I<:Integer}

        kx = T.([0:nx-1;] .* 0.5 * (2π / lx))
        ky = T.([0:ny-1;] .* 0.5 * (2π / ly))
        dx = lx / T(nx)
        dy = ly / T(ny)
        nthreads_fftw = max(I(1), fftw_num_threads)
        kxn = I(length(kx))
        kyn = I(length(ky))
        inv_laplacian = zeros(T, nx, ny)

        for j in 1:ny
            θy = ky[j] * dy
            λy = (T(730) - T(783) * cos(θy) + T(54) * cos(T(2) * θy) - cos(T(3) * θy)) / (T(288) * dy^2)
            for i in 1:nx
                θx = kx[i] * dx
                λx = (T(730) - T(783) * cos(θx) + T(54) * cos(T(2) * θx) - cos(T(3) * θx)) / (T(288) * dx^2)
                inv_laplacian[i, j] = ifelse(i == 1 && j == 1, T(0.0), -inv(λx + λy))
            end
        end

        inv_norm = inv(T(2 * (nx - 1)) * T(2 * (ny - 1)))
        FFTW.set_num_threads(Int(nthreads_fftw))

        planf = Utils.FFT.plan_redft00_fftw(T, (nx, ny))

        new{T,I}(backend, planf, kx, ky, inv_laplacian, inv_norm, nthreads_fftw, kxn, kyn)
    end

    function FFTPoissonSolver(
        nx::I,
        ny::I,
        lx::T,
        ly::T,
        backend::CUFFTPlanBackend,
        fftw_num_threads::I=I(Threads.nthreads()),
    ) where {T<:AbstractFloat,I<:Integer}

        kx = T.([0:nx-1;] .* 0.5 * (2π / lx))
        ky = T.([0:ny-1;] .* 0.5 * (2π / ly))
        dx = lx / T(nx)
        dy = ly / T(ny)
        nthreads_fftw = max(I(1), fftw_num_threads)
        kxn = I(length(kx))
        kyn = I(length(ky))
        inv_laplacian = zeros(T, nx, ny)

        for j in 1:ny
            θy = ky[j] * dy
            λy = (T(730) - T(783) * cos(θy) + T(54) * cos(T(2) * θy) - cos(T(3) * θy)) / (T(288) * dy^2)
            for i in 1:nx
                θx = kx[i] * dx
                λx = (T(730) - T(783) * cos(θx) + T(54) * cos(T(2) * θx) - cos(T(3) * θx)) / (T(288) * dx^2)
                inv_laplacian[i, j] = ifelse(i == 1 && j == 1, T(0.0), -inv(λx + λy))
            end
        end
        kx = CUDA.CuArray(kx)
        ky = CUDA.CuArray(ky)
        inv_laplacian = CUDA.CuArray(inv_laplacian)

        inv_norm = inv(T(2 * (nx - 1)) * T(2 * (ny - 1)))

        planf = Utils.FFT.plan_redft00_cuda(T, (nx, ny))

        new{T,I}(backend, planf, kx, ky, inv_laplacian, inv_norm, nthreads_fftw, kxn, kyn)
    end
end

# ============================================================================================

"""
    FDMParameters(ν, pressure_solver)

Finite-difference and physical parameters used by the solver.
"""
struct FDMParameters{T<:AbstractFloat,PS<:FFTPoissonSolver}
    ν::T
    α::T
    pressure_solver::PS
end

# ============================================================================================

struct CylinderParameters{T<:AbstractFloat}
    cx::T
    cy::T
    r::T
    d::T
    temp_wall::T
    temp_cylinder::T

    function CylinderParameters(cx::T, cy::T, r::T, temp_wall::T, temp_cylinder::T) where {T<:AbstractFloat}
        d = T(2) * r
        new{T}(cx, cy, r, d, temp_wall, temp_cylinder)
    end
end

# ============================================================================================

struct Parameters{T<:AbstractFloat,I<:Integer,AT}
    space::SpaceParameters{T,I}
    time::TimeParameters{T,I}
    fdm::FDMParameters{T,FFTPoissonSolver{T,I}}
    cylinder::CylinderParameters{T}
    groupsize::NTuple{2,I}
    fftw_num_threads::I
    dev::KernelAbstractions.Backend
    Ttype::Type{T}
    Itype::Type{I}
    ArrayType::AT
    debug::Bool

    function Parameters(
        xy_range::NTuple{2,NTuple{2,T}},
        num_grids::NTuple{2,I},
        t_range::Tuple{T,T},
        num_time_total::I,
        num_time_interval::I,
        ν::T,
        α::T,
        cx::T,
        cy::T,
        r::T;
        temp_wall::T=T(1.0),
        temp_cylinder::T=T(10.0),
        groupsize::NTuple{2,I}=(I(16), I(16)),
        fftw_num_threads::I=I(Threads.nthreads()),
        dev::KernelAbstractions.Backend=KernelAbstractions.CPU(),
        fftplan_backend::AbstractFFTBackendPlan=FFTWPlanBackend(),
        ArrayType::Type=Array,
        debug::Bool=false,
    ) where {T<:AbstractFloat,I<:Integer}

        space = SpaceParameters(xy_range, num_grids)
        time = TimeParameters(t_range, num_time_total, num_time_interval)
        pressure_solver = FFTPoissonSolver(
            num_grids[1],
            num_grids[2],
            xy_range[1][2] - xy_range[1][1],
            xy_range[2][2] - xy_range[2][1],
            fftplan_backend,
            fftw_num_threads,
        )
        fdm = FDMParameters(ν, α, pressure_solver)
        cylinder = CylinderParameters(cx, cy, r, temp_wall, temp_cylinder)

        new{T,I,typeof(ArrayType)}(
            space,
            time,
            fdm,
            cylinder,
            groupsize,
            pressure_solver.fftw_num_threads,
            dev,
            T,
            I,
            ArrayType,
            debug,
        )

    end
end

# ============================================================================================

struct Fields{A}
    x::A
    y::A
    ux::A
    uy::A
    p::A
    temp::A
    ux_s::A
    uy_s::A
    temp_s::A
    rhs::A
    solid::A
    u_in::A
    temp_in::A
    ux_mem::A
    uy_mem::A
    temp_mem::A
    ibm_x::A
    ibm_y::A
    C::AbstractArray

    function Fields(param::Parameters{T,I}) where {T<:AbstractFloat,I<:Integer}
        nx, ny = param.space.num_grids
        A = param.ArrayType

        x_range = range(
            param.space.xy_range[1][1] + param.space.dx / T(2),
            param.space.xy_range[1][2] - param.space.dx / T(2);
            length=Int(nx),
        )
        y_range = range(
            param.space.xy_range[2][1] + param.space.dy / T(2),
            param.space.xy_range[2][2] - param.space.dy / T(2);
            length=Int(ny),
        )

        x = A(collect(T, x_range))
        y = A(collect(T, y_range))
        dims = (Int(nx) + 6, Int(ny) + 6)
        zeros2() = A(zeros(T, dims))
        rhs = A(zeros(T, nx, ny))
        u_in = A(zeros(T, ny))
        temp_in = A(zeros(T, ny))
        solid = A(zeros(I, nx + 6, ny + 6))
        C = Array(zeros(T, 2))

        new{A}(x, y, zeros2(), zeros2(), zeros2(), zeros2(), zeros2(), zeros2(), zeros2(), rhs, solid, u_in, temp_in, zeros2(), zeros2(), zeros2(), zeros2(), zeros2(), C)
    end
end

end
