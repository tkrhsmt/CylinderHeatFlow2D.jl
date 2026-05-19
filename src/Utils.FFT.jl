module FFT

using CUDA
using FFTW

export plan_redft00_cuda, plan_redft00_fftw

"""
    plan_redft00_cuda(::Type{T}, dims)

Create a GPU DCT-I (FFTW.REDFT00) plan using cuFFT.

Returns a callable object:

    p(x)

where `x` is a CuArray.

Currently supports:
- 1D
- 2D

The implementation uses even extension + cuFFT.
"""
function plan_redft00_cuda(::Type{T}, dims::NTuple{N,Int}) where {T<:AbstractFloat,N}

    # extended sizes
    edims = ntuple(i -> 2 * dims[i] - 2, N)

    # workspace
    buf = CUDA.zeros(Complex{T}, edims)

    # FFT plan
    fft_plan = plan_fft!(buf)

    if N == 1
        nx = dims[1]
        reflected_x = (nx - 1):-1:2

        return function (x::CuArray{T,1})

            @assert size(x) == dims

            @views begin
                buf[1:nx] .= x
                buf[(nx+1):end] .= x[reflected_x]
            end

            fft_plan * buf

            return real(buf[1:nx])
        end
    elseif N == 2
        nx, ny = dims
        reflected_x = (nx - 1):-1:2
        reflected_y = (ny - 1):-1:2

        return function (x::CuArray{T,2})

            @assert size(x) == dims

            @views begin
                buf[1:nx, 1:ny] .= x
                buf[(nx+1):end, 1:ny] .= x[reflected_x, :]
                buf[:, (ny+1):end] .= buf[:, reflected_y]
            end

            fft_plan * buf

            return real(buf[1:nx, 1:ny])
        end
    end

    error("Only 1D and 2D currently supported")
end

function plan_redft00_fftw(::Type{T}, dims::NTuple{N,Int}) where {T<:AbstractFloat,N}
    plan = FFTW.plan_r2r(
        zeros(T, dims...), FFTW.REDFT00;
        flags=FFTW.PATIENT,
        timelimit=Inf
    )

    function execute(x::Array{T,N})
        return plan * x
    end

    return execute
end

end
