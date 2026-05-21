module Parallel

using KernelAbstractions
using ..Utils

export foraxes

function foraxes(f, ::KernelAbstractions.CPU, ::Tuple, axes::Tuple)
    Threads.@threads for j in 1:axes[2]
        for i in 1:axes[1]
            @inline f(i, j)
        end
    end
end

function foraxes(f, backend::KernelAbstractions.GPU, groupsize::Tuple, axes::Tuple)
    (@kernel unsafe_indices = true function k()
        ix, iy = @index(Global, NTuple)
        if ix <= axes[1] && iy <= axes[2]
            @inline f(ix, iy)
        end
    end)(backend, groupsize)(; ndrange=(axes[1], axes[2]))
    KernelAbstractions.synchronize(backend)
end

end
