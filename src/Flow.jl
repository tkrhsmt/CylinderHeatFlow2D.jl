module Flow

export initialize_field

using ..Utils

function initialize_field(param::Parameters)
    field = Fields(param)
    nx, ny = param.space.num_grids
    x = Array(field.x)
    y = Array(field.y)
    cx, cy, r = param.cylinder.cx, param.cylinder.cy, param.cylinder.r
    I = param.Itype
    T = param.Ttype

    # Mark the solid region (cylinder) in the `solid` field
    solid = zeros(I, nx + 6, ny + 6)  # Include ghost cells
    for j in 1:ny
        for i in 1:nx
            if (x[i] - cx)^2 + (y[j] - cy)^2 <= r^2
                solid[i+3, j+3] = I(1)
            else
                solid[i+3, j+3] = I(0)
            end
        end
    end
    field.solid .= param.ArrayType(solid)

    field.u_in .= T(1.0)  # Set the inlet velocity (for example, uniform flow)
    field.ux .= T(1.0)   # Initialize velocity fields to ux
    field.uy .= T(0.0)   # Initialize velocity fields to uy
    field.temp .= T(0.0) # Initialize temperature field to 0
    field.temp_in .= param.cylinder.temp_wall # Set the inlet temperature to the wall temperature

    return field
end

end
