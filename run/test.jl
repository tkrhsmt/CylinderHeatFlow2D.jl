using CylinderHeatFlow2D
using CairoMakie
using CSV, DataFrames

Re = 100.0
α = 0.01

param = CylinderHeatFlow2D.Parameters(
    ((0.0, 8.0), (0.0, 4.0)),
    (256, 128),
    (0.0, 100.0),
    10000,
    10,
    1/Re,
    α,
    2.0, 2.0, 0.5,
)

field = CylinderHeatFlow2D.initialize_field(param)
mkpath("result")

field.ux .= 1.0 .+ 0.001 * randn(size(field.ux))
field.uy .= 1.0 .+ 0.001 * randn(size(field.uy))

nx, ny = param.space.num_grids

for time in 1:param.time.num_time_total÷param.time.num_time_interval
    println("Time step: ", time * param.time.num_time_interval, "/", param.time.num_time_total)
    CylinderHeatFlow2D.time_march!(param, field)

    ux = Array(field.ux)[4:nx+3, 4:ny+3]
    uy = Array(field.uy)[4:nx+3, 4:ny+3]
    temp = Array(field.temp)[4:nx+3, 4:ny+3]

    fig = Figure()
    ax = Axis(fig[1, 1], aspect = DataAspect())
    #heatmap!(ax, field.x, field.y, sqrt.(ux.^2 + uy.^2))
    heatmap!(ax, field.x, field.y, temp)
    save("result/result_$(lpad(time, 4, "0")).png", fig)
end
