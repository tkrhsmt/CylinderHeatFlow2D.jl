using CylinderHeatFlow2D
using Documenter

DocMeta.setdocmeta!(CylinderHeatFlow2D, :DocTestSetup, :(using CylinderHeatFlow2D); recursive=true)

makedocs(;
    modules=[CylinderHeatFlow2D],
    authors="T. Hashimoto",
    sitename="CylinderHeatFlow2D.jl",
    format=Documenter.HTML(;
        canonical="https://tkrhsmt.github.io/CylinderHeatFlow2D.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/tkrhsmt/CylinderHeatFlow2D.jl",
    devbranch="main",
)
