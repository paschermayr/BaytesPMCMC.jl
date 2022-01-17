using BaytesPMCMC
using Documenter

DocMeta.setdocmeta!(BaytesPMCMC, :DocTestSetup, :(using BaytesPMCMC); recursive=true)

makedocs(;
    modules=[BaytesPMCMC],
    authors="Patrick Aschermayr <p.aschermayr@gmail.com>",
    repo="https://github.com/paschermayr/BaytesPMCMC.jl/blob/{commit}{path}#{line}",
    sitename="BaytesPMCMC.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://paschermayr.github.io/BaytesPMCMC.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Introduction" => "intro.md",
    ],
)

deploydocs(;
    repo="github.com/paschermayr/BaytesPMCMC.jl",
    devbranch="main",
)
