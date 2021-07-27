using Documenter, DaggerFlux

DocMeta.setdocmeta!(DaggerFlux, :DocTestSetup, :(using DaggerFlux); recursive = true)

makedocs(modules = [DaggerFlux],
         doctest = VERSION == v"1.6",
         sitename = "DaggerFlux.jl",
         pages = ["Home" => "index.md",
                  "API" => "api.md"],
         format = Documenter.HTML(
             analytics = "UA-36890222-9",
             # assets = ["assets/flux.css"],
             prettyurls = get(ENV, "CI", nothing) == "true"),
         )

deploydocs(repo = "github.com/DhairyaLGandhi/DaggerFlux.jl.git",
           target = "build",
           push_preview = true)
