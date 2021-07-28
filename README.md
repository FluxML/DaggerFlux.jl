# DaggerFlux.jl

This is currently an early stage integration between [`Dagger.jl`](https://github.com/JuliaParallel/Dagger.jl) and [`Flux.jl`](https://github.com/FluxML/Flux.jl) to allow for distributed computation of differentiation pipelines to use multiple workers, devices, GPUs etc. This package enables model parallelism for Flux models.

## Basic Usage

To see the package in action, we would have to start julia with multiple workers.

Also make sure that the workers have access to the environment and code that is going to be run. This is typically done with the help of the `exeflags` keyword in `addprocs`. Something like `addprocs(2, exeflags = "--project")` is usually enough. Please ensure that the environment has access to `DaggerFlux`.

```julia
julia> using DaggerFlux, Dagger, Flux, Zygote

julia> @everywhere function layer(x)
         @show myid()
         x
       end

julia> ip = rand(3,3);

julia> c = Chain(layer, layer, layer, layer)
Chain(layer, layer, layer, layer)

julia> dc = DaggerChain(c)
DaggerChain(Chain(layer, layer, layer, layer))

julia> dc(ip) # notice the output is a Dagger Thunk rather than an eager evaluation
Thunk[4](layer, (Thunk[3](layer, ...),))

julia> collect(dc(ip))
      From worker 2:    myid() = 2
      From worker 3:    myid() = 3
      From worker 2:    myid() = 2
      From worker 3:    myid() = 3
3×3 Matrix{Float64}:
 0.813575   0.828228  0.0630336
 0.0755053  0.215495  0.64503
 0.462957   0.345485  0.83312
```

Notice that the model was now evaluated across multiple workers.

## Flux models

This is basically the same as before, but we will demo how to differentiate through Flux models.

```julia
julia> y, back = Zygote.pullback((m,x) -> m(x), dc, ip)
(Thunk[135](layer, (Thunk[131](layer, ...),)), Zygote.var"#46#47"{typeof(∂(#11))}(∂(#11)))

julia> collect(y)
      From worker 3:    myid() = 3
      From worker 3:    myid() = 3
      From worker 2:    myid() = 2
      From worker 2:    myid() = 2
3×3 Matrix{Float64}:
 0.813575   0.828228  0.0630336
 0.0755053  0.215495  0.64503
 0.462957   0.345485  0.83312

julia> back(one.(y))
      From worker 2:    myid() = 2
      From worker 2:    myid() = 2
      From worker 3:    myid() = 3
      [...]
      From worker 2:    myid() = 2
      From worker 3:    myid() = 3
      From worker 2:    myid() = 2
((chain = (layers = (nothing, nothing, nothing, nothing),),), [1.0 1.0 1.0; 1.0 1.0 1.0; 1.0 1.0 1.0])
```

And now one can optimise over entire models!

Of course one can substitute our dummy model here with more routine models such as ResNet from Metalhead.jl. Here's a slightly simpler model for an example.

```julia
julia> m = Chain(Dense(2,2), Dense(2,2))
Chain(
  Dense(2, 2),                          # 6 parameters
  Dense(2, 2),                          # 6 parameters
)                   # Total: 4 arrays, 12 parameters, 304 bytes.

julia> dm = DaggerChain(m)
DaggerChain(Chain(Dense(2, 2), Dense(2, 2)))

julia> y, b = Zygote.pullback((m,x) -> m(x), dm, rand(Float32, 2
,2))
(Thunk[150](Dense(2, 2), (Thunk[149](Dense(2, 2), ...),)), Zygote.var"#46#47"{typeof(∂(#13))}(∂(#13)))

julia> b(one.(y))
((chain = (layers = ((weight = Float32[1.0398567 0.45392603; 0.4867683 0.21248773], bias = Float32[1.6065784, 0.75205684], σ = nothing), (weight = Float32[-1.247205 1.2783735; -1.247205 1.278
735], bias = Float32[2.0, 2.0], σ = nothing)),),), Float32[-0.14533046 -0.14533046; -0.58934844 -0.58934844])
```


Contributions welcome to the GitHub repository!
