using Dagger
using Flux, Zygote
using Zygote: @adjoint 
using DaggerGPU, CUDA

struct DaggerChain
    chain::Chain
end

# TODO: Remove/adapt once https://github.com/JuliaParallel/Dagger.jl/pull/271 is merged
daglayer(f, args...) = delayed((m,x...) -> m(x...))(Dagger.tochunk(f, DaggerGPU.CuArrayDeviceProc(1, CUDA.device().handle, CUDA.uuid(CUDA.device()))), args...)
daglayer(par::Parallel, ip...) = delayed((x...) -> par.connection(x...))(daglayer(f, ip...) for f in par.layers)

function (dc::DaggerChain)(x)
    t = foldl(dc.chain.layers; init = x) do l1, l2
        # delayed(l2)(l1)
	daglayer(l2, l1)
    end
end

Flux.@functor DaggerChain

@adjoint function (dc::DaggerChain)(x...)
  thy, thb = dag_chain(dc.chain, x...)
  thy, Δ -> begin
    gm, gx = thb(Δ)
    ((chain = gm,), gx...)
  end
end

function reverse_graph(t::Dagger.Thunk, x...)
    pb = delayed(Zygote.pullback)(t.f, x...)
    for (idx, arg) in enumerate(t.inputs)
        @show typeof(arg)
        # reverse_graph(arg, pb[idx]...)
        # reverse_graph(arg.f, arg.inputs...)
        reverse_graph(arg, pb.inputs[idx]...)
    end
    return delayed() do args...
        reverse_graph.(t.inputs)
    end
end

reverse_graph(f, args...) = nothing # f(args...)

ispurethunk(th) = !any(x -> x isa Thunk, (th.inputs..., th.f))

@adjoint function Dagger.collect(th::Union{Thunk, Dagger.Chunk})
  d = delayed(Zygote.pullback)((m,x) -> m(x...), th.f, th.inputs)
  y, back = collect(d)
  y, Δ -> begin
    nt = Zygote.nt_nothing(th)
    gnt = NamedTuple{(:f, :inputs)}(back(Δ))
    (Zygote.accum(nt, gnt),)
  end
end

function dagger_train!(loss, ps, data, opt; cb = ()->())
    ps = Flux.Params(ps)
    cb = Flux.Optimise.runall(cb)
    batches = [
        delayed() do d
            Flux.gradient(ps) do
                loss(Flux.Optimise.batchmemaybe(d)...)
            end
        end(d) for d in data]
    redu = delayed(; single=1) do gs...
        for g in gs
            Flux.Optimise.update!(opt, ps, g)
            cb()
        end
    end
    reducer = redu(batches...)
    compute(reducer)
end

