module DaggerFlux

using Dagger
using Flux, Zygote
using Zygote: @adjoint 

export dfs, dfs2, f_rev
export f2, dag_chain

struct DaggerChain
    chain::Chain
end

function (dc::DaggerChain)(x)
    t = foldl(dc.chain.layers; init = x) do l1, l2
        # @show l2
        delayed(l2)(l1)
    end
    # collect(t)
end

# include("treewalk.jl")
include("dflux.jl")

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

function dowalk(th::Thunk, r = Thunk[])
  node = th
  # if isleaf(node)
  #   @show node
  #   push!(r, node)
  # end
  next_nodes = node.inputs
  for next_node in next_nodes
    @show next_node
    dowalk(next_node, r)
  end

  push!(r, node)
  r
end

dowalk(x, r) = nothing
dothing = identity

function isleaf(th::Thunk)
  !any(x -> x isa Thunk, Dagger.inputs(th))
end
isleaf(x::Union{Tuple,AbstractVector}) = any(isleaf, x)
isleaf(x) = true

ip = rand(Float32, 1, 1)
m = Chain(Dense(1,2), Dense(2,2), Dense(2,3))
dc = DaggerChain(m)

export ip, m, dc
end
