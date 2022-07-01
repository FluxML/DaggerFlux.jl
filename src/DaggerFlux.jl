module DaggerFlux

using Dagger
using Flux, Zygote
using Zygote: @adjoint 

export DaggerChain, dag_chain

# include("treewalk.jl")
include("dflux.jl")
include("dag_chain.jl")

function dowalk(th::Thunk, r = Thunk[])
  node = th
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

end
