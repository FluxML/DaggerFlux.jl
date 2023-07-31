module DaggerFlux

using Dagger
using Flux, Zygote
using Zygote: @adjoint

export DaggerChain, dag_chain

include("core.jl")

end
