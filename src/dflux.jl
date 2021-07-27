using Dagger
using Flux, Zygote
using Zygote: @adjoint 

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

# @adjoint function (dc::DaggerChain)(x)
#     function back(Δ)
#       t = foldr(dc.chain.layers; init = x) do l1, l2
#         @show l1, l2
#         delayed(l1)(l2)
#       end
#       # @show Δ
#       ((chain = t,), t,)
#     end
#     return dc(x), back
# end

# ip = rand(Float32, 1, 1)
# m = Chain(Dense(1,2), Dense(2,2), Dense(2,3))
# dc = DaggerChain(m)

@adjoint function (dc::DaggerChain)(x)
  thy, thb = dag_chain(dc.chain, x)
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

function reverse_graph2(f, t::Dagger.Thunk, x...)
  # if ispurethunk(
  pb = delayed(f)(x...)
end

# @adjoint function collect(t::Dagger.Thunk)
#     result = collect(t)
#     pb = x->collect(reverse_graph(t, x))
#     return (result, pb)
# end

Zygote.@adjoint function Dagger.collect(th::Union{Thunk, Dagger.Chunk})
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

struct Res{T,S}
  f::T
  y::S
end


# function at_node(ctx, curr, depth)
# end

function dfs2(current_node,
             next_nodes,
             at_node,
             pre::Bool = true,
             depth::Integer = 0)
  @show pre
  if pre
    at_node(current_node, depth)
  end

  for child_node in next_nodes(current_node)
    dfs2(child_node, next_nodes, at_node, pre, depth + 1)
  end

  if !pre
    # at_node(identity, 1, current_node, depth)
    at_node(current_node, depth)
  end
  # @show fv
end


function bfs(start_node,
	     next_nodes,
	     at_node)
  to_process = [] # Array(Any, 0)
  depths = Int[] # Array(Int, 0)
  
  push!(to_process, start_node)
  push!(depths, 0)
  
  while !isempty(to_process)
    current_node = popfirst!(to_process)
    depth = popfirst!(depths)
    
    at_node(current_node, depth)
    
    for child_node in next_nodes(current_node)
      push!(to_process, child_node)
      push!(depths, depth + 1)
    end
  end
end


function dfs(current_node,
             next_nodes,
             at_node,
             pre::Bool = true,
             depth::Integer = 0,)
             # rev = Res(nothing, nothing))
  @show pre
  f, curr_in = if pre
    @show "in not loop"
    y, curr_in = at_node(identity, nothing, current_node, depth)
    # error("out of it", current_node)
  else
    y, curr_in = identity, 1
  end
  @warn isleaf(current_node)
  for child_node in next_nodes(current_node)
    @show child_node
    # global f, curr_in
    # @error collect(curr_in)
    y, curr_in = dfs(child_node, next_nodes, (x...) -> x[1] == identity ? at_node(curr_in, y, x[3:end]...) : at_node(curr_in, y, x...), pre, depth + 1)
  end

  if !pre
    # at_node(identity, 1, current_node, depth)
    y, curr_in = at_node(curr_in, y, current_node, depth)
  end
  # @show fv
  return y, curr_in
end

function f_rev(f, args...; cache = IdDict())
  rev = f_rev(args..., cache = cache)
  if isempty(rev)
    back = identity
    y = f
  else
    pb = delayed(Zygote.pullback)((m,x) -> m(x), f, rev[1])
    b = delayed(getindex)(pb, 2)
    # pb1(pb2(one.(y1))[2])
    Δ = ones(Float32, 2)
    b_ = delayed((m,x) -> m(x))(b, Δ)
    b__ = delayed(getindex)(b_, 2)
    back = delayed((m,x) -> m(x))(cache[rev[1].f], b__)
    cache[f] = back
    y = delayed((m,x) -> m(x))(f, rev[1])
  end
  y, back
end


function f_rev(th::Thunk, args...; cache = IdDict())
  if ispurethunk(th) # TODO: rename to isleafthunk
    pb = delayed(Zygote.pullback)((m,x) -> m(x), th.f, th.inputs...)
    b = delayed(getindex)(pb, 2)
    cache[th.f] = b
    th, b
  else
    i = f_rev(th.inputs..., cache = cache)
    f_rev(th.f, i[1], cache = cache)
  end
end

f_rev(args...; cache = IdDict()) = Tuple(get!(cache, x, nothing) for x in args)


########## HACKS ###########

function f1(c::Chain, ip)
  outs = []
  pbs = []
  y = ip
  for m in c
    @show m
    y, pb = Zygote.pullback((m,x) -> m(x), m, y)
    push!(outs, y)
    push!(pbs, pb)
  end
  outs, pbs, c(ip)
end

function g(pbs)
  s = delayed(getindex)(
        delayed(pbs[end])(1.f0),
      2)
  @show collect(s)
  for p in reverse(pbs[1:end-1])
    s = delayed(getindex)(delayed(p)(s), 2)
  end
  # s2 = delayed(pbs[end-2])(s)
  s.inputs[1]
end

# g(f1(m, ip)[2]) |> collect
