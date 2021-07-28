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

function f_rev(f, ystart, th::Dagger.Thunk, x...)
  @show "in f_rev"
  @warn ystart
  if isleaf(th) # !any(x -> x isa Thunk, Dagger.inputs(th))
    y, pb = Zygote.pullback(th.f, Dagger.inputs(th)...)
    @info "in the leaf f: $f, $(th.f)"
    # @show th.f
    @show y, pb(1.f0)
    rev = delayed(f)(delayed(getindex)(delayed(pb)(1.f0), 1))
    return y, rev
  else
    @info "in the non leaf f: $f, $(th.f), $ystart"
    # @show f, th.f
    # @show curr_in = collect(Dagger.inputs(th)[1])
    @show th.f, x, ystart
    yn, pbn = Zygote.pullback(th.f, curr_in)
    # yn, pbn = Zygote.pullback(x.f, curr_in)
    @show yn, ystart
    rev = delayed(f)(delayed(getindex)(delayed(pbn)(curr_in), 1))
    return yn, rev
  end
  y, rev
end

# f_rev(th::Thunk, x) = f_rev(identity, 1, th, x)
# f_rev(x::AbstractArray, t...) = x, _ -> t
f_rev(x...) = (x[1], _ -> collect(x[1]))# (nothing, nothing)

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


## From dflux.jl

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
