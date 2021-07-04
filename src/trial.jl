function dag_chain(c::Chain, ip)
  # outs = []
  # pbs = []
  # y = ip
  thpb = delayed(Zygote.pullback)((m,x) -> m(x), c[1], ip)
  # thy = delayed(getindex)(thpb, 1)
  # outs = [thy]
  # pbs = [thpb]
  pb = delayed(getindex)(thpb, 2)
  # backval = delayed((m,x) -> m(x))(pb, thy)
  thy = delayed(c[1])(ip)
  y, back1 = Zygote.pullback((m,x) -> m(x), c[1], ip)
  backs = Any[back1]
  # pbs = [backval]
  for m in c[2:end]
    @show m
    # thy = delayed(getindex)(thpb, 1)
    thpb = delayed(Zygote.pullback)((m,x) -> m(x), m, thy)
    y, back = Zygote.pullback((m,x) -> m(x), m, y)
    # thy = delayed(getindex)(thpb, 1)
    thy = delayed(m)(thy)
    pb = delayed(getindex)(thpb, 2)
    backval = delayed((m,x) -> m(x))(pb, thy)
    pushfirst!(backs, back)
    # push!(pbs, backval)
  end
  thy, Δ -> makedag(backs, Δ) # , pbs # thpb
end

dag_chain(f, x...) = delayed(Zygote.pullback)(f, (x...))

function makedag(backs, Δ)
  b = delayed(first(backs))(Δ)
  out = [b]
  for b_ in backs[2:end]
    b = delayed(b_)(delayed(getindex)(b, 2))
    push!(out, b)
  end
  ∂m, ∂x = collect(out[end])
  ((layers = (∂m, reverse!(first.(collect.(out[begin:end-1])))...),), ∂x)
end
