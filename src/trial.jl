function dag_chain(c::Chain, ip)
  # y = ip
  pb = delayed(Zygote.pullback)((m,x) -> m(x), c[1], ip)
  thy = delayed(getindex)(pb, 1)
  back = delayed(getindex)(pb, 2)
  # back = delayed((m,x) -> m(x))(back_, thy)
  backs = [back]
  ys = [thy]
  for m in c[2:end]
    @show m
    # thy = delayed(getindex)(thpb, 1)
    pb = delayed(Zygote.pullback)((m,x) -> m(x), m, thy)
    thy = delayed(getindex)(pb, 1)
    back_ = delayed(getindex)(pb, 2)
    # back_(back(y)[2])
    b = delayed((m,x) -> m(x))(back_, thy)
    b_ = delayed(getindex)(b, 2)
    back = delayed((m,x) -> m(x))(back_, b_)
    push!(backs, back_)
    push!(ys, thy)
  end
  ys[end], Δ -> makedag(backs, Δ)
end

function makedag(backs, Δ)
  b = delayed((m,x) -> m(x))(backs[end], Δ)
  out = [b]
  # reverse the elements of backs except the last one
  for b_ in backs[end-1:-1:1]
    g = delayed(getindex)(b, 2)
    b = delayed((m,x) -> m(x))(b_, g)

    push!(out, b)
  end
  cout = collect.(out)
  ((layers = Tuple(reverse(first.(cout))),)), cout[end][2]
end

dag_chain(f, x...) = delayed(Zygote.pullback)(f, (x...))
