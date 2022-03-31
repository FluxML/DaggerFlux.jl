function dag_chain(c, ip...)
  # y = ip
  pb = delayed(Zygote.pullback)((m,x) -> m(x...), c[1], ip)
  thy = delayed(getindex)(pb, 1)
  back = delayed(getindex)(pb, 2)
  f = delayed(c[1])(ip...)
  backs = [back]
  for m in c[2:end]
    f = delayed(m)(f)
    pb = delayed(Zygote.pullback)((m,x) -> m(x), m, thy)
    thy = delayed(getindex)(pb, 1)
    back_ = delayed(getindex)(pb, 2)

    # replicate back_(back(y)[2])
    # b = delayed_call(back_, thy)
    # b_ = delayed(getindex)(b, 2)
    # back = delayed_call(back_, b_)
    push!(backs, back_)
  end
  collect(f), Δ -> makedag(backs, Δ)
end

delayed_call(f, args) = delayed((m,x) -> m(x))(f, args)

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

# dag_chain(f, x...) = delayed(Zygote.pullback)(f, (x...))
