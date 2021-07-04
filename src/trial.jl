function dag_chain(c::Chain, ip)
  outs = []
  pbs = []
  # y = ip
  thpb = delayed(Zygote.pullback)((m,x) -> m(x), c[1], ip)
  thy = delayed(getindex)(thpb, 1)

  for m in c[2:end]
    @show m
    # thy = delayed(getindex)(thpb, 1)
    thpb = delayed(Zygote.pullback)((m,x) -> m(x), m, thy)
    thy = delayed(getindex)(thpb, 1)
    push!(pbs, thpb)
  end
  thy, thpb
end

dag_chain(f, x...) = delayed(Zygote.pullback)(f, (x...))
