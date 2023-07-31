using Dagger
using Flux, Zygote
using Zygote: @adjoint

struct DaggerChain
    chain::Chain
end

daglayer(f, args...) = Dagger.spawn((m,x...) -> m(x...), f, args...)
daglayer(par::Parallel, ip...) = Dagger.@spawn par.connection((daglayer(f, ip...) for f in par.layers)...)

function (dc::DaggerChain)(x)
    t = foldl(dc.chain.layers; init = x) do l1, l2
        daglayer(l2, l1)
    end
end

Flux.@functor DaggerChain

@adjoint function (dc::DaggerChain)(x...)
  thy, thb = dag_chain(dc.chain, x...)
  thy, Δ -> begin
    gm, gx = thb(Δ)
    ((chain = gm,), gx)
  end
end

@adjoint function Dagger.collect(th::Union{Thunk, Dagger.Chunk})
  d = Dagger.@spawn Zygote.pullback((m,x) -> m(x...), th.f, th.inputs)
  y, back = collect(d)
  y, Δ -> begin
    nt = Zygote.nt_nothing(th)
    gnt = NamedTuple{(:f, :inputs)}(back(Δ))
    (Zygote.accum(nt, gnt),)
  end
end

function dag_chain(c, ip...)
  # y = ip
  pb = Dagger.@spawn Zygote.pullback((m,x) -> m(x...), c[1], ip)
  thy = Dagger.@spawn getindex(pb, 1)
  back = Dagger.@spawn getindex(pb, 2)
  f = Dagger.@spawn c[1](ip...)
  backs = [back]
  for m in c[2:end]
    f = Dagger.@spawn m(f)
    pb = Dagger.@spawn Zygote.pullback((m,x) -> m(x), m, thy)
    thy = Dagger.@spawn getindex(pb, 1)
    back_ = Dagger.@spawn getindex(pb, 2)

    push!(backs, back_)
  end
  fetch(f), Δ -> makedag(backs, Δ)
end

function makedag(backs, Δ)
  b = Dagger.spawn((m,x) -> m(x), backs[end], Δ)
  out = [b]
  # reverse the elements of backs except the last one
  for b_ in backs[end-1:-1:1]
    g = Dagger.@spawn getindex(b, 2)
    b = Dagger.spawn((m,x) -> m(x), b_, g)

    push!(out, b)
  end
  cout = fetch(Dagger.spawn((xs...)->xs, out...))
  (((layers = Tuple(reverse(first.(cout))),)), cout[end][2]...)
end
