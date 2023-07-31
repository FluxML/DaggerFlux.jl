function dagger_train!(loss, ps, data, opt; cb = ()->())
    ps = Flux.Params(ps)
    cb = Flux.Optimise.runall(cb)
    batches = [
        Dagger.spawn(d) do _d
            Flux.gradient(ps) do
                loss(Flux.Optimise.batchmemaybe(_d)...)
            end
        end for d in data]
    reducer = Dagger.spawn(; scope=Dagger.scope(worker=1)) do gs...
        for g in gs
            Flux.Optimise.update!(opt, ps, g)
            cb()
        end
    end(batches...)
    compute(reducer)
end
