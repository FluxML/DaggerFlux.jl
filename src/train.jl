using Dagger, DaggerMPI, DaggerFlux
using Flux
using Flux.Functors
using MPI
using Optimisers

function train!(loss, model, data, opt_state;
                nepochs::Integer=1, comm=MPI.COMM_WORLD)
    # Initialize MPI if currently uninitialized
    need_init = !MPI.Initialized()
    if need_init
        MPI.Init()
    end

    # Start with the same model everywhere
    model = MPI.bcast(model, comm, root=0)

    csz = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

    for epoch in 1:nepochs
        for (x, y) in data
            # Run the local model across the batch and collect gradients
            gs, _ = gradient(model, x) do model, x
                l = loss(model(x), y)
                @show "[$rank](Epoch: $epoch) Loss: $l"
                l
            end

            # Reduce gradients across ranks
            gs_red = Flux.Functors.fmap(gs) do x
                x isa AbstractArray || return x

                # Wrap this gradient array in a DArray
                dx = distribute(x, MPIParallelBlocks{ndims(x)}())

                # Average gradients in-place
                # TODO: Combine these
                DaggerMPI.reduce!(+, dx)
                map!(x->x ./ csz, dx, dx)

                return x
            end

            # Optimize the model with the (now uniform) gradients
            opt_state, model = Optimisers.update(opt_state, model, gs_red)
        end
    end

    # If we had to initialize MPI, we should undo that now
    if need_init
        MPI.Finalize()
    end

    # Return the final model
    return model
end
