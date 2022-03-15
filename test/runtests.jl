using DaggerFlux, Dagger
using Flux, Metalhead
using Flux.Zygote
using Distributed
using CUDA
using Test

function compare(y::Tuple, ŷ)
  foreach((a,b) -> compare(a, b), y, ŷ)
end

function compare(y::NamedTuple, ŷ)
  foreach((a,b) -> compare(a, b), y, ŷ)
end

function compare(a::AbstractArray, b)
  @testset "Arrays" begin
    @test a ≈ b
  end
end

function compare(a::Base.RefValue, b::Base.RefValue)
  compare(a[], b[])
end

function compare(::Nothing, ::Nothing)
  @testset "Nothings" begin
    @test true
  end
end

function compare(a, b)
  @testset "Generic" begin
    @test a ≈ b
  end
end

@testset "ResNet test" begin
  # addprocs(2, exeflags = "--project=.")
  resnet = ResNet50()
  ip = rand(Float32, 224, 224, 3, 1)
  y_, b_ = Zygote.pullback((m,x) -> m(x), resnet.layers, ip)
  g2 = b_(ones(Float32, 1000, 1))

  resy, resbacks = DaggerFlux.dag_chain(resnet.layers, ip)
  resgs = resbacks(ones(Float32, 1000, 1))

  # @test collect(resy) ≈ y_
  compare(g2, resgs)
end

@testset "DaggerChain tests" begin
  m = Chain(Dense(2,2), Dense(2,2))
  dm = DaggerChain(m)
  ip = rand(Float32, 2, 4)

  thy, thback = Zygote.pullback((m,x) -> m(x), dm, ip)
  @test collect(thy) ≈ m(ip)
  y, back = Zygote.pullback((m,x) -> m(x), m, ip)

  Δ = ones(Float32, 2,4)
  thgs = thback(Δ)
  gs = back(Δ)
  compare(thgs[1].chain, gs[1])
  @test thgs[2] ≈ gs[2]
end
