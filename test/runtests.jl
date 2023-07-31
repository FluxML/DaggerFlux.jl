using DaggerFlux, Dagger
using Flux, Metalhead
using Flux.Zygote
using Distributed
using CUDA
using Test
CUDA.allowscalar(false)

function compare(y::Tuple, ŷ)
  foreach((a,b) -> compare(a, b), y, ŷ)
end

function compare(y::NamedTuple, ŷ)
  foreach((a,b) -> compare(a, b), y, ŷ)
end

function compare(a::AbstractArray{<:Number}, b)
  @testset "Arrays" begin
    @test a ≈ b
  end
end

function compare(y::AbstractVector{<:Union{NamedTuple, Nothing}}, ŷ)
    foreach((a,b) -> compare(a, b), y, ŷ)
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
  resnet = ResNet(50)
  ip = rand(Float32, 224, 224, 3, 1) |> gpu
  y_, b_ = Zygote.pullback((m,x) -> m(x), gpu(resnet.layers), ip)
  Δ = ones(Float32, 1000, 1) |> gpu
  g2 = b_(Δ)

  resy, resbacks = DaggerFlux.dag_chain(gpu(resnet.layers), ip)
  resgs = resbacks(Δ)
  # @test collect(resy) ≈ y_
  compare(g2, resgs)
end

@testset "DaggerChain tests" begin
  m = Chain(Dense(2,2), Dense(2,2)) |> gpu
  dm = DaggerChain(m) |> gpu
  ip = rand(Float32, 2, 4) |> gpu

  thy, thback = Zygote.pullback((m,x) -> m(x), dm, ip)
  @test collect(thy) ≈ m(ip)
  y, back = Zygote.pullback((m,x) -> m(x), m, ip)

  Δ = ones(Float32, 2,4) |> gpu
  thgs = thback(Δ)
  gs = back(Δ)
  compare(thgs[1].chain, gs[1])
  @test thgs[2] ≈ gs[2]

  # dag_chain
  y_, b_ = Zygote.pullback((m,x) -> m(x), m, ip)
  _gs = b_(Δ)

  resy, resbacks = DaggerFlux.dag_chain(m, ip)
  th_gs = resbacks(Δ)
  compare(_gs, th_gs)
end
