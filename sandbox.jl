using BenchmarkTools
using SoftGlobalScope


# l, pb = Flux.Zygote.pullback(θ) do
l, pb = Flux.Tracker.forward(θ) do
   loss(xs, ys)
end
dldθ = pb(1.0f0)
Flux.Optimise.update!(optimiser, θ, dldθ)
@show l

loss(xs,ys)

(x,y) = rand(3,1,5), rand(3,1,5)

@btime cat($x,$y; dims=2)



hs = las.listen(xs)

f(a) = cat(a...; dims=3)
g(a) = reduce((x, y) -> cat(x, y; dims=3), a)

@btime f($hs)
@btime g($hs)


ψHs = las.attention_ψ.(hs)

O = gpu(zeros(Float32, size(las.state.decoding, 1), batch_size))
las.state.decoding = las.spell([las.state.decoding; las.state.prediction; las.state.context]) .+ O
dim_out = size(las.state.prediction, 1)

ϕSᵢᵀ = permutedims(las.attention_ϕ(las.state.decoding)) # workaround for bug in encountered during training
# compute attention context
Eᵢs = diag.(Ref(ϕSᵢᵀ) .* ψHs)
αᵢs = softmax(vcat(Eᵢs'...))

softmax(hcat(Eᵢs...)')

f(a) = softmax(vcat(a'...))
g(a) = softmax(reduce(vcat, a'))
h(a) = softmax(hcat(a...)')

f(Eᵢs) == h(Eᵢs)

@btime f($Eᵢs);
@btime g($Eᵢs);
@btime h($Eᵢs);




# load data
X, y,
Xs_train, ys_train, maxTs_train,
# Xs_test, ys_test, maxTs_test,
Xs_eval, ys_eval, maxT_eval,
Xs_val, ys_val, maxT_val =
let batch_size = 77, val_set_size = 32
   JLD2.@load "/Users/aza/Projects/LAS/data/TIMIT/TIMIT_MFCC/data_test.jld" Xs ys

   ys_val = ys[1:val_set_size]
   maxT_val = maximum(length, ys_val)
   Xs_val = batch_inputs!(Xs[1:val_set_size], maxT_val)

   Xs_test, ys_test, maxTs_test = batch!(Xs[(val_set_size+1):end], ys[(val_set_size+1):end], batch_size)

   JLD2.@load "/Users/aza/Projects/LAS/data/TIMIT/TIMIT_MFCC/data_train.jld" Xs ys
   X, y = first(Xs), first(ys)
   Xs_train, ys_train, maxTs_train = batch!(Xs, ys, batch_size)

   eval_idxs = sample(eachindex(ys), val_set_size; replace=false)
   ys_eval = ys[eval_idxs]
   maxT_eval = maximum(length, ys_eval)
   Xs_eval = batch_inputs!(Xs[eval_idxs], maxT_eval)

   X, y,
   Xs_train, ys_train, maxTs_train,
   # Xs_test, ys_test, maxTs_test,
   Xs_eval, ys_eval, maxT_eval,
   Xs_val, ys_val, maxT_val
end

xs = Xs_train[end]
ys = ys_train[end]
maxT = maxTs_train[end]


1.0f0

length(θ)

θ = Flux.params(m)

loss(m::Flux.Recur{Flux.LSTMCell{Array{Float32,2},Array{Float32,1}}}, xs, ys) = sum(sum(m.(xs)))


m = LSTM(39, 61)
m isa Flux.Recur{Flux.LSTMCell{T}} where T

struct Model{T}
   f::T
end
@functor Model

m = Model(Chain(LSTM(39, 61), logsoftmax))


function (m::Model)(xs::AbstractVector{<:AbstractMatrix})
   Ŷs = cat(reshape.(m.f.(xs), 61, 1, :)...; dims=2)
   reset!(m)
   return Ŷs
end


function loss(xs::AbstractVector{<:AbstractMatrix{<:Real}}, ys::AbstractVector{<:AbstractVector{<:Integer}})::Real
   Ŷs = m(xs)
   x, y, z = size(Ŷs)
   colsrng = range(0; step=x, length=y)
   slicesrng = range(0; step=x*y, length=z)
   # true_linindices = vcat([y .+ colsrng[eachindex(y)] .+ slicesrng[n] for (n, y) ∈ enumerate(ys)]...)
   true_linindices = vcat(map(1:length(ys), ys) do n, y
      y .+ colsrng[1:length(y)] .+ slicesrng[n]
   end...)
   l = -sum(Ŷs[true_linindices])
   return l
end
