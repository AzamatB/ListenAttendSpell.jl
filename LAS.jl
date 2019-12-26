# Listen, Attend and Spell: arxiv.org/abs/1508.01211
# using CuArrays
# CuArrays.allowscalar(false)

using Flux
using Flux: flip, reset!, onecold, throttle, train!, @treelike, @epochs
using IterTools
using LinearAlgebra
using JLD2
using StatsBase
import Base.Iterators

import Base.AbstractVecOrTuple # Base.AbstractVecOrTuple{T} = Union{AbstractVector{<:T}, Tuple{Vararg{T}}}

# Bidirectional LSTM
struct BLSTM{L}
   forward  :: L
   backward :: L
end

@treelike BLSTM

function BLSTM(in::Integer, out::Integer)
   iseven(out) || throw("output dimension of the BLSTM layer must be even")
   hidden = out ÷ 2
   forward  = LSTM(in, hidden)
   backward = LSTM(in, hidden)
   return BLSTM(forward, backward)
end

(m::BLSTM)(xs::AbstractVector{<:AbstractVecOrMat})::AbstractVector{<:AbstractVecOrMat} = vcat.(m.forward.(xs), flip(m.backward, xs))

# Flux.reset!(m::BLSTM) = reset!((m.forward, m.backward)) # not needed as taken care of by @treelike

function restack(xs::VV)::VV where VV <: AbstractVector{<:AbstractVecOrMat}
   return vcat.(xs[1:2:end], xs[2:2:end])
end

"""
   PBLSTM(in::Integer, out::Integer)

Pyramidal BLSTM is the same as BLSTM, with the addition that the outputs of BLSTM are concatenated at consecutive steps.
"""
function PBLSTM(in::Integer, out::Integer)
   (mod(out, 4) == 0) || throw("output dimension of the pyramidal BLSTM layer must be multiple of 4")
   hidden = out ÷ 2
   return Chain(BLSTM(in, hidden), restack)
end


"""
   Encoder(layer_sizes)
   Encoder(in::Integer, out::Integer; nlayers::Integer = 4)
   Encoder(in::Integer, out::Integer, hidden_sizes)

Encoder that consists of block of PBLSTMs. It accepts filter bank spectra as inputs and acts as acoustic model encoder.
"""
function Encoder(layer_sizes)
   (length(layer_sizes) ≥ 3) || throw("number of layers of Encoder must be ≥ 2")
   layer_dims = Tuple(partition(layer_sizes, 2, 1))
   pblstm_layers = ( PBLSTM(in, out) for (in, out) ∈ layer_dims[2:end] )
   model = Chain(BLSTM(layer_dims[1]...), pblstm_layers...)
   return model
end

function Encoder(in::Integer, out::Integer; nlayers::Integer = 4)
   layer_sizes = map(x -> 4ceil(Int, x/4), range(in, out; length=nlayers+1))
   layer_sizes[1]   = in
   layer_sizes[end] = out
   return Encoder(layer_sizes)
end

Encoder(in::Integer, out::Integer, hidden_sizes) = Encoder((in, hidden_sizes..., out))

function MLP(layer_sizes, σs)
   layers = Tuple(Dense(in, out, σ) for ((in, out), σ) ∈ zip(partition(layer_sizes, 2, 1), σs))
   model = length(layers) == 1 ? first(layers) : Chain(layers...)
   return model
end

function MLP(layer_sizes, σ::Function)
   σs = ntuple(i -> σ, length(layer_sizes))
   return MLP(layer_sizes, σs)
end

function MLP(in::Integer, out::Integer, σs)
   layer_sizes = ceil.(Int, range(in, out; length=length(σs)+1))
   return MLP(layer_sizes, σs)
end

function MLP(in::Integer, out::Integer, σ::Function=identity; nlayers::Integer = 1)
   σs = ntuple(i -> σ, nlayers)
   return MLP(in, out, σs)
end

function Decoder(layer_sizes)
   layers = ( LSTM(in, out) for (in, out) ∈ partition(layer_sizes, 2, 1) )
   model = Chain(layers...)
   return model
end

function Decoder(in::Integer, out::Integer; nlayers::Integer = 2)
   layer_sizes = ceil.(Int, range(in, out; length=nlayers+1))
   return Decoder(layer_sizes)
end

Decoder(in::Integer, out::Integer, hidden_sizes) = Decoder((in, hidden_sizes..., out))

function CharacterDistribution(in::Integer, out::Integer, σ::Function; nlayers::Integer, applylog::Bool=true)
   f = applylog ? logsoftmax : softmax
   layer_sizes = ceil.(Int, range(in, out; length=nlayers+1))
   layer_dims = Tuple(partition(layer_sizes, 2, 1))
   layers = ( Dense(in, out, σ) for (in, out) ∈ layer_dims[1:end-1] )
   return Chain(layers..., Dense(layer_dims[end]...), f)
end

CharacterDistribution(in::Integer, out::Integer; applylog::Bool=true) = Chain(Dense(in, out), applylog ? logsoftmax : softmax)

mutable struct State{M <: AbstractMatrix{<:Real}}
   context     :: M   # last attention context
   decoding    :: M   # last decoder state
   prediction  :: M   # last prediction
   # reset values
   context₀    :: M
   decoding₀   :: M
   prediction₀ :: M
end

@treelike State

function State(dim_c::Integer, dim_d::Integer, dim_p::Integer)
   context₀    = zeros(Float32, dim_c, 1)
   decoding₀   = zeros(Float32, dim_d, 1)
   prediction₀ = zeros(Float32, dim_p, 1)
   return State(context₀, decoding₀, prediction₀, context₀, decoding₀, prediction₀)
end

function Flux.reset!(s::State)
   s.context    = s.context₀
   s.decoding   = s.decoding₀
   s.prediction = s.prediction₀
   return nothing
end

struct LAS{V, E, Dϕ, Dψ, L, C}
   state       :: State{V} # current state of the model
   listen      :: E   # encoder function
   attention_ϕ :: Dϕ  # attention context function
   attention_ψ :: Dψ  # attention context function
   spell       :: L   # RNN decoder
   infer       :: C   # character distribution inference function
end

@treelike LAS

function LAS(dim_in::Integer, dim_out::Integer;
             dim_encoding::Integer,
             dim_attention::Integer,
             dim_decoding::Integer)
   state       = State(dim_encoding, dim_decoding, dim_out)
   listen      = Encoder(dim_in, dim_encoding)
   attention_ϕ = MLP(dim_decoding, dim_attention)
   attention_ψ = MLP(dim_encoding, dim_attention)
   spell       = Decoder(dim_encoding + dim_decoding + dim_out, dim_decoding)
   infer       = CharacterDistribution(dim_encoding + dim_decoding, dim_out)
   las = LAS(state, listen, attention_ϕ, attention_ψ, spell, infer) |> gpu
   return las
end

function (m::LAS)(xs::AbstractVector{<:AbstractMatrix}, maxT::Integer = length(xs))::AbstractVector{<:AbstractMatrix{<:Real}}
   batch_size = size(first(xs), 2)
   # compute input encoding
   hs = m.listen(xs)
   # concatenate sequence of D×N matrices into ssingle D×N×T 3-dimdimensional array
   Hs = cat(hs...; dims=3)
   # precompute ψ(H)
   ψHs = m.attention_ψ.(hs)
   # compute inital decoder state for a batch
   O = gpu(zeros(Float32, size(m.state.decoding, 1), batch_size))
   m.state.decoding = m.spell([m.state.decoding; m.state.prediction; m.state.context]) .+ O

   ŷs = map(1:maxT) do _
      # compute ϕ(sᵢ)
      ϕSᵢᵀ = m.attention_ϕ(m.state.decoding)'
      # compute attention context
      Eᵢs = diag.(Ref(ϕSᵢᵀ) .* ψHs)
      αᵢs = softmax(vcat(Eᵢs'...))
      # compute attention context, i.e. contextᵢ = Σᵤαᵢᵤhᵤ
      m.state.context = dropdims(sum(reshape(αᵢs, 1, batch_size, :) .* Hs; dims=3); dims=3)
      # predict probability distribution over character alphabet
      m.state.prediction = m.infer([m.state.decoding; m.state.context])
      # compute decoder state
      m.state.decoding = m.spell([m.state.decoding; m.state.prediction; m.state.context])
      return m.state.prediction
   end

   reset!(m)
   return ŷs
end

function (m::LAS)(xs::AbstractVector{<:AbstractVector{<:Real}})::AbstractVector{<:AbstractVector{<:Real}}
   T = length(xs)
   xs = gpu.(reshape.(pad(xs), :,1))
   ŷs = dropdims.(las(xs, T); dims=2)
   return ŷs
end

function Flux.reset!(m::LAS)
   reset!(m.state)
   reset!(m.listen)
   reset!(m.spell)
   return nothing
end

function pad(xs::VV; multiplicity::Integer=8)::VV where VV <: AbstractVector{<:AbstractVector}
   T = length(xs)
   newlength = ceil(Int, T / multiplicity)multiplicity
   el_min = minimum(minimum(xs))
   x = fill!(similar(first(xs)), el_min)
   xs = resize!(copy(xs), newlength)
   xs[(T+1):end] .= Ref(x)
   return xs
end

function batch_inputs!(Xs, maxT::Integer = maximum(length, Xs), multiplicity::Integer = 8)::Vector{<:AbstractMatrix}
   # Xs must be an iterable, whose each element is a vector of vectors,
   # and dimensionality of all element vectors must be the same
   # find the smallest multiple of `multiplicity` that is no less than `maxT`
   newT = ceil(Int, maxT / multiplicity)multiplicity
   # resize each sequence `xs` to the size `newT` paddding with vector filled with smallest values
   for xs ∈ Xs
      T = length(xs)
      el_min = minimum(minimum(xs))
      x = fill!(similar(first(xs)), el_min)
      resize!(xs, newT)
      xs[(T+1):end] .= (x,)
   end
   # for each time step `t`, get `t`ᵗʰ vector x across all sequences and concatenate them into matrix
   return [hcat(getindex.(Xs, t)...) for t ∈ 1:newT]
end

function batch_targets(ys::VV, maxT::Integer = maximum(length, ys))::VV where VV <: AbstractVector{<:AbstractVector{<:Integer}}
   batch_size = length(ys)
   lin_idxs = similar(ys, maxT)
   idxs = similar(first(ys), batch_size)
   offsets = range(0; step=length(PHONEMES), length=batch_size)
   for t ∈ 1:maxT
      n = 0
      for (y, offset) ∈ zip(ys, offsets)
         if t <= length(y)
            n += 1
            idxs[n] = offset + y[t]
         end
      end
      lin_idxs[t] = idxs[1:n]
   end
   return lin_idxs
end

function batch(Xs::AbstractVector{<:AbstractVector{<:AbstractVector}}, ys::AbstractVector{<:AbstractVector}, batch_size::Integer, multiplicity::Integer = 8)
   sortidxs = sortperm(Xs; by=length)
   Xs = Xs[sortidxs]
   ys = ys[sortidxs]

   cumseqlengths = cumsum(length.(ys))
   nbatches = floor(Int, length(Xs) / batch_size)
   # subtract 0.5 from the last element of the range
   # to ensure that i index inside the loop won't go out of bounds due to floating point rounding errors
   cum_n_elts_rng = range(0, cumseqlengths[end]-0.5; length = nbatches+1)[2:end]
   lastidxs = similar(sortidxs, nbatches)
   i = 1
   for (n, cum_n_elts_for_a_batch) ∈ enumerate(cum_n_elts_rng)
      while cumseqlengths[i] < cum_n_elts_for_a_batch
         i += 1
      end
      lastidxs[n] = i
   end
   firstidxs = [1; lastidxs[1:(end-1)] .+ 1]
   maxTs = length.(Xs[lastidxs])

   xs_batches = [ batch_inputs!(Xs[firstidx:lastidx], maxT, multiplicity) for (firstidx, lastidx, maxT) ∈ zip(firstidxs, lastidxs, maxTs) ]
   idxs_batches = [ batch_targets(ys[firstidx:lastidx], maxT) for (firstidx, lastidx, maxT) ∈ zip(firstidxs, lastidxs, maxTs) ]
   return xs_batches, idxs_batches, maxTs
end

# const las, PHONEMES = let
#    JLD2.@load "/Users/Azamat/Projects/LAS/data/TIMIT/TIMIT_MFCC/data_train.jld" PHONEMES
#    # dim_in = length(first(X))
#    dim_in = 39
#    dim_out = length(PHONEMES)
#    # dim_encoding  = (512, 512, 512, 512)
#    dim_encoding  = 512
#    dim_attention = 512 # attention dimension
#    dim_decoding  = 512
#    # initialize with uniform(-0.1, 0.1)
#
#    dim_feed_forward = 128
#    dim_LSTM_speller = 512
#
#    las = LAS(dim_in, dim_out; dim_encoding=dim_encoding, dim_attention=dim_attention, dim_decoding=dim_decoding)
#    las, PHONEMES
# end

const las, PHONEMES = let
   JLD2.@load "/Users/Azamat/Projects/LAS/data/TIMIT/TIMIT_MFCC/data_train.jld" PHONEMES
   # dim_in = length(first(X))
   dim_in = 39
   dim_out = length(PHONEMES)
   # dim_encoding  = (512, 512, 512, 512)
   dim_encoding  = 16
   dim_attention = 16 # attention dimension
   dim_decoding  = 16
   # initialize with uniform(-0.1, 0.1)

   dim_feed_forward = 16
   dim_LSTM_speller = 16

   las = LAS(dim_in, dim_out; dim_encoding=dim_encoding, dim_attention=dim_attention, dim_decoding=dim_decoding)
   las, PHONEMES
end

function loss(xs::AbstractVector{<:AbstractMatrix{<:Real}}, indexes::AbstractVector{<:AbstractVector{<:Integer}})::Real
   ŷs = las(xs, length(indexes))
   l = -sum(sum.(getindex.(ŷs, indexes)))
   return l
end

# best path decoding
function predict(xs::AbstractVector{<:AbstractMatrix{<:Real}}, lengths::AbstractVector{<:Integer}, labels=PHONEMES)::AbstractVector{<:AbstractVector}
   maxT = maximum(lengths)
   Ŷs = las(gpu.(xs), maxT) |> cpu
   predictions = [onecold(Ŷs[:, 1:len, n], labels) for (n, len) ∈ enumerate(lengths)]
   return predictions
end

function predict(xs::AbstractVector{<:AbstractVector{<:Real}}, labels=PHONEMES)::AbstractVector
   Ŷ = las(xs) |> cpu
   prediction = onecold(Ŷ, labels)
   return prediction
end

function main()
# load data
X, y,
Xs_train, ys_train, maxTs_train,
# Xs_test, ys_test, maxTs_test,
Xs_eval, ys_eval, maxT_eval,
Xs_val, ys_val, maxT_val =
let batch_size = 77, val_set_size = 32
   JLD2.@load "/Users/Azamat/Projects/LAS/data/TIMIT/TIMIT_MFCC/data_test.jld" Xs ys

   ys_val = ys[1:val_set_size]
   maxT_val = maximum(length, ys_val)
   Xs_val = batch_inputs!(Xs[1:val_set_size], maxT_val)
   ys_val = batch_targets(ys_val, maxT_val)

   Xs_test, ys_test, maxTs_test = batch(Xs[(val_set_size+1):end], ys[(val_set_size+1):end], batch_size)

   JLD2.@load "/Users/Azamat/Projects/LAS/data/TIMIT/TIMIT_MFCC/data_train.jld" Xs ys
   X, y = first(Xs), first(ys)
   Xs_train, ys_train, maxTs_train = batch(Xs, ys, batch_size)

   eval_idxs = sample(eachindex(ys), val_set_size; replace=false)
   ys_eval = ys[eval_idxs]
   maxT_eval = maximum(length, ys_eval)
   Xs_eval = batch_inputs!(Xs[eval_idxs], maxT_eval)
   ys_eval = batch_targets(ys_eval, maxT_eval)

   X, y,
   Xs_train, ys_train, maxTs_train,
   # Xs_test, ys_test, maxTs_test,
   Xs_eval, ys_eval, maxT_eval,
   Xs_val, ys_val, maxT_val
end

global loss_val_saved
θ = params(las)
optimiser = ADAM()
# optimiser = Flux.RMSProp(0.0001)
# optimiser = AMSGrad()
# optimiser = AMSGrad(0.0001)
# optimiser = AMSGrad(0.00001)

using BenchmarkTools
@btime loss(Xs_eval, ys_eval)
(xs, ys) = first.([Xs_train, ys_train])

xs = gpu.(xs)
l, pb = Flux.pullback(θ) do
   loss(xs, ys)
end
dldθ = pb(one(l))


n_epochs = 2
for epoch ∈ 1:n_epochs
   for (xs, ys) ∈ zip(Xs_train, ys_train)
      xs = gpu.(xs)
      l, pb = Flux.pullback(θ) do
         loss(xs, ys)
      end
      dldθ = pb(1.0f0)
      Flux.Optimise.update!(optimiser, θ, dldθ)
      @show l
   end
   @info "finished epoch $epoch"
   @show loss(Xs_eval, ys_eval)
   loss_val = loss(Xs_val, ys_val)
   @show loss_val
   if loss_val < loss_val_saved
      loss_val_saved = loss_val
      @save "/Users/Azamat/Projects/LAS/models/TIMIT/LAS.jld2" las optimiser loss_val_saved
   end
end
end


const loss_val_saved = let
   JLD2.@load "/Users/Azamat/Projects/LAS/models/TIMIT/LAS.jld2" loss_val_saved
   loss_val_saved
end

# main()

"""
   levendist(seq₁::AbstractVector, seq₂::AbstractVector)::Int
   levendist(seq₁::AbstractString, seq₂::AbstractString)::Int

Levenshtein distance between sequences `seq₁` and `seq₂`.
"""
function levendist(seq₁::AbstractVector, seq₂::AbstractVector)::Int
   # ensure that length(seq₁) <= length(seq₂)
   if length(seq₁) > length(seq₂)
      seq₁, seq₂ = seq₂, seq₁
   end
   # ignore prefix common to both sequences
   start = length(seq₁) + 1
   for (i, (el₁, el₂)) ∈ enumerate(zip(seq₁, seq₂))
      if el₁ != el₂
         start = i
         break
      end
   end
   @views begin
      seq₁, seq₂ = seq₁[start:end], seq₂[start:end]
      # ignore suffix common to both sequences
      lenseq₁ = length(seq₁)
      offset = lenseq₁
      for (i, el₁, el₂) ∈ zip(0:lenseq₁, Iterators.reverse(seq₁), Iterators.reverse(seq₂))
         if el₁ != el₂
            offset = i
            break
         end
      end
      seq₁, seq₂ = seq₁[1:(end-offset)], seq₂[1:(end-offset)]
   end
   lenseq₁ = length(seq₁)
   dist = length(seq₂)
   # if all of shorter sequence matches prefix and/or suffix of longer sequence, then Levenshtein
   # distance is just the delete cost of the additional characters present in longer sequence
   lenseq₁ == 0 && return dist

   costs = collect(eachindex(seq₂))
   @inbounds for (i, el₁) ∈ zip(0:(lenseq₁-1), seq₁)
      left = dist = i
      for (j, el₂) ∈ enumerate(seq₂)
         # cost on diagonal (substitution)
         above, dist, left = dist, left, costs[j]
         if el₁ != el₂
            # minimum of substitution, insertion and deletion costs
            dist = 1 + min(dist, left, above)
         end
         costs[j] = dist
      end
      # @show costs, dist
   end
   return dist
end

levendist(seq₁::AbstractString, seq₂::AbstractString)::Int = levendist(collect(seq₁), collect(seq₂))

per(source_phoneme, target_phoneme)::Real = levendist(source_phoneme, target_phoneme)/length(target_phoneme)
cer(source_chars, target_chars)::Real = levendist(source_chars, target_chars)/length(target_chars)
wer(source_words, target_words)::Real = levendist(source_words, target_words)/length(target_words)
