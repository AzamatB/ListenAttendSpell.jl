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

# Bidirectional LSTM
struct BLSTM{L,D}
   forward  :: L
   backward :: L
   dense    :: D
end

@treelike BLSTM

function BLSTM(in::Integer, hidden::Integer, out::Integer, σ = identity)
   forward  = LSTM(in, hidden)
   backward = LSTM(in, hidden)
   dense    = Dense(2hidden, out, σ)
   return BLSTM(forward, backward, dense)
end

BLSTM(D_in::Integer, D_out::Integer) = BLSTM(D_in, ceil(Int, (D_in + D_out)/2), D_out)

(m::BLSTM)(xs::AbstractVector{<:AbstractVecOrMat})::AbstractVector{<:AbstractVecOrMat} = m.dense.(vcat.(m.forward.(xs), flip(m.backward, xs)))

# Flux.reset!(m::BLSTM) = reset!((m.forward, m.backward)) # not needed as taken care of by @treelike

function restack(xs::VV)::VV where VV <: AbstractVector{<:AbstractVecOrMat}
   T = length(xs)
   return vcat.(xs[1:2:T], xs[2:2:T])
end

"""
   PBLSTM(D_in::Integer, D_out::Integer)

Pyramidal BLSTM is the same as BLSTM, with the addition that the outputs of BLSTM are concatenated at consecutive steps.
"""
function PBLSTM(D_in::Integer, D_out::Integer)
   iseven(D_out) || throw("output dimension of the pyramidal BLSTM layer must be even")
   D_out_blstm = Int(D_out/2)
   return Chain(BLSTM(D_in, D_out_blstm), restack)
end

"""
   Encoder(layer_sizes)
   Encoder(D_in::Integer, D_out::Integer; nlayers::Integer = 4)
   Encoder(D_in::Integer, D_out::Integer, hidden_sizes)

Encoder that consists of block of PBLSTMs. It accepts filter bank spectra as inputs and acts as acoustic model encoder.
"""
function Encoder(layer_sizes)
   (length(layer_sizes) ≥ 3) || throw("number of layers of Encoder must be ≥ 2")
   layer_dims = Tuple(partition(layer_sizes, 2, 1))
   layers = ( PBLSTM(D_in, D_out) for (D_in, D_out) ∈ layer_dims[1:end-1] )
   model = Chain(layers..., BLSTM(layer_dims[end]...))
   return model
end

function Encoder(D_in::Integer, D_out::Integer; nlayers::Integer = 4)
   layer_sizes = range(D_in, D_out; length=nlayers+1)
   layer_sizes = map(layer_sizes) do x
      n = ceil(Int, x)
      n = iseven(n) ? n : (n + 1)
      return n
   end
   layer_sizes[1]   = D_in
   layer_sizes[end] = D_out
   return Encoder(layer_sizes)
end

Encoder(D_in::Integer, D_out::Integer, hidden_sizes) = Encoder((D_in, hidden_sizes..., D_out))

function MLP(layer_sizes, σs)
   layers = Tuple(Dense(D_in, D_out, σ) for ((D_in, D_out), σ) ∈ zip(partition(layer_sizes, 2, 1), σs))
   model = length(layers) == 1 ? first(layers) : Chain(layers...)
   return model
end

function MLP(layer_sizes, σ::Function)
   σs = ntuple(i -> σ, length(layer_sizes))
   return MLP(layer_sizes, σs)
end

function MLP(D_in::Integer, D_out::Integer, σs)
   layer_sizes = ceil.(Int, range(D_in, D_out; length=length(σs)+1))
   return MLP(layer_sizes, σs)
end

function MLP(D_in::Integer, D_out::Integer, σ::Function=identity; nlayers::Integer = 1)
   σs = ntuple(i -> σ, nlayers)
   return MLP(D_in, D_out, σs)
end

function Decoder(layer_sizes)
   layers = ( LSTM(D_in, D_out) for (D_in, D_out) ∈ partition(layer_sizes, 2, 1) )
   model = Chain(layers...)
   return model
end

function Decoder(D_in::Integer, D_out::Integer; nlayers::Integer = 2)
   layer_sizes = ceil.(Int, range(D_in, D_out; length=nlayers+1))
   return Decoder(layer_sizes)
end

Decoder(D_in::Integer, D_out::Integer, hidden_sizes) = Decoder((D_in, hidden_sizes..., D_out))

function CharacterDistribution(D_in::Integer, D_out::Integer, σ::Function; nlayers::Integer, applylog::Bool=true)
   f = applylog ? logsoftmax : softmax
   layer_sizes = ceil.(Int, range(D_in, D_out; length=nlayers+1))
   layer_dims = Tuple(partition(layer_sizes, 2, 1))
   layers = ( Dense(D_in, D_out, σ) for (D_in, D_out) ∈ layer_dims[1:end-1] )
   return Chain(layers..., Dense(layer_dims[end]...), f)
end

CharacterDistribution(D_in::Integer, D_out::Integer; applylog::Bool=true) = Chain(Dense(D_in, D_out), applylog ? logsoftmax : softmax)

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

function State(D_c::Integer, D_d::Integer, D_p::Integer)
   context₀    = param(zeros(Float32, D_c, 1))
   decoding₀   = param(zeros(Float32, D_d, 1))
   prediction₀ = param(zeros(Float32, D_p, 1))
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

function LAS(D_in::Integer, D_out::Integer;
             D_encoding::Integer,
             D_attention::Integer,
             D_decoding::Integer)
   state       = State(D_encoding, D_decoding, D_out)
   listen      = Encoder(D_in, D_encoding)
   attention_ϕ = MLP(D_decoding, D_attention)
   attention_ψ = MLP(D_encoding, D_attention)
   spell       = Decoder(D_encoding + D_decoding + D_out, D_decoding)
   infer       = CharacterDistribution(D_encoding + D_decoding, D_out)
   las = LAS(state, listen, attention_ϕ, attention_ψ, spell, infer) |> gpu
   return las
end

function (m::LAS{M})(xs::AbstractVector{<:AbstractMatrix}, maxT::Integer = length(xs))::AbstractArray{<:Real,3} where {M <: AbstractMatrix{<:Real}}
   batch_size = size(first(xs), 2)
   # compute input encoding
   hs = m.listen(xs)
   # concatenate sequence of D×N matrices into ssingle D×N×T 3-dimdimensional array
   Hs = cat(hs...; dims=3)
   # precompute ψ(H)
   ψHs = m.attention_ψ.(hs)
   # initialize prediction
   ŷs = similar(xs, M, maxT)
   # compute inital decoder state
   O = gpu(zeros(Float32, size(m.state.decoding, 1), batch_size))
   m.state.decoding = m.spell([m.state.decoding; m.state.prediction; m.state.context]) .+ O
   @inbounds for i ∈ eachindex(ŷs)
      # compute ϕ(sᵢ)
      # ϕSᵢᵀ = m.attention_ϕ(m.state.decoding)'
      ϕSᵢᵀ = permutedims(m.attention_ϕ(m.state.decoding)) # workaround for bug in encountered during training
      # compute attention context
      Eᵢs = diag.(Ref(ϕSᵢᵀ) .* ψHs)
      αᵢs = softmax(vcat(Eᵢs'...))
      # compute attention context, i.e. contextᵢ = Σᵤαᵢᵤhᵤ
      m.state.context = dropdims(sum(reshape(αᵢs, 1, batch_size, :) .* Hs; dims=3); dims=3)
      # predict probability distribution over character alphabet
      m.state.prediction = m.infer([m.state.decoding; m.state.context])
      ŷs[i] = m.state.prediction
      # compute decoder state
      m.state.decoding = m.spell([m.state.decoding; m.state.prediction; m.state.context])
   end
   # concatenate sequence of D×N pprediction matrices into ssingle D×T×N 3-dimdimensional array
   Ŷs = cat((ŷ -> reshape(ŷ, size(ŷ,1), 1, :)).(ŷs)...; dims=2)
   reset!(m)
   return Ŷs
end

function (m::LAS)(xs::AbstractVector{<:AbstractVector})::AbstractMatrix{<:Real}
   T = length(xs)
   xs = gpu.(reshape.(pad(xs), :,1))
   Ŷ = dropdims(las(xs, T); dims=3)
   return Ŷ
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

function batch!(Xs, maxT::Integer = maximum(length, Xs), multiplicity::Integer = 8)::Vector{<:AbstractMatrix}
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

function build_batches!(Xs::AbstractVector{<:AbstractVector{<:AbstractVector}}, ys::AbstractVector{<:AbstractVector}, batch_size::Integer, multiplicity::Integer = 8)
   sortingidxs = sortperm(Xs; by=length)
   Xs = Xs[sortingidxs]
   ys = ys[sortingidxs]

   lengthdata = length(Xs)
   firstidxs = 1:batch_size:lengthdata
   lastidxs = collect(firstidxs .+ (batch_size - 1))
   lastidxs[end] = min(lengthdata, lastidxs[end])
   maxTs = length.(Xs[lastidxs])

   xs_batches = [ batch!(Xs[firstidx:lastidx], maxT, multiplicity) for (firstidx, lastidx, maxT) ∈ zip(firstidxs, lastidxs, maxTs) ]
   ys_batches = [ ys[firstidx:lastidx] for (firstidx, lastidx) ∈ zip(firstidxs, lastidxs) ]

   return xs_batches, ys_batches, maxTs
end

const las, PHONEMES = let
   JLD2.@load "/Users/Azamat/Projects/LAS/data/TIMIT/TIMIT_MFCC/data_train.jld" PHONEMES
   # D_x = length(first(X))
   D_x = 39
   D_y = length(PHONEMES)
   D_encoding  = 128
   D_attention = 64 # attention dimension
   D_decoding  = 256
   D_feed_forward = 128
   D_LSTM_speller = 512
   las = LAS(D_x, D_y; D_encoding=D_encoding, D_attention=D_attention, D_decoding=D_decoding)
   las, PHONEMES
end

function loss(xs::AbstractVector{<:AbstractMatrix{<:Real}}, ys::AbstractVector{<:AbstractVector{<:Integer}}, maxT::Integer = length(xs))::Real
   Ŷs = las(gpu.(xs), maxT)
   x, y, z = size(Ŷs)
   colsrng = range(0; step=x, length=y)
   slicesrng = range(0; step=x*y, length=z)
   # true_linindices = vcat([y .+ colsrng[eachindex(y)] .+ slicesrng[n] for (n, y) ∈ enumerate(ys)]...)
   true_linindices = mapreduce((n, y) -> y .+ colsrng[eachindex(y)] .+ slicesrng[n], vcat, eachindex(ys), ys)
   l = -sum(Ŷs[true_linindices])
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
   let batch_size = 154, val_set_size = 32
      JLD2.@load "/Users/Azamat/Projects/LAS/data/TIMIT/TIMIT_MFCC/data_test.jld" Xs ys

      ys_val = ys[1:val_set_size]
      maxT_val = maximum(length, ys_val)
      Xs_val = batch!(Xs[1:val_set_size], maxT_val)

      Xs_test, ys_test, maxTs_test = build_batches!(Xs[(val_set_size+1):end], ys[(val_set_size+1):end], batch_size)

      JLD2.@load "/Users/Azamat/Projects/LAS/data/TIMIT/TIMIT_MFCC/data_train.jld" Xs ys
      X, y = first(Xs), first(ys)
      Xs_train, ys_train, maxTs_train = build_batches!(Xs, ys, batch_size)

      eval_idxs = sample(eachindex(ys), val_set_size; replace=false)
      ys_eval = ys[eval_idxs]
      maxT_eval = maximum(length, ys_eval)
      Xs_eval = batch!(Xs[eval_idxs], maxT_eval)

      X, y,
      Xs_train, ys_train, maxTs_train,
      # Xs_test, ys_test, maxTs_test,
      Xs_eval, ys_eval, maxT_eval,
      Xs_val, ys_val, maxT_val
   end

   @time predict(X)
   @time @show loss_val_saved = loss(Xs_val, ys_val)
   @time @show loss(Xs_eval, ys_eval)

   θ = params(las)
   optimizer = ADAM()
   data = zip(Xs_train, ys_train)

   show_loss_val() = @show(loss(Xs_val, ys_val))
   show_loss_eval() = @show(loss(Xs_eval, ys_eval))

   @time @epochs 1 begin
      @time train!(loss, θ, data, optimizer; cb = throttle(show_loss_eval, 300))
      loss_val = loss(Xs_val, ys_val)
      @show loss_val
      if loss_val < loss_val_saved
         loss_val_saved = loss_val
         @save "/Users/Azamat/Projects/LAS/models/TIMIT/LAS.jld2" las optimizer
      end
   end
end

main()

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
