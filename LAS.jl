# Listen, Attend and Spell: arxiv.org/abs/1508.01211

module ListenAttendSpell

# using CuArrays
# CuArrays.allowscalar(false)

using Flux
using Flux: reset!, onecold, @functor
using Zygote
using Zygote: Buffer
using LinearAlgebra
using JLD2
using IterTools
using Base.Iterators: reverse
using StatsBase

# Bidirectional LSTM
struct BLSTM{L}
   forward  :: L
   backward :: L
   dim_out  :: Int
end

@functor BLSTM (forward, backward)

function BLSTM(in::Integer, out::Integer)
   forward  = LSTM(in, out)
   backward = LSTM(in, out)
   return BLSTM(forward, backward, out)
end

Base.show(io::IO, l::BLSTM)  = print(io,  "BLSTM(", size(l.forward.cell.Wi, 2), ", ", l.dim_out, ")")

function flip(f, xs)
   flipped_xs = Buffer(xs)
   @inbounds for t ∈ reverse(eachindex(xs))
      flipped_xs[t] = f(xs[t])
   end
   return copy(flipped_xs)
end

(m::BLSTM)(xs::DenseVector{<:DenseVecOrMat})::DenseVector{<:DenseVecOrMat} =
   vcat.(m.forward.(xs), flip(m.backward, xs))

function (m::BLSTM)(Xs::T)::T where T <: DenseArray{<:Real,3}
   Ys = Buffer(Xs, 2m.dim_out, size(Xs,2), size(Xs,3))
   slice_f = axes(Ys,1)[1:m.dim_out]
   slice_b = axes(Ys,1)[(m.dim_out+1):end]
   @inbounds @views for (t_f, t_b) ∈ zip(axes(Xs,3), reverse(axes(Xs,3)))
      Ys[slice_f,:,t_f] = m.forward(Xs[:,:,t_f])
      Ys[slice_b,:,t_b] = m.backward(Xs[:,:,t_b])
   end
   return copy(Ys)
end

# Flux.reset!(m::BLSTM) = reset!((m.forward, m.backward)) # not needed as taken care of by @functor

"""
    PBLSTM(in::Integer, out::Integer)

Pyramidal BLSTM is the same as BLSTM, with the addition that the outputs of BLSTM are concatenated at every two consecutive steps.
"""
struct PBLSTM{L}
   forward  :: L
   backward :: L
   dim_out  :: Int
end

@functor PBLSTM (forward, backward)

function PBLSTM(in::Integer, out::Integer)
   forward  = LSTM(in, out)
   backward = LSTM(in, out)
   return PBLSTM(forward, backward, out)
end

Base.show(io::IO, l::PBLSTM) = print(io, "PBLSTM(", size(l.forward.cell.Wi, 2), ", ", l.dim_out, ")")

@views function (m::PBLSTM)(xs::DenseVector{<:DenseVecOrMat})::DenseVector{<:DenseVecOrMat}
   ys = vcat.(m.forward.(xs), flip(m.backward, xs))
   # restack step
   # return @inbounds(vcat.(ys[1:2:end], ys[2:2:end]))
   return [@inbounds vcat(ys[i-1], ys[i]) for i ∈ 2:2:lastindex(ys)]
end

function (m::PBLSTM)(Xs::T)::T where T <: DenseArray{<:Real,3}
   Ys = Buffer(Xs, 4m.dim_out, size(Xs,2), size(Xs,3) ÷ 2)
   slice_f_odd  = axes(Ys, 1)[1:m.dim_out]
   slice_b_odd  = axes(Ys, 1)[(m.dim_out+1):(2m.dim_out)]
   slice_f_even = axes(Ys, 1)[(2m.dim_out+1):(3m.dim_out)]
   slice_b_even = axes(Ys, 1)[(3m.dim_out+1):end]

   @inbounds @views for t ∈ axes(Ys, 3)
      Ys[slice_f_odd,:,t]  = m.forward(Xs[:,:,2t-1])
      Ys[slice_f_even,:,t] = m.forward(Xs[:,:,2t])
      Ys[slice_b_odd,:,t]  = m.backward(Xs[:,:,end-2t+2])
      Ys[slice_b_even,:,t] = m.backward(Xs[:,:,end-2t+1])
   end
   return copy(Ys)
end

# Flux.reset!(m::PBLSTM) = reset!((m.forward, m.backward)) # not needed as taken care of by @functor

"""
    Encoder(dims::NamedTuple{(:blstm, :pblstms_out),Tuple{NamedTuple{(:in, :out),Tuple{Int,Int}}, T}}) where T <: Union{Int, NTuple{N,Int} where N} -> Chain

Construct Encoder neural network a.k.a Listener from dimensions specified by named tuple `dims`.
`Encoder` consists BLSTM layer followed by a block of PBLSTM layers. It accepts filter bank spectra as input and acts as an acoustic model encoder.

# Examples
```jldoctest
julia> dims = (
          blstm       = (in = 4, out = 8),
          pblstms_out = (12, 16, 20)
       );

julia> Encoder(dims)
Chain(BLSTM(4, 8), PBLSTM(16, 12), PBLSTM(48, 16), PBLSTM(64, 20))
```
"""
function Encoder(dims::NamedTuple{(:blstm, :pblstms_out),Tuple{NamedTuple{(:in, :out),Tuple{Int,Int}}, T}}) where T <: Union{Int, NTuple{N,Int} where N}
   (length(dims.pblstms_out) >= 1) || throw("Encoder must have at least 1 pyramidal BLSTM layer")

   pblstm_layers = ( PBLSTM(4in, out) for (in, out) ∈ partition(dims.pblstms_out, 2, 1) )
   model = Chain(
      BLSTM(dims.blstm.in, dims.blstm.out),
      PBLSTM(2dims.blstm.out, first(dims.pblstms_out)),
      pblstm_layers...
   )
   return model
end


function MLP(in::Integer, out::NTuple{N,Integer}, σs::NTuple{N,Function}) where N
   model = Dense(in, first(out), first(σs))
   if N > 1
      layers = ( Dense(in, out, σ) for ((in, out), σ) ∈ zip(partition(out, 2, 1), σs[2:end]) )
      model = Chain(model, layers...)
   end
   return model
end

function MLP(in::Integer, out::Union{Integer, NTuple{N,Integer} where N}, σ::Function=identity)
   model = Dense(in, first(out), σ)
   if length(out) > 1
      layers = ( Dense(in, out, σ) for (in, out) ∈ partition(out, 2, 1) )
      model = Chain(model, layers...)
   end
   return model
end


function Decoder(in::Integer, out::Union{Integer, NTuple{N,Integer} where N})
   model = LSTM(in, first(out))
   if length(out) > 1
      layers = ( LSTM(in, out) for (in, out) ∈ partition(out, 2, 1) )
      model = Chain(model, layers...)
   end
   return model
end

CharacterDistribution(in::Integer, out::Integer; applylog::Bool=true) = Chain(Dense(in, out), applylog ? logsoftmax : softmax)

mutable struct State{M <: DenseMatrix{<:Real}}
   context     :: M   # last attention context
   decoding    :: M   # last decoder state
   prediction  :: M   # last prediction
   # reset values
   context₀    :: M
   decoding₀   :: M
   prediction₀ :: M
end

@functor State (context₀, decoding₀, prediction₀)

function State(dim_c::Integer, dim_d::Integer, dim_p::Integer)
   context₀    = zeros(Float32, dim_c, 1)
   decoding₀   = zeros(Float32, dim_d, 1)
   prediction₀ = zeros(Float32, dim_p, 1)
   return State(context₀, decoding₀, prediction₀, context₀, decoding₀, prediction₀)
end

Base.show(io::IO, s::State) = print(io, "State(", size(s.context, 1), ", ", size(s.decoding, 1), ", ", size(s.prediction, 1), ")")

function Flux.reset!(s::State)
   s.context    = s.context₀
   s.decoding   = s.decoding₀
   s.prediction = s.prediction₀
   return nothing
end

struct LAS{V, E, Aϕ, Aψ, D, C}
   state       :: State{V} # current state of the model
   listen      :: E   # encoder function
   attention_ψ :: Aψ  # keys attention context function
   attention_ϕ :: Aϕ  # query attention context function
   spell       :: D   # LSTM decoder
   infer       :: C   # character distribution inference function
end

@functor LAS

function LAS(encoder_dims::NamedTuple,
             attention_dim::Integer,
             decoder_out_dims::Tuple{Integer,Integer},
             out_dim::Integer)

   dim_encoding = 4last(encoder_dims.pblstms_out)
   dim_decoding =  last(decoder_out_dims)

   state       = State(dim_encoding, dim_decoding, out_dim)
   listen      = Encoder(encoder_dims)
   attention_ψ = MLP(dim_encoding, attention_dim)
   attention_ϕ = MLP(dim_decoding, attention_dim)
   spell       = Decoder(dim_encoding + dim_decoding + out_dim, decoder_out_dims)
   infer       = CharacterDistribution(dim_encoding + dim_decoding, out_dim)

   las = LAS(state, listen, attention_ψ, attention_ϕ, spell, infer) |> gpu
   return las
end

function Base.show(io::IO, m::LAS)
   print(io,
      "LAS(\n    ",
           m.state, ",\n    ",
           m.listen, ",\n    ",
           m.attention_ψ, ",\n    ",
           m.attention_ϕ, ",\n    ",
           m.spell, ",\n    ",
           m.infer,
      "\n)"
   )
end

# Flux.reset!(m::LAS) = reset!((m.state, m.listen, m.spell)) # not needed as taken care of by @functor

time_squashing_factor(m::LAS)::Integer = 2^(length(m.listen) - 1)

function (m::LAS)(xs::DenseVector{<:DenseMatrix}, maxT::Integer = length(xs))::DenseVector{<:DenseMatrix{<:Real}}
   batch_size = size(first(xs), 2)
   # compute input encoding, which are also values for the attention layer
   hs = m.listen(xs)
   # concatenate sequence of D×N matrices into ssingle D×N×T 3-dimdimensional array
   h = first(hs)
   Hs_buffer = Buffer(h, size(h,1), batch_size, length(hs))
   setindex!.(Ref(Hs_buffer), hs, :, :, axes(Hs_buffer, 3))
   Hs = copy(Hs_buffer)
   # Hs = cat(hs...; dims=3)
   # precompute keys ψ(H)
   ψhs = m.attention_ψ.(hs)
   # compute inital decoder state for a batch
   O = gpu(zeros(Float32, size(m.state.decoding, 1), batch_size))
   m.state.decoding = m.state.decoding .+ O

   ŷs = map(1:maxT) do _
      # compute query ϕ(sᵢ)
      ϕsᵢᵀ = m.attention_ϕ(m.state.decoding)'
      # compute energies
      Eᵢs = diag.((ϕsᵢᵀ,) .* ψhs)
      # compute attentions weights
      αᵢs = softmax(hcat(Eᵢs...); dims=2)
      # αᵢs = softmax(hcat(Eᵢs...)')
      # αᵢs = softmax(reduce(hcat, Eᵢs); dims=2)
      # αᵢs = softmax(reduce(hcat, Eᵢs)')
      # αᵢs = softmax(vcat(Eᵢs'...))
      # αᵢs = softmax(reduce(vcat, Eᵢs'))
      # compute attended context by normalizing values with respect to attention weights, i.e. contextᵢ = Σᵤαᵢᵤhᵤ
      # hcat(@inbounds([sum(αᵢs[b,u] * hs[u][:,b] for u ∈ eachindex(hs)) for b ∈ axes(αᵢs, 1)])...)
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

function (m::LAS)(xs::DenseVector{<:DenseVector{<:Real}})::DenseVector{<:DenseVector{<:Real}}
   T = length(xs)
   xs = gpu.(reshape.(pad(xs, time_squashing_factor(m)), Val(2)))
   ŷs = dropdims.(m(xs, T); dims=2)
   return ŷs
end


function pad(xs::VV, multiplicity)::VV where VV <: DenseVector{<:DenseVector}
   T = length(xs)
   newT = ceil(Int, T / multiplicity)multiplicity
   z = similar(first(xs))
   fill!(z, zero(eltype(z)))
   xs = resize!(copy(xs), newT)
   xs[(T+1):end] .= Ref(z)
   return xs
end

"""
    batch_inputs!(Xs, multiplicity::Integer, maxT::Integer = maximum(length, Xs))::Vector{<:DenseMatrix}

Given a collection `Xs` of input sequences, returns a vector of length T whose tᵗʰ element is a D×B matrix of inputs at time t across all sequences in a given batch.
Here T is the maximum time length in the batch, D is the dimensionality of the input elements of sequences and B is the batch size.
"""
function batch_inputs!(Xs, multiplicity::Integer, maxT::Integer = maximum(length, Xs))::Vector{<:DenseMatrix}
   # find the smallest multiple of `multiplicity` that is no less than `maxT`
   newT = ceil(Int, maxT / multiplicity)multiplicity
   # initialize & populate padding vector
   z = similar(first(first(Xs)))
   fill!(z, zero(eltype(z)))
   # resize each sequence `xs` to the size `newT` by paddding it with vector z of zeros
   for xs ∈ Xs
      T = length(xs)
      resize!(xs, newT)
      xs[(T+1):end] .= Ref(z)
   end
   # for each time step `t`, get `t`ᵗʰ vector x across all sequences and concatenate them into matrix
   return [hcat(getindex.(Xs, t)...) for t ∈ 1:newT]
end

"""
    batch_targets(ys::VV, output_dim::Integer, maxT::Integer = maximum(length, ys))::VV where VV <: DenseVector{<:DenseVector{<:Integer}}

Given a batch vector of target sequences `ys` returns a vector of corresponding linear indexes into the prediction Ŷs, which is assumed to be a vector of length T whose tᵗʰ element is a D×B matrix of predictions at time t across all sequences in a given batch.
Here T is the maximum time length in the batch, D is the dimensionality of the output and B is the batch size.
"""
function batch_targets(ys::VV, output_dim::Integer, maxT::Integer = maximum(length, ys))::VV where VV <: DenseVector{<:DenseVector{<:Integer}}
   batch_size = length(ys)
   linidxs = similar(ys, maxT)
   idxs = similar(first(ys), batch_size)
   offsets = range(0; step=output_dim, length=batch_size)
   @views for t ∈ 1:maxT
      n = 0
      for (y, offset) ∈ zip(ys, offsets)
         if t <= length(y)
            n += 1
            idxs[n] = offset + y[t]
         end
      end
      linidxs[t] = idxs[1:n]
   end
   return linidxs
end

"""
    batch(Xs::DenseVector{<:DenseVector{<:DenseVector}}, ys::DenseVector{<:DenseVector}, batch_size::Integer, multiplicity::Integer)

Arranges dataset into batches such that the number of batches approximately equals the ratio of dataset size to `batch_size`.
Batches are formed by first sorting sequences in the dataset according to their length (which minimizes the total number of elements to pad in inputs) and then partitioning the result into batches such that each batch approximately the same total number of sequence elements (this ensures that each batch takes up the same amount of memory, so as to avoid memory overflow).
"""
function batch(Xs::DenseVector{<:DenseVector{<:DenseVector}},
               ys::DenseVector{<:DenseVector},
               output_dim::Integer,
               batch_size::Integer,
               multiplicity::Integer)

   sortidxs = sortperm(Xs; by=length)
   Xs, ys = Xs[sortidxs], ys[sortidxs]

   cumseqlengths = cumsum(length.(ys))
   nbatches = ceil(Int, length(Xs) / batch_size)
   # subtract 0.5 from the last element of the range
   # to ensure that i index inside the loop won't go out of bounds due to floating point rounding errors
   cum_n_elts_rng = range(0, last(cumseqlengths)-0.5; length = nbatches+1)[2:end]
   lastidxs = similar(sortidxs, nbatches)
   i = 1
   for (n, cum_n_elts_for_batch) ∈ enumerate(cum_n_elts_rng)
      while cumseqlengths[i] < cum_n_elts_for_batch
         i += 1
      end
      lastidxs[n] = i
   end
   firstidxs = [1; @view(lastidxs[1:(end-1)]) .+ 1]

   maxTs = length.(@view Xs[lastidxs])
   xs_batches = [ batch_inputs!(Xs[firstidx:lastidx], multiplicity, maxT) for (firstidx, lastidx, maxT) ∈ zip(firstidxs, lastidxs, maxTs) ]
   linidxs_batches = [ batch_targets(ys[firstidx:lastidx], output_dim, maxT) for (firstidx, lastidx, maxT) ∈ zip(firstidxs, lastidxs, maxTs) ]
   return xs_batches, linidxs_batches, maxTs
end

# dim_encoding  = (512, 512, 512, 512)
# dim_attention = 512
# dim_decoding  = 512
# dim_feed_forward = 128
# dim_LSTM_speller = 512
# initialize with uniform(-0.1, 0.1)

# const las, PHONEMES = let
#    JLD2.@load "/Users/aza/Projects/LAS/data/TIMIT/TIMIT_MFCC/data_train.jld" PHONEMES
#
#    encoder_dims = (
#       blstm       = (in = 39, out = 2),
#       pblstms_out = (3, 4, 5)
#    )
#    attention_dim = 6
#    decoder_out_dims = (7, 8)
#    out_dim = 61
#
#    las = LAS(encoder_dims, attention_dim, decoder_out_dims, out_dim)
#    las, PHONEMES
# end

function loss(m::LAS, xs::DenseVector{<:DenseMatrix{<:Real}}, linidxs::DenseVector{<:DenseVector{<:Integer}})::Real
   ŷs = m(xs, length(linidxs))
   l = -sum(sum.(getindex.(ŷs, linidxs)))
   return l
end

function loss(m::LAS, xs_batches::DenseVector{<:DenseVector{<:DenseMatrix{<:Real}}},
         linidxs_batches::DenseVector{<:DenseVector{<:DenseVector{<:Integer}}})::Real
   return sum(loss.(m, xs_batches, linidxs_batches))
end

# best path decoding
function predict(m::LAS, xs::DenseVector{<:DenseMatrix{<:Real}}, lengths::DenseVector{<:Integer}, labels)::DenseVector{<:DenseVector}
   maxT = maximum(lengths)
   Ŷs = m(gpu.(xs), maxT) |> cpu
   predictions = [onecold(@view(Ŷs[:, 1:len, n]), labels) for (n, len) ∈ enumerate(lengths)]
   return predictions
end

function predict(m::LAS, xs::DenseVector{<:DenseVector{<:Real}}, labels)::DenseVector
   Ŷ = m(xs) |> cpu
   prediction = onecold(Ŷ, labels)
   return prediction
end

function main(; saved_results::Bool=false)
   # load data & construct the neural net
   las, phonemes,
   Xs_train, linidxs_train, maxTs_train,
   Xs_val,   linidxs_val,   maxTs_val =
   let batch_size = 77, valsetsize = 344
      JLD2.@load "/Users/aza/Projects/LAS/data/TIMIT/TIMIT_MFCC/data_train.jld" PHONEMES
      JLD2.@load "/Users/aza/Projects/LAS/data/TIMIT/TIMIT_MFCC/data_train.jld" Xs ys

      encoder_dims = (
         blstm       = (in = (length ∘ first ∘ first)(Xs), out = 64),
         pblstms_out = (128, 128, 64)
      )
      attention_dim = 128
      decoder_out_dims = (256, 256)
      out_dim = length(PHONEMES)
      las = LAS(encoder_dims, attention_dim, decoder_out_dims, out_dim)

      multiplicity = time_squashing_factor(las)
      Xs_train, linidxs_train, maxTs_train = batch(Xs, ys, out_dim, batch_size, multiplicity)

      JLD2.@load "/Users/aza/Projects/LAS/data/TIMIT/TIMIT_MFCC/data_test.jld" Xs ys
      idxs_val = sample(eachindex(Xs), valsetsize; replace=false, ordered=true)
      Xs_val, linidxs_val, maxTs_val = batch(Xs[idxs_val], ys[idxs_val], out_dim, batch_size, multiplicity)

      las, PHONEMES,
      Xs_train, linidxs_train, maxTs_train,
      Xs_val,   linidxs_val,   maxTs_val
   end

   θ = Flux.params(las)
   optimiser = ADAM()
   # optimiser = Flux.RMSProp(0.0001)
   # optimiser = AMSGrad()
   # optimiser = AMSGrad(0.0001)
   # optimiser = AMSGrad(0.00001)

   if saved_results
      JLD2.@load "/Users/aza/Projects/LAS/models/TIMIT/LAS.jld2" loss_val_saved
   else
      loss_val_saved = (eltype ∘ eltype ∘ eltype)(Xs_train)(Inf)
   end

   n_epochs = 3
   for epoch ∈ 1:n_epochs
      @info "Starting training for epoch $epoch"
      duration = @elapsed for (xs, linidxs) ∈ zip(Xs_train, linidxs_train)
         # move current batch to GPU
         xs = gpu.(xs)
         l, pb = Flux.pullback(θ) do
            loss(las, xs, linidxs)
         end
         dldθ = pb(one(l))
         Flux.Optimise.update!(optimiser, θ, dldθ)
         @show l
      end
      duration = round(duration / 60; sigdigits = 2)
      @info "Finished training for epoch $epoch in $duration minutes"
      loss_val = loss(las, Xs_val, ys_val)
      @info "Validation loss after epoch $epoch is $loss_val"
      if loss_val < loss_val_saved
         loss_val_saved = loss_val
         @save "/Users/aza/Projects/LAS/models/TIMIT/LAS.jld2" las optimiser loss_val_saved
         @info "Saved training results after epoch $epoch to: /Users/aza/Projects/LAS/models/TIMIT/LAS.jld2"
      end
   end
end


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
      for (i, el₁, el₂) ∈ zip(0:lenseq₁, reverse(seq₁), reverse(seq₂))
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

end # ListenAttendSpell
