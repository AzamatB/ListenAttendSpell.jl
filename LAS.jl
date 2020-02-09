# Listen, Attend and Spell: arxiv.org/abs/1508.01211

module ListenAttendSpell

using CuArrays
CuArrays.allowscalar(false)

using Flux
using Flux: reset!, onecold, @functor, Recur, LSTMCell
using Zygote
using Zygote: Buffer, @adjoint
using LinearAlgebra
using JLD2
using IterTools
using Base.Iterators: reverse
using OMEinsum

"""
    BLSTM(in::Integer, out::Integer)

Constructs a bidirectional LSTM layer.
"""
struct BLSTM{M <: DenseMatrix, V <: DenseVector}
   forward  :: Recur{LSTMCell{M,V}}
   backward :: Recur{LSTMCell{M,V}}
   dim_out  :: Int
end

@functor BLSTM (forward, backward)

function BLSTM(in::Integer, out::Integer)
   forward  = LSTM(in, out)
   backward = LSTM(in, out)
   return BLSTM(forward, backward, out)
end

function BLSTM(forward::Recur{LSTMCell{M,V}}, backward::Recur{LSTMCell{M,V}}) where {M <: DenseMatrix, V <: DenseVector}
    size(forward.cell.Wi, 2) == size(backward.cell.Wi, 2) || throw(DimensionMismatch("input dimension, $(size(forward.cell.Wi, 2)), of the forward-time LSTM layer does not match the input dimension, $(size(backward.cell.Wi, 2)), of the backward-time LSTM layer"))

    out_dim = length(forward.cell.h)
    out_dim == length(backward.cell.h) || throw(DimensionMismatch("output dimension, $out_dim, of the forward-time LSTM layer does not match the output dimension, $(length(backward.cell.h)), of the backward-time LSTM layer"))
    return BLSTM(forward, backward, out_dim)
end

Base.show(io::IO, l::BLSTM)  = print(io,  "BLSTM(", size(l.forward.cell.Wi, 2), ", ", l.dim_out, ")")

function flip(f, xs)
   rev_time = reverse(eachindex(xs))
   return getindex.(Ref(f.(getindex.(Ref(xs), rev_time))), rev_time)
   # the same as
   # flipped_xs = Buffer(xs)
   # @inbounds for t ∈ rev_time
   #    flipped_xs[t] = f(xs[t])
   # end
   # return copy(flipped_xs)
   # but implemented via broadcasting as Zygote differentiates loops much slower than broadcasting
end

"""
    (m::BLSTM)(xs::DenseVector{<:DenseVecOrMat}) -> DenseVector{<:DenseVecOrMat}

Forward pass of the bidirectional LSTM layer for a vector of matrices input.
Input must be a vector of length T (time duration), whose each element is a matrix of size D×B (input dimension × # of batches).
"""
function (m::BLSTM)(xs::VM)::VM where VM <: DenseVector
   vcat.(m.forward.(xs), flip(m.backward, xs))
end

"""
    (m::BLSTM)(Xs::DenseArray{<:Real,3}) -> DenseArray{<:Real,3}

Forward pass of the bidirectional LSTM layer for a 3D tensor input.
Input tensor must be arranged in D×T×B (input dimension × time duration × # batches) order.
"""
function (m::BLSTM)(Xs::T₃)::T₃ where T₃ <: DenseArray{<:Real,3}
   # preallocate output buffer
   Ys = Buffer(Xs, 2m.dim_out, size(Xs,2), size(Xs,3))
   axisYs₁ = axes(Ys, 1)
   time    = axes(Ys, 2)
   rev_time = reverse(time)
   @inbounds begin
      # get forward and backward slice indices
      slice_f = axisYs₁[1:m.dim_out]
      slice_b = axisYs₁[(m.dim_out+1):end]
      # bidirectional run step
      setindex!.(Ref(Ys),  m.forward.(view.(Ref(Xs), :, time, :)), Ref(slice_f), time, :)
      setindex!.(Ref(Ys), m.backward.(view.(Ref(Xs), :, rev_time, :)), Ref(slice_b), rev_time, :)
      # the same as
      # @views for (t_f, t_b) ∈ zip(time, rev_time)
      #    Ys[slice_f, t_f, :] =  m.forward(Xs[:, t_f, :])
      #    Ys[slice_b, t_b, :] = m.backward(Xs[:, t_b, :])
      # end
      # but implemented via broadcasting as Zygote differentiates loops much slower than broadcasting
   end
   return copy(Ys)
end

# Flux.reset!(m::BLSTM) = reset!((m.forward, m.backward)) # not needed as taken care of by @functor

"""
    PBLSTM(in::Integer, out::Integer)

Constructs pyramid BLSTM layer, which is the same as the BLSTM layer, with the addition that the input is first concatenated at every two consecutive time steps before feeding it to the usual BLSTM layer.
"""
struct PBLSTM{M <: DenseMatrix, V <: DenseVector}
   forward  :: Recur{LSTMCell{M,V}}
   backward :: Recur{LSTMCell{M,V}}
   dim_out  :: Int
end

@functor PBLSTM (forward, backward)

function PBLSTM(in::Integer, out::Integer)
   forward  = LSTM(2in, out)
   backward = LSTM(2in, out)
   return PBLSTM(forward, backward, out)
end

function PBLSTM(forward::Recur{LSTMCell{M,V}}, backward::Recur{LSTMCell{M,V}}) where {M <: DenseMatrix, V <: DenseVector}
   size(forward.cell.Wi, 2) == size(backward.cell.Wi, 2) || throw(DimensionMismatch("input dimension, $(size(forward.cell.Wi, 2)), of the forward-time LSTM layer does not match the input dimension, $(size(backward.cell.Wi, 2)), of the backward-time LSTM layer"))

   out_dim = length(forward.cell.h)
   out_dim == length(backward.cell.h) || throw(DimensionMismatch("output dimension, $out_dim, of the forward-time LSTM layer does not match the output dimension, $(length(backward.cell.h)), of the backward-time LSTM layer"))
   return PBLSTM(forward, backward, out_dim)
end

Base.show(io::IO, l::PBLSTM) = print(io, "PBLSTM(", size(l.forward.cell.Wi, 2)÷2, ", ", l.dim_out, ")")

"""
    (m::PBLSTM)(xs::DenseVector{<:DenseVecOrMat}) -> DenseVector{<:DenseVecOrMat}

Forward pass of the pyramid BLSTM layer for a vector of matrices input.
Input must be a vector of length T (time duration), whose each element is a matrix of size D×B (input dimension × # of batches).
"""
function (m::PBLSTM)(xs::VM)::VM where VM <: DenseVector
   # reduce time duration by half by restacking consecutive pairs of input along the time dimension
   x̄s = [@inbounds [xs[i-1]; xs[i]] for i ∈ 2:2:lastindex(xs)]
   # counterintuitively the gradient of the following version is not much faster (on par in fact),
   # even though it is implemented via broadcasting
   # x̄s = vcat.(getindex.(Ref(xs), 1:2:lastindex(xs)), getindex.(Ref(xs), 2:2:lastindex(xs)))
   # x̄s = @views @inbounds(vcat.(xs[1:2:end], xs[2:2:end]))
   # x̄s = vcat.(xs[1:2:end], xs[2:2:end])
   # bidirectional run step
   return vcat.(m.forward.(x̄s), flip(m.backward, x̄s))
end

"""
    (m::PBLSTM)(Xs::DenseArray{<:Real,3}) -> DenseArray{<:Real,3}

Forward pass of the pyramid BLSTM layer for a 3D tensor input.
Input tensor must be arranged in D×T×B (input dimension × time duration × # batches) order.
"""
function (m::PBLSTM)(Xs::T₃)::T₃ where T₃ <: DenseArray{<:Real,3}
   D, T, B = size(Xs)
   T½ = T÷2
   # reduce time duration by half by restacking consecutive pairs of input along the time dimension
   X̄s = reshape(Xs, 2D, T½, B)
   # preallocate output buffer
   Ys = Buffer(Xs, 2m.dim_out, T½, B)
   axisYs₁ = axes(Ys, 1)
   time    = axes(Ys, 2)
   rev_time = reverse(time)
   # get forward and backward slice indices
   slice_f = axisYs₁[1:m.dim_out]
   slice_b = axisYs₁[(m.dim_out+1):end]
   # bidirectional run step
   setindex!.(Ref(Ys),  m.forward.(view.(Ref(X̄s), :, time, :)), Ref(slice_f), time, :)
   setindex!.(Ref(Ys), m.backward.(view.(Ref(X̄s), :, rev_time, :)), Ref(slice_b), rev_time, :)
   # the same as
   # @views for (t_f, t_b) ∈ zip(time, reverse(time))
   #    Ys[slice_f, t_f, :] =  m.forward(X̄s[:, t_f, :])
   #    Ys[slice_b, t_b, :] = m.backward(X̄s[:, t_b, :])
   # end
   # but implemented via broadcasting as Zygote differentiates loops much slower than broadcasting
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
   (length(dims.pblstms_out) >= 1) || throw("Encoder must have at least 1 pyramid BLSTM layer")

   pblstm_layers = ( PBLSTM(2in, out) for (in, out) ∈ partition(dims.pblstms_out, 2, 1) )
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

CharacterDistribution(in::Integer, out::Integer) = Chain(Dense(in, out), logsoftmax)

mutable struct State{M <: DenseMatrix}
   context     :: M   # last attention context
   decoding    :: M   # last decoder state
   prediction  :: M   # last prediction
   # reset values
   context₀    :: M
   decoding₀   :: M
   prediction₀ :: M
   dim         :: Int
end

@functor State (context₀, decoding₀, prediction₀)

function State(dim_c::Integer, dim_d::Integer, dim_p::Integer)
   context₀    = zeros(Float32, dim_c, 1) |> gpu
   decoding₀   = zeros(Float32, dim_d, 1) |> gpu
   prediction₀ = zeros(Float32, dim_p, 1) |> gpu
   dim = dim_c + dim_d + dim_p
   return State(context₀, decoding₀, prediction₀, context₀, decoding₀, prediction₀, dim)
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
   listen      :: E        # encoder function
   attention_ψ :: Aψ       # keys attention context function
   attention_ϕ :: Aϕ       # query attention context function
   spell       :: D        # LSTM decoder
   infer       :: C        # character distribution inference function
end

@functor LAS

function LAS(encoder_dims::NamedTuple,
             attention_dim::Integer,
             decoder_out_dims::Tuple{Integer,Integer},
             out_dim::Integer)

   dim_encoding = 2last(encoder_dims.pblstms_out)
   dim_decoding =  last(decoder_out_dims)

   state       = State(dim_encoding, dim_decoding, out_dim)
   listen      = Encoder(encoder_dims) |> gpu
   attention_ψ = MLP(dim_encoding, attention_dim) |> gpu
   attention_ϕ = MLP(dim_decoding, attention_dim) |> gpu
   spell       = Decoder(dim_encoding + dim_decoding + out_dim, decoder_out_dims) |> gpu
   infer       = CharacterDistribution(dim_encoding + dim_decoding, out_dim) |> gpu

   las = LAS(state, listen, attention_ψ, attention_ϕ, spell, infer)
   return las
end

function LAS(encoder_dims::NamedTuple,
             attention_dim::Integer,
             decoder_out_dim::Integer,
             out_dim::Integer)
   decoder_out_dim₁ = last(encoder_dims.pblstms_out) + decoder_out_dim + out_dim÷2
   decoder_out_dims = (decoder_out_dim₁, decoder_out_dim)
   LAS(encoder_dims, attention_dim, decoder_out_dims, out_dim)
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

time_squashing_factor(m::LAS) = 2^(length(m.listen) - 1)

@inline function decode(m::LAS{M}, Hs::DenseArray{R,3}, maxT::Integer)::Vector{M} where {M <: DenseMatrix, R <: Real}
   batch_size = size(Hs, 3)
   # precompute keys ψ(H) by gluing the slices of Hs along the batch dimension into a single D×TB matrix, then
   # passing it through the ψ dense layer in a single pass and then reshaping the result back into D′×T×B tensor
   ψHs = reshape(m.attention_ψ(reshape(Hs, size(Hs,1), :)), size(m.attention_ψ.W, 1), :, batch_size)
   # ψhs = m.attention_ψ.(getindex.(Ref(Hs), :, axes(Hs,2), :))
   # check: all(ψhs .≈ eachslice(ψHs; dims=2))
   # batchify state, i.e. initialize a state for each sequence in the batch
   m.state.decoding   = m.state.decoding   .+ gpu(zeros(R, size(m.state.decoding, 1),   batch_size))::M
   m.state.prediction = m.state.prediction .+ gpu(zeros(R, size(m.state.prediction, 1), batch_size))::M
   m.state.context    = m.state.context    .+ gpu(zeros(R, size(m.state.context, 1),    batch_size))::M
   ŷs = map(1:maxT) do _
      # compute decoder state
      m.state.decoding = m.spell([m.state.decoding; m.state.prediction; m.state.context])
      # compute query ϕ(sᵢ)
      ϕsᵢ = m.attention_ϕ(m.state.decoding)
      # compute energies via batch matrix multiplication
      @ein Eᵢs[t,b] := ϕsᵢ[d,b] * ψHs[d,t,b]
      # check: Eᵢs ≈ reduce(hcat, diag.((ϕsᵢ',) .* ψhs))'
      # compute attentions weights
      αᵢs = softmax(Eᵢs)
      # compute attended context using Einstein summation convention, i.e. contextᵢ = Σᵤαᵢᵤhᵤ
      @ein m.state.context[d,b] := αᵢs[t,b] * Hs[d,t,b]
      # check: m.state.context ≈ reduce(hcat, [sum(αᵢs[t,b] *Hs[:,t,b] for t ∈ axes(αᵢs, 1)) for b ∈ axes(αᵢs,2)])
      # predict probability distribution over character alphabet
      m.state.prediction = m.infer([m.state.decoding; m.state.context])
   end
   return ŷs
end

function (m::LAS{M})(xs::DenseVector{<:DenseMatrix}, maxT::Integer = length(xs))::Vector{M} where {M <: DenseMatrix}
   # compute input encoding, which are also values for the attention layer
   hs = m.listen(xs)
   dim_out, batch_size = size(first(hs))
   # transform T-length sequence of D×B matrices into the D×T×B tensor by first conconcatenating matrices
   # along the 1st dimension and to get singe DT×B matrix and then reshaping it into D×T×B tensor
   Hs = reshape(reduce(vcat, hs), dim_out, :, batch_size)
   # perform attend and spell steps
   ŷs = decode(m, Hs, maxT)
   reset!(m)
   return ŷs
end

function (m::LAS{M})(Xs::DenseArray{<:Real,3}, maxT::Integer = size(Xs,2))::Vector{M} where {M <: DenseMatrix}
   # compute input encoding, which are also values for the attention layer
   Hs = m.listen(Xs)
   # perform attend and spell steps
   ŷs = decode(m, Hs, maxT)
   reset!(m)
   return ŷs
end

function (m::LAS)(xs::VV)::VV where VV <: DenseVector{<:DenseVector}
   T = length(xs)
   Xs = reshape(reduce(hcat, pad(xs, time_squashing_factor(m))), Val(3)) |> gpu
   ŷs = dropdims.(m(Xs, T); dims=2)
   return ŷs
end

function pad(xs::DenseVector{<:DenseVector}, multiplicity::Integer)
   T = length(xs)
   newT = ceil(Int, T / multiplicity)multiplicity
   z = zero(first(xs))
   return [xs; (_ -> z).(1:(newT - T))]
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
   z = (similar ∘ first ∘ first)(Xs)
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

function loss(m::LAS, xs::DenseVector{<:DenseMatrix{<:Real}}, linidxs::DenseVector{<:DenseVector{<:Integer}})::Real
   ŷs = m(xs, length(linidxs))
   l = -sum(sum.(getindex.(ŷs, linidxs)))
   return l
end

function loss(m::LAS, xs_batches::DenseVector{<:DenseVector{<:DenseMatrix{<:Real}}},
         linidxs_batches::DenseVector{<:DenseVector{<:DenseVector{<:Integer}}})::Real
   return sum(loss.((m,), xs_batches, linidxs_batches))
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

function main(; n_epochs::Integer=1, saved_results::Bool=false)
   # load data & construct the neural net
   las, phonemes,
   Xs_train, linidxs_train, maxTs_train,
   Xs_val,   linidxs_val,   maxTs_val =
   let batch_size = 77, valsetsize = 344
      JLD2.@load "data/TIMIT/TIMIT_MFCC/data_train.jld2" Xs ys PHONEMES
      out_dim = length(PHONEMES)

      if saved_results
         JLD2.@load "ListenAttendSpell/models/TIMIT/las.jld2" las loss_val_saved
      else
         # encoder_dims = (
         #    blstm       = (in = (length ∘ first ∘ first)(Xs), out = 2),
         #    pblstms_out = (2, 2, 2)
         # )
         # attention_dim = 2
         # decoder_out_dim = 2
         encoder_dims = (
            blstm       = (in = (length ∘ first ∘ first)(Xs), out = 128),
            pblstms_out = (128, 128, 128)
         )
         attention_dim   = 128
         decoder_out_dim = 128
         las = LAS(encoder_dims, attention_dim, decoder_out_dim, out_dim)
      end

      multiplicity = time_squashing_factor(las)
      Xs_train, linidxs_train, maxTs_train = batch(Xs, ys, out_dim, batch_size, multiplicity)

      JLD2.@load "data/TIMIT/TIMIT_MFCC/data_test.jld2" Xs ys
      Xs_val, linidxs_val, maxTs_val = batch(Xs[1:valsetsize], ys[1:valsetsize], out_dim, batch_size, multiplicity)

      las, PHONEMES,
      Xs_train, linidxs_train, maxTs_train,
      Xs_val,   linidxs_val,   maxTs_val
   end

   θ = Flux.params(las)
   # 1. optimiser = RMSProp()       # 5 epochs
   # 2. optimiser = ADAM()          # 2 epochs
   # 3. optimiser = ADAM(0.0002)    # 1 epoch
   # 4. optimiser = RMSProp(0.0001) # 2 epochs
   optimiser = NADAM(0.0001)     # 2 epochs
   # optimiser = AMSGrad(0.0001)       # 2 epoch
   # optimiser = AMSGrad(0.0001)
   # optimiser = AMSGrad(0.00001)

   loss_val_saved = loss(las, Xs_val, linidxs_val)
   @info "Validation loss before start of the training is $loss_val_saved"

   nds = ndigits(length(Xs_train))
   for epoch ∈ 1:n_epochs
      @info "Starting to train epoch $epoch with optimiser $(typeof(optimiser))$(getproperty.(Ref(optimiser), propertynames(optimiser)[1:end-1]))"
      println(typeof(optimiser), getproperty.(Ref(optimiser), propertynames(optimiser)[1:end-1]))
      duration = @elapsed for (n, (xs, linidxs)) ∈ enumerate(zip(Xs_train, linidxs_train))
         # move current batch to GPU
         xs = gpu.(xs)
         l, pb = Flux.pullback(θ) do
            loss(las, xs, linidxs)
         end
         println("Loss for a batch # ", ' '^(nds - ndigits(n)), n, " is ", l)
         dldθ = pb(one(l))
         Flux.Optimise.update!(optimiser, θ, dldθ)
      end
      duration = round(duration / 60; sigdigits = 1)
      @info "Completed training epoch $epoch in $duration minutes"
      loss_val = loss(las, Xs_val, linidxs_val)
      @info "Validation loss after training epoch $epoch is $loss_val"
      if loss_val < loss_val_saved
         loss_val_saved = loss_val
         JLD2.@save "ListenAttendSpell/models/TIMIT/las.jld2" las loss_val_saved
         @info "Saved results after training epoch $epoch to ListenAttendSpell/models/TIMIT/las.jld2"
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
