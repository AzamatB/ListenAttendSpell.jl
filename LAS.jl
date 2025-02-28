# Listen, Attend and Spell: arxiv.org/abs/1508.01211

module ListenAttendSpell

using CuArrays
using Flux
using Flux: reset!, onecold, @functor, Recur, LSTMCell
using Zygote
using Zygote: Buffer, @adjoint
using LinearAlgebra
using JLD2
using FileIO
using IterTools
using Base.Iterators: reverse
using OMEinsum
using Logging
using TensorBoardLogger
using StatsBase

CuArrays.allowscalar(false)

include("utils.jl")

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

Base.show(io::IO, l::BLSTM) = print(io,  "BLSTM(", size(l.forward.cell.Wi, 2), ", ", l.dim_out, ")")

function flip(f, xs::V) where V <: AbstractVector
   rev_time = reverse(eachindex(xs))
   return getindex.(Ref(
      f.(getindex.(Ref(xs), rev_time))::V
   ), rev_time)
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
function (m::BLSTM)(xs::VM) where VM <: AbstractVector{<:DenseVecOrMat}
   vcat.(m.forward.(xs), flip(m.backward, xs))::VM
end

"""
    (m::BLSTM)(Xs::DenseArray{<:Real,3}) -> DenseArray{<:Real,3}

Forward pass of the bidirectional LSTM layer for a 3D tensor input.
Input tensor must be arranged in D×T×B (input dimension × time duration × # batches) order.
"""
function (m::BLSTM)(Xs::DenseArray{<:Real,3})
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
function (m::PBLSTM)(xs::VM) where VM <: AbstractVector{<:DenseVecOrMat}
   # reduce time duration by half by restacking consecutive pairs of input along the time dimension
   evenidxs = (firstindex(xs)+1):2:lastindex(xs)
   x̄s = (i -> [xs[i-1]; xs[i]]).(evenidxs)::VM
   # counterintuitively the gradient of the following version is not much faster (on par in fact),
   # even though it is implemented via broadcasting
   # x̄s = vcat.(getindex.(Ref(xs), 1:2:lastindex(xs)), getindex.(Ref(xs), 2:2:lastindex(xs)))
   # x̄s = @views @inbounds(vcat.(xs[1:2:end], xs[2:2:end]))
   # x̄s = vcat.(xs[1:2:end], xs[2:2:end])
   # bidirectional run step
   return vcat.(m.forward.(x̄s), flip(m.backward, x̄s))::VM
end

"""
    (m::PBLSTM)(Xs::DenseArray{<:Real,3}) -> DenseArray{<:Real,3}

Forward pass of the pyramid BLSTM layer for a 3D tensor input.
Input tensor must be arranged in D×T×B (input dimension × time duration × # batches) order.
"""
function (m::PBLSTM)(Xs::DenseArray{<:Real,3})
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

struct LAS{M <: DenseMatrix, E, Aϕ, Aψ, D, C}
   context₀  :: M  # trainable initial value of the context
   decoding₀ :: M  # trainable initial value of the decoding
   listen    :: E  # encoder function
   key_ψ     :: Aψ # keys attention context function
   query_ϕ   :: Aϕ # query attention context function
   spell     :: D  # LSTM decoder
   infer     :: C  # character distribution inference function
end

@functor LAS

function LAS(encoder_dims::NamedTuple,
             attention_dim::Integer,
             decoder_dims::Tuple{Integer,Integer},
             output_dim::Integer)

   encoding_dim = 2last(encoder_dims.pblstms_out)
   decoding_dim =  last(decoder_dims)
   context₀  = zeros(Float32, encoding_dim, 1)
   decoding₀ = zeros(Float32, decoding_dim, 1)
   listen  = Encoder(encoder_dims)
   key_ψ   = MLP(encoding_dim, attention_dim)
   query_ϕ = MLP(decoding_dim, attention_dim)
   spell   = Decoder(encoding_dim + decoding_dim + output_dim, decoder_dims)
   infer   = CharacterDistribution(encoding_dim + decoding_dim, output_dim)

   LAS(context₀, decoding₀, listen, key_ψ, query_ϕ, spell, infer) |> gpu
end

function LAS(encoder_dims::NamedTuple,
             attention_dim::Integer,
             decoder_out_dim::Integer,
             output_dim::Integer)
   decoder_out_dim₁ = last(encoder_dims.pblstms_out) + decoder_out_dim + out_dim÷2
   decoder_dims = (decoder_out_dim₁, decoder_out_dim)
   LAS(encoder_dims, attention_dim, decoder_dims, output_dim)
end

function Base.show(io::IO, m::LAS)
   input_dim     = size(first(m.listen).forward.cell.Wi, 2)
   encoding_dim  = size(m.context₀, 1)
   attention_dim = length(m.key_ψ.b)
   decoding_dim  = size(m.decoding₀, 1)
   output_dim    = length(first(m.infer).b)
   print(io,
      """
      LAS(
         # (input_dim=$input_dim, encoding_dim=$encoding_dim, attention_dim=$attention_dim, decoding_dim=$decoding_dim, output_dim=$output_dim)
         $(m.listen),
         $(m.key_ψ),
         $(m.query_ϕ),
         $(m.spell),
         $(m.infer)
      )
      """
   )
end

# Flux.reset!(m::LAS) = reset!((m.listen, m.spell)) # not needed as taken care of by @functor

time_squashing_factor(m::LAS) = 2^(length(m.listen) - 1)

# 0. rewrite `@ein context[d,b] := αᵢs[t,b] * Hs[d,t,b]` as `@ein context[d,b] := Hs[d,t,b] * αᵢs[t,b]` to elide the call to permutedims and ensure that `context` did not change.
# 1. include actual one-hot prediction instead of current just output of the network
# 2. add teacher forcing
# 3. add shufling batch order after every epoch and withing batches during construction


@inline function decode(m::LAS{M}, Hs::T, maxT::Integer) where {M <: DenseMatrix, T <: DenseArray{<:Real,3}}
   prediction′₀ = Zygote.bufferfrom(zeros(eltype(Hs), length(first(m.infer).b), 1))
   prediction′₀[1] = 1.0f0
   prediction₀ = M(copy(prediction′₀))
   M′ = addparent(M, T)
   return _decode(m, Hs, maxT, prediction₀, M′)
end

@inline function _decode(m::LAS{M}, Hs::DenseArray{<:Real,3}, maxT::Integer, prediction::M, M′::DataType) where M <: DenseMatrix
   batch_size = size(Hs, 3)
   # precompute keys ψ(H) by gluing the slices of Hs along the batch dimension into a single D×TB matrix, then
   # passing it through the ψ dense layer in a single pass and then reshaping the result back into D′×T×B tensor
   ψHs = reshape(m.key_ψ(reshape(Hs, size(Hs,1), :)), size(m.key_ψ.W, 1), :, batch_size)
   # ψhs = m.key_ψ.(getindex.(Ref(Hs), :, axes(Hs,2), :))
   # check: all(ψhs .≈ eachslice(ψHs; dims=2))
   # compute inital decoder state for the entire batch
   decoding = m.spell(repeat([m.decoding₀; prediction; m.context₀]::M, 1, batch_size))::M
   # allocate D×B×T output tensor
   Ŷs = Buffer(Hs, size(prediction, 1), batch_size, maxT)
   @inbounds for t ∈ axes(Ŷs, 3)
      # compute query ϕ(sᵢ)
      ϕsᵢ = m.query_ϕ(decoding)
      # compute energies via batch matrix multiplication
      # @ein Eᵢs[t,b] := ϕsᵢ[d,b] * ψHs[d,t,b]
      Eᵢs = einsum(EinCode{((1,2), (1,3,2)), (3,2)}(), (ϕsᵢ, ψHs))::M′ # dispatches to batched_contract
      # check: Eᵢs ≈ reduce(hcat, diag.((ϕsᵢ',) .* ψhs))'
      # compute attentions weights
      αᵢs = softmax(Eᵢs)
      # compute attended context using Einstein summation convention, i.e. contextᵢ = Σᵤαᵢᵤhᵤ
      # @ein context[d,b] := αᵢs[t,b] * Hs[d,t,b]
      context = einsum(EinCode{((1,2), (3,1,2)), (3,2)}(), (αᵢs, Hs))::M′ # dispatches to batched_contract
      # check: context ≈ reduce(hcat, [sum(αᵢs[t,b] *Hs[:,t,b] for t ∈ axes(αᵢs, 1)) for b ∈ axes(αᵢs,2)])
      # predict probability distribution over character alphabet
      Ŷs[:,:,t] = prediction = m.infer([decoding; context]::M)
      # compute decoder state
      decoding = m.spell([decoding; prediction; context]::M)::M
   end
   return copy(Ŷs)
end

function (m::LAS)(xs::AbstractVector{<:DenseMatrix}, maxT::Integer)
   # compute input encoding, which are also values for the attention layer
   hs = m.listen(xs)
   dim_out, batch_size = size(first(hs))
   # transform T-length sequence of D×B matrices into the D×T×B tensor by first conconcatenating matrices
   # along the 1st dimension and to get singe DT×B matrix and then reshaping it into D×T×B tensor
   Hs = reshape(reduce(vcat, hs), dim_out, :, batch_size)
   # perform attend and spell steps
   Ŷs = decode(m, Hs, maxT)
   return Ŷs
end

function (m::LAS)(Xs::DenseArray{<:Real,3}, maxT::Integer)
   # compute input encoding, which are also values for the attention layer
   Hs = m.listen(Xs)
   # perform attend and spell steps
   Ŷs = decode(m, Hs, maxT)
   return Ŷs
end

function (m::LAS)(x::AbstractVector{<:DenseVector})
   T = length(x)
   X = reshape(reduce(hcat, pad(x, time_squashing_factor(m))), Val(3))
   return m(X, T)
end

# encoding_dim  = (512, 512, 512, 512)
# attention_dim = 512
# decoding_dim  = 512
# dim_feed_forward = 128
# dim_LSTM_speller = 512
# initialize with uniform(-0.1, 0.1)

function loss(m::LAS, X::DenseArray{<:Real,3}, indices::DenseVector, maxT::Integer)
   reset!(m)
   Ŷs = m(X, maxT)
   l = -sum(Ŷs[indices])
   return l
end

function loss(m::LAS, dataset::AbstractVector{<:Tuple{DenseArray{<:Real,3}, DenseVector, Integer}})
   return sum(dataset) do (X, indices, maxT)
      loss(m, X, indices, maxT)
   end
end

# best path decoding
function predict(m::LAS, xs::DenseVector{<:DenseMatrix{<:Real}}, lengths::DenseVector{<:Integer}, labels)::DenseVector{<:DenseVector}
   maxT = maximum(lengths)
   Ŷs = m(xs, maxT) |> cpu
   predictions = [onecold(@view(Ŷs[:, 1:len, n]), labels) for (n, len) ∈ enumerate(lengths)]
   return predictions
end

function predict(m::LAS, xs::DenseVector{<:DenseVector{<:Real}}, labels)::DenseVector
   Ŷ = m(xs) |> cpu
   prediction = onecold(Ŷ, labels)
   return prediction
end

function main(; n_epochs::Integer=1, saved_results::Bool=false, log_weights::Bool=true)
# load data & construct the neural net
las, phonemes, data_trn,
data_evl, length_evl,
data_val, length_val =
let batch_size = 88, valsetsize = 352
   JLD2.@load "data/TIMIT/TIMIT_MFCC/data_train.jld2" Xs ys PHONEMES
   output_dim = length(PHONEMES)

   if saved_results
      JLD2.@load "ListenAttendSpell.jl/models/TIMIT/las.jld2" las loss_val_saved
   else
      # encoder_dims = (
      #    blstm       = (in = (length ∘ first ∘ first)(Xs), out = 17),
      #    pblstms_out = (7, 5, 3)
      # )
      # attention_dim = 11
      # decoder_out_dim = 13
      encoder_dims = (
         blstm       = (in = (length ∘ first ∘ first)(Xs), out = 240),
         pblstms_out = (230, 230, 230)
      )
      attention_dim   = 230
      decoder_out_dim = 230
      las = LAS(encoder_dims, attention_dim, decoder_out_dim, output_dim)
   end

   multiplicity = time_squashing_factor(las)
   data_trn = batch_dataset(Xs, ys, output_dim, batch_size, multiplicity)

   idxs_evl = sample(eachindex(ys), valsetsize; replace=false)
   ys_evl = ys[idxs_evl]
   data_evl = batch_dataset(Xs[idxs_evl], ys_evl, output_dim, batch_size, multiplicity)
   length_evl = sum(length, ys_evl)

   JLD2.@load "data/TIMIT/TIMIT_MFCC/data_test.jld2" Xs ys
   ys_val = ys[1:valsetsize]
   data_val = batch_dataset(Xs[1:valsetsize], ys_val, output_dim, batch_size, multiplicity)
   length_val = sum(length, ys_val)

   las, PHONEMES, data_trn,
   data_evl, length_evl,
   data_val, length_val
end

# initialize TensorBoard logger
tblogger = TBLogger("log", tb_overwrite)
loss_val_prev = loss_val_saved = loss(las, data_val)
optimiser = ADAM(0.0001)

if log_weights
   logweights = () -> begin
      params_dict = param_dict(las, "las")
      @info "model" params = params_dict log_step_increment=0
   end
else
   logweights = () -> nothing
end

function callback()
   loss_evl = loss(las, data_evl)
   loss_val = loss(las, data_val)
   truthprob_evl  = truthprob(loss_evl, length_evl)
   truthprob_val  = truthprob(loss_val, length_val)
   println()
   @show loss_evl loss_val truthprob_evl truthprob_val
   println()
   if loss_val < loss_val_saved
      loss_val_saved = loss_val
      save("ListenAttendSpell.jl/models/TIMIT/las.jld2", Dict("las" => cpu(las), "loss_val_saved" => loss_val_saved))
      @info "Saved results!"
      println()
   end

   # halve_η = false
   # if loss_val > loss_val_prev
   #    # if the validation error increased from previous epoch
   #    # then halve the learning rate
   #    halve_η = true
   # else # ask user if he still wants to halve the learning rate
   #    println("Do you want to halve the current learning rate of η = $(optimiser.eta)? [yes/\e[4mno\e[0m]")
   #    ans = "\n"
   #    @async(ans = readline(stdin))
   #    timedwait(() -> ans != "\n", 10.0; pollint=0.5)
   #    (lowercase(ans) ∈ ("yes", "y")) && (halve_η = true)
   # end
   # if halve_η
   #    # halve the learning rate
   #    η = max(optimiser.eta/2, 1e-5)
   #    optimiser.eta = η
   #    println()
   #    @info "Adjusted learning rate to:" η
   # end

   with_logger(tblogger) do
      logweights()
      @info "train" loss=loss_evl truthprob=truthprob_evl log_step_increment=0
      @info "valid" loss=loss_val truthprob=truthprob_val
   end
   loss_val_prev = loss_val
end

θ = Flux.params(las)

nds = ndigits(length(data_trn))
callback()
# main training loop
for epoch ∈ 1:n_epochs
   println()
   @info "Epoch $epoch:"
   duration = @elapsed for (n, (X, indices, maxT)) ∈ enumerate(data_trn)
      l, pb = Flux.pullback(θ) do
         loss(las, X, indices, maxT)
      end
      print("batch ", ' '^(nds - ndigits(n)), n, ": ")
      @show l
      θ̄ = pb(one(l))
      Flux.Optimise.update!(optimiser, θ, θ̄)
   end
   println("\nCompleted in ", round(duration/60; sigdigits=2), " minutes")
   callback()
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
