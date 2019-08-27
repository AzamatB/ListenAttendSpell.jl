# Listen, Attend and Spell: arxiv.org/abs/1508.01211
using Flux
using Flux: flip, reset!, onehotbatch, throttle, train!, @treelike, @epochs
using IterTools
using JLD2
using StatsBase
# using CuArrays


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

(m::BLSTM)(xs::AbstractVector)::AbstractVector = m.dense.(vcat.(m.forward.(xs), flip(m.backward, xs)))

Flux.reset!(m::BLSTM) = Flux.reset!((m.forward, m.backward))

function restack(xs::AbstractVector)::AbstractVector
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


function CharacterDistribution(D_in::Integer, D_out::Integer, σ::Function; nlayers::Integer, log::Bool=true)
   f = log ? logsoftmax : softmax
   layer_sizes = ceil.(Int, range(D_in, D_out; length=nlayers+1))
   layer_dims = Tuple(partition(layer_sizes, 2, 1))
   layers = ( Dense(D_in, D_out, σ) for (D_in, D_out) ∈ layer_dims[1:end-1] )
   return Chain(layers..., Dense(layer_dims[end]...), f)
end

CharacterDistribution(D_in::Integer, D_out::Integer; log::Bool=true) = Chain(Dense(D_in, D_out), log ? logsoftmax : softmax)


mutable struct State{T <: AbstractVector}
   context    :: T   # last attention context
   decoding   :: T   # last decoder state
   prediction :: T   # last prediction
end

@treelike State

function Flux.reset!(s::State)
   s.context    = param(zero(s.context))
   s.decoding   = param(zero(s.decoding))
   s.prediction = param(zero(s.prediction))
end


struct LAS{E, Dϕ, Dψ, L, C, S}
   listen      :: E   # encoder function
   attention_ϕ :: Dϕ  # attention context function
   attention_ψ :: Dψ  # attention context function
   spell       :: L   # RNN decoder
   infer       :: C   # character distribution inference function
   state       :: S   # current state of the model
end

@treelike LAS

function LAS(D_in::Integer, D_out::Integer;
             D_encoding::Integer,
             D_attention::Integer,
             D_decoding::Integer)
   listen      = Encoder(D_in, D_encoding)
   attention_ϕ = MLP(D_decoding, D_attention)
   attention_ψ = MLP(D_encoding, D_attention)
   spell       = Decoder(D_encoding + D_decoding + D_out, D_decoding)
   infer       = CharacterDistribution(D_encoding + D_decoding, D_out)
   context₀    = param(zeros(Float32, D_encoding))
   decoding₀   = param(zeros(Float32, D_decoding))
   prediction₀ = param(zeros(Float32, D_out))
   las = LAS(listen, attention_ϕ, attention_ψ, spell, infer, State(context₀, decoding₀, prediction₀)) |> gpu
   return las
end

function (m::LAS)(X::AbstractMatrix)::AbstractMatrix
   # compute input encoding
   H = m.listen(gpu(pad(X)))
   # precompute ψ(H)
   ψH = m.attention_ψ(H)
   # initialize prediction
   Ŷ = m.state.prediction[:,1:0]
   for _ ∈ axes(X,2)
      # compute decoder state
      m.state.decoding = m.spell([m.state.decoding; m.state.prediction; m.state.context])
      # compute attention context
      αs = softmax(ψH'm.attention_ϕ(m.state.decoding))
      m.state.context = sum( α * h for (α, h) ∈ zip(αs, eachcol(H)) )
      # predict probability distribution over character alphabet
      m.state.prediction = m.infer([m.state.decoding; m.state.context])
      # concatenate with previous character predictions
      Ŷ = [Ŷ m.state.prediction]
   end
   # reset!(m::LAS)
   return Ŷ
end

function Flux.reset!(m::LAS)
   reset!(m.listen)
   reset!(m.spell)
   reset!(m.state)
   return nothing
end

function pad(X::AbstractMatrix; multiplicity::Integer=8)::AbstractMatrix
   T = size(X,2)
   Δ = ceil(Int, T / multiplicity)multiplicity - T
   x_min = minimum(X)
   X = [X fill(x_min, size(X,1), Δ)]
   return X
end


# load data
X, y, Xs_train, ys_train, Xs_eval, ys_eval, Xs_val, ys_val, Xs_test, ys_test, PHN2IDX, PHN_IDCS = let val_set_size = 32
   JLD2.@load "/Users/Azamat/Projects/LAS/data/TIMIT/TIMIT_MFCC/data_test.jld" Xs ys PHN2IDX PHN_IDCS
   Xs_val, ys_val, Xs_test, ys_test = Xs[1:val_set_size], ys[1:val_set_size], Xs[(val_set_size+1):end], ys[(val_set_size+1):end]
   JLD2.@load "/Users/Azamat/Projects/LAS/data/TIMIT/TIMIT_MFCC/data_train.jld" Xs ys PHN2IDX PHN_IDCS
   eval_idcs = sample(eachindex(ys), val_set_size; replace=false)
   Xs_eval, ys_eval = Xs[eval_idcs], ys[eval_idcs]
   first(Xs), first(ys), Xs, ys, Xs_eval, ys_eval, Xs_val, ys_val, Xs_test, ys_test, PHN2IDX, PHN_IDCS
end


const D_x = size(X,1)
const D_y = length(PHN_IDCS)

D_encoding = 128
D_attention = 64 # attention dimension
D_decoding = 256

D_feed_forward = 128
D_LSTM_speller = 512


const las = LAS(D_x, D_y; D_encoding=D_encoding, D_attention=D_attention, D_decoding=D_decoding)

function loss(X::AbstractMatrix, y::AbstractVector)::Real
   Ŷ = las(X)
   # compute l = -sum( ŷ[i] for (ŷ, i) ∈ zip(eachcol(Ŷ), y) )
   # in a GPU friendly way, i.e. without scalar indexing
   num_phn = size(Ŷ,1)
   idcs = map((n, i) -> num_phn*n + i, 0:(length(y)-1), y)
   l = -sum(Ŷ[idcs])
   return l
end

loss(Xs::AbstractVector{<:AbstractMatrix}, ys::AbstractVector{<:AbstractVector})::Real = sum(loss.(Xs, ys))

show_loss_val() = @show(loss(Xs_val, ys_val))
show_loss_eval() = @show(loss(Xs_eval, ys_eval))

@time las(X)
@time @show loss(Xs_val, ys_val)
@time @show loss(Xs_eval, ys_eval)

θ = params(las)
optimizer = ADAM()
data = zip(Xs_train, ys_train)

@time @epochs 10 begin
   @time train!(loss, θ, data, optimizer;
      cb = throttle(show_loss_eval, 60))
   loss_val = loss(X_val, Y_val)
   @show loss_val
   if loss_val < loss_val_saved
      loss_val_saved = loss_val
      @save "/Users/Azamat/Projects/LAS/models/TIMIT/LAS.jld2" model optimiser
   end
end





# beamsearch()

using Levenstein

function cer()
end

function wer()
end
