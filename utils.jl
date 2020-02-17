addparent(A::Type{<:AbstractArray}, P′::DataType) = A
addparent(::Type{CuArray{T,N,P}}, P′::DataType) where {T,N,P} = CuArray{T,N,P′}

const StatefulOptimiser = Union{Flux.Momentum, Flux.Nesterov, Flux.RMSProp, Flux.ADAM, Flux.RADAM, Flux.AdaMax, Flux.ADAGrad, Flux.ADADelta, Flux.AMSGrad, Flux.NADAM}

function Base.show(io::IO, optimiser::StatefulOptimiser)
    print(io, typeof(optimiser), getproperty.(Ref(optimiser), propertynames(optimiser)[1:end-1]))
end

@adjoint function reduce(::typeof(hcat), As::AbstractVector{<:AbstractVecOrMat})
   cumsizes = cumsum(size.(As, 2))
   return reduce(hcat, As), Δ -> (nothing, map((sz, A) -> Zygote.pull_block_horz(sz, Δ, A), cumsizes, As))
end

@adjoint function reduce(::typeof(vcat), As::AbstractVector{<:AbstractVecOrMat})
   cumsizes = cumsum(size.(As, 1))
   return reduce(vcat, As), Δ -> (nothing, map((sz, A) -> Zygote.pull_block_vert(sz, Δ, A), cumsizes, As))
end

@adjoint repeat(x::AbstractVector, m::Integer) =
   repeat(x, m), ȳ -> (dropdims(sum(reshape(ȳ, length(x), :); dims=2); dims=2), nothing)

@adjoint function repeat(x::AbstractVecOrMat, m::Integer, n::Integer=1)
   return repeat(x, m, n), function (ȳ)
      ȳ′ = reshape(ȳ, size(x,1), m, size(x,2), n)
      return reshape(sum(ȳ′; dims=(2,4)), size(x)), nothing, nothing
   end
end

"""
    tensor2vecofmats(Xs::DenseArray{<:Real,3}) -> Vector{<:DenseVecOrMat}

Given a 3D tensor `Xs` of dimensions D×T×B, constructs a vector of length T of D×B slices of `Xs` along the second dimension.
"""
tensor2vecofmats(Xs::DenseArray{<:Real,3}) = [Xs[:,t,:] for t ∈ axes(Xs, 2)]

"""
    vecofmats2tensor(xs::DenseVector{<:DenseVecOrMat}) -> DenseArray{<:Real,3}

Constructs a 3D tensor of dimensions D×T×B by concatenating along the second dimension the elements of the input vector `xs` of length T, whose each element is a D×B matrix.
"""
function vecofmats2tensor(xs::DenseVector{<:DenseVecOrMat})
   x₁ = first(xs)
   D, B = size(x₁)
   Xs = similar(x₁, D, length(xs), B)
   @inbounds for t ∈ eachindex(xs)
      Xs[:,t,:] = xs[t]
   end
   return Xs
end

function pad(xs::DenseVector{<:DenseVector}, multiplicity::Integer)
   T = length(xs)
   newT = ceil(Int, T / multiplicity)multiplicity
   z = zero(first(xs))
   return [xs; (_ -> z).(1:(newT - T))]
end

"""
    batch_inputs!(Xs::AbstractVector{<:AbstractVector{<:DenseVector}}, multiplicity::Integer, maxT::Integer = maximum(length, Xs))::DenseArray{<:Real,3}

Given a B-length collection `Xs`, whose each element is a sequence vector of D-dimensional input features, pads sequences to ensure every sequence has the same length `T` that is a multiple of `multiplicity`. Then it arranges these padded sequences into a single tensor of size D×T×B.
"""
function batch_inputs!(Xs::AbstractVector{<:AbstractVector{<:DenseVector}}, multiplicity::Integer, maxT::Integer = maximum(length, Xs))::DenseArray{<:Real,3}
   # find the smallest multiple of `multiplicity` that is no less than `maxT`
   newT = ceil(Int, maxT / multiplicity)multiplicity
   # construct padding element vector
   z = (zero ∘ first ∘ first)(Xs)
   # resize each sequence `xs` to the size `newT` by paddding it with vector z of zeros
   for xs ∈ Xs
      T = length(xs)
      resize!(xs, newT)
      xs[(T+1):end] .= Ref(z)
   end
   # first concatenate D-dimensional input features of each padded sequence of length newT into the single vector of length D*newT, then concatenate the resulting vectors along the 2nd dimension to get the D*newT×B matrix and then finally reshape the resulting matrix into D×newT×B tensor
      X = reshape(reduce(hcat, reduce.(vcat, Xs)), length(z), newT, :)
      # check: X == cat(reduce.(hcat, Xs)...; dims=3)
   return X
end

"""
    batch_targets(ys::AbstractVector{V}, output_dim::Integer, maxT::Integer = maximum(length, ys))::V where V <: DenseVector{<:Integer}

Given a batch vector of target sequences `ys` returns a vector of corresponding linear indexes into the prediction `Ŷs`, which is assumed to be a tensor od dimensions D×B×T. Here D denotes the dimensionality of the output, B is the batch size and T is the maximum time length in the batch.
"""
function batch_targets(ys::AbstractVector{V}, dim_out::Integer, maxT::Integer = maximum(length, ys))::V where V <: DenseVector{<:Integer}
   batch_size = length(ys)
   cartesian_indices = Vector{Vector{CartesianIndex{3}}}(undef, maxT)
   cartesian_indices_t = Vector{CartesianIndex{3}}(undef, batch_size)
   @views for time ∈ 1:maxT
      n = 0
      for (batch, yᵇ) ∈ enumerate(ys)
         if time <= length(yᵇ)
            n += 1
            cartesian_indices_t[n] = CartesianIndex(yᵇ[time], batch, time)
         end
      end
      cartesian_indices[time] = cartesian_indices_t[1:n]
   end
   cartesian_indices′ = reduce(vcat, cartesian_indices)
   linear_indices = LinearIndices((dim_out, batch_size, maxT))[cartesian_indices′]
   @assert issorted(linear_indices)
   return cartesian_indices′
end

"""
    batch_dataset(Xs::DenseVector{<:DenseVector{<:DenseVector}}, ys::DenseVector{<:DenseVector}, batch_size::Integer, multiplicity::Integer)

Arranges dataset into batches such that the number of batches approximately equals the ratio of dataset size to `batch_size`.
Batches are formed by first sorting sequences in the dataset according to their length (which minimizes the total number of elements to pad in inputs) and then partitioning the result into batches such that each batch approximately the same total number of sequence elements (this ensures that each batch takes up the same amount of memory, so as to avoid memory overflow when loading data into the GPU).
"""
function batch_dataset(Xs::DenseVector{<:DenseVector{<:DenseVector}},
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
   X_batches = [ batch_inputs!(Xs[firstidx:lastidx], multiplicity, maxT) for (firstidx, lastidx, maxT) ∈ zip(firstidxs, lastidxs, maxTs) ]
   indices_batches = [ batch_targets(ys[firstidx:lastidx], output_dim, maxT) for (firstidx, lastidx, maxT) ∈ zip(firstidxs, lastidxs, maxTs) ]

   batches = [(X |> gpu, indices, maxT) for (X, indices, maxT) ∈ zip(X_batches, indices_batches, maxTs)]
   return batches
end

"""
    mean_prob_of_correct_prediction(l::Real, total_length::Integer)::Real
    mean_prob_of_correct_prediction(l::Real, indices::DenseVector)::Real
    mean_prob_of_correct_prediction(l::Real, dataset::Vector{<:Tuple{DenseArray{<:Real,3}, DenseVector{<:Integer}, Integer}})::Real

Given a loss `l` for either a batch of length `total_length` or a batch with linear indices `indices` of correct labels or a collection of batches, `dataset`, returns mean probability of the correct prediction
"""
mean_prob_of_correct_prediction(l::Real, total_length::Integer)::Real = exp(-l / total_length)
mean_prob_of_correct_prediction(l::Real, indices::DenseVector)::Real = exp(-l / length(indices))
mean_prob_of_correct_prediction(l::Real, dataset::Vector{<:Tuple{DenseArray{<:Real,3}, DenseVector{<:Integer}, Integer}})::Real =
   exp(-l / sum(((_, indices, _),) -> length(indices), dataset))

function printlog(io::IO, message...)
   println(io, message...)
   flush(io)
   println(message...)
end
