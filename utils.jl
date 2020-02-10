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
