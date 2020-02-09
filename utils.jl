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
