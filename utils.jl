@adjoint function reduce(::typeof(hcat), As::AbstractVector{<:AbstractVecOrMat})
    cumsizes = cumsum(size.(As, 2))
    return reduce(hcat, As), Δ -> (nothing, map((sz, A) -> Zygote.pull_block_horz(sz, Δ, A), cumsizes, As))
end

@adjoint function reduce(::typeof(vcat), As::AbstractVector{<:AbstractVecOrMat})
    cumsizes = cumsum(size.(As, 1))
    return reduce(vcat, As), Δ -> (nothing, map((sz, A) -> Zygote.pull_block_vert(sz, Δ, A), cumsizes, As))
end

@adjoint function repeat(x::AbstractVecOrMat, m::Integer, n::Integer=1)
   size₁, size₂ = size(x, 1), size(x, 2)
   begin₁, begin₂ = firstindex(x, 1), firstindex(x, 2)
   end₁,   end₂   =  lastindex(x, 1),  lastindex(x, 2)
   return repeat(x, m, n), ȳ -> (sum(@view ȳ[(begin₁ + i*size₁):(end₁ + i*size₁), (begin₂ + j*size₂):(end₂ + j*size₂)] for i ∈ 0:(m-1), j ∈ 0:(n-1)), nothing, nothing)
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