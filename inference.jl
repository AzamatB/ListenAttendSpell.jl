using LinearAlgebra
using StaticArrays

# node of a tree structured stack
mutable struct TSS{V <: AbstractVector{Int}}
	id::Int
	length::Int
	parent_id::Int
	parent_idx::Int # index of the parent at which stack joins to it
	chars::V
end

function TSS(id::Integer, parent_id::Integer, parent_idx::Integer, firstchars::Union{AbstractVector{T}, NTuple{N,T} where N} where {T<:Integer}, max_length::Integer)
	len = length(firstchars)
	chars = Vector{Int}(undef, max_length)
	chars[1:len] = collect(firstchars)
	return TSS(id, len, parent_id, parent_idx, chars)
end

TSS(id::Integer, firstchars::Union{AbstractVector{T}, NTuple{N,T} where N} where {T<:Integer}, max_length::Integer) =
TSS(id, 0, 0, firstchars, max_length)

function Base.push!(beam::TSS, char)
	beam.chars[beam.length += 1] = char
	return nothing
end

function Base.replace!(beam::TSS, parent_id::Integer, parent_idx::Integer, char::Int)
	beam.parent_id = parent_id
	beam.parent_idx = parent_idx
	beam.chars[1] = char
	beam.length = 1
	return nothing
end

@generated function getsequences(ps::StaticVector{N,<:StaticVector{D,T}}) where {T<:Real,D,N}
	head = quote
		scores = Vector{T}(undef, D^N)
		sequences = similar(scores, NTuple{N,Int})
		n = 1
	end

	indentation = "   "
	N_indentations = indentation^N
	strloop = string(
		"@inbounds ", # optional optimization
		ntuple(k -> indentation^(k-1) * "for (i$k, p$k) in enumerate(ps[$k])\n", N)...,
		N_indentations, "sequences[n] = (", ntuple(k -> "i$k, ", N)..., ")\n",
		N_indentations, "scores[n] = ", mapreduce(k -> "p$k", (ls, rs) -> ls * " * " * rs, 1:N), "\n",
		N_indentations, "n += 1\n",
		ntuple(k -> indentation^(N - k) * "end\n", N)...)

	# loop = Meta.quot(strloop)
	loop = Meta.parse(strloop)

	tail = :(return scores, sequences)
	expr = Expr(:block, head, loop, tail)
	return expr
end

function beam_search(ŷs::AbstractVector{<:AbstractVector}, width::Integer)::Vector{Vector{Int}}
	# alphabet length
	D = length(first(ŷs))
	# get integer initlen such that D^initlen >= beam width
	initlen = max(ceil(Int, log(D, width)), 1)
	# compute D^initlen number of candidate sequences and their corresponding scores
	scores, sequences = getsequences(SVector(ntuple(i -> Size(D)(ŷs[i]), initlen)))
	widthrng = 1:width
	# find the width number of top scoring candidate sequences
	topidxs = partialsortperm(scores, widthrng; rev=true)
	topscores = scores[topidxs]
	topsequences = sequences[topidxs]
	max_length = length(ŷs)
	# initialize beams with top candidate sequences
	beams = [ TSS(id, firstchars, max_length) for (id, (score, firstchars)) ∈ enumerate(zip(topscores, topsequences)) ]
	# preallocate arrays before main loop
	scores = Matrix{Float64}(undef, D, width)
	vecscores = vec(scores)
	idxs = collect(eachindex(vecscores))
	beamcounts = similar(widthrng)
	sortedbeamids = similar(beamcounts)
	@views ŷs = ŷs[(initlen+1):end]
	availablepairs = Vector{NTuple{2,Int}}(undef, width)

	for ŷ ∈ ŷs
		# compute the scores for all candidate sequences
		mul!(scores, ŷ, topscores')
		# pick the top scoring ones
		topidxs = partialsortperm!(idxs, vecscores, widthrng; rev=true)
		topscores = vecscores[topidxs]
		# each element is a (charidx, beamidx) tuple
		topsequences = CartesianIndices(scores)[topidxs]
		# how many times each beam is scored in the top
		fill!(beamcounts, 0)
		# flag indicating if the beam is unassigned
		unassigned = trues(width)
		i = 0
		for ci ∈ topsequences
			char, beamid = Tuple(ci)
			beamcounts[beamid] += 1
			if unassigned[beamid]
				# if the beam is not assigned, then assign it
				push!(beams[beamid], char)
				# and flag it as such
				unassigned[beamid] = false
			else # if assigned,
				# then add the pair to the list of available pairs
				availablepairs[i += 1] = (beamid, char)
			end
		end
		@views unassignedbeams = beams[unassigned]
		for (beam, (beamid, char)) ∈ zip(unassignedbeams, availablepairs)
			parentbeam = beams[beamid]
			parent_id = parentbeam.id
			parent_idx = parentbeam.length - 1
			replace!(beam, parent_id, parent_idx, char)
		end
	end

	predictions = Vector{Vector{Int}}(undef, width)
	for (i, beam) ∈ enumerate(beams)
		prediction = Vector{Int}(undef, max_length)
		Δ = max_length
		lastidx = beam.length
		while true
			rng = 1:lastidx
			Δ -= lastidx
			prediction[rng .+ Δ] = @view beam.chars[rng]
			parentbeam_id = beam.parent_id
			(parentbeam_id == 0) && break
			lastidx = beam.parent_idx
			beam = beams[parentbeam_id]
		end
		predictions[i] = prediction
	end

	return predictions
end


ŷs = [[0.1, 0.2, 0.3, 0.4, 0.5],
      [0.5, 0.4, 0.3, 0.2, 0.1],
      [0.1, 0.2, 0.3, 0.4, 0.5],
      [0.5, 0.4, 0.3, 0.2, 0.1],
      [0.1, 0.2, 0.3, 0.4, 0.5],
      [0.5, 0.4, 0.3, 0.2, 0.1],
      [0.1, 0.2, 0.3, 0.4, 0.5],
      [0.5, 0.4, 0.3, 0.2, 0.1],
      [0.1, 0.2, 0.3, 0.4, 0.5],
      [0.5, 0.4, 0.3, 0.2, 0.1]]
width = 3

predictions = beam_search(ŷs, width)

wan = ones(Int, length(first(predictions)))

paths = hcat((predictions .- (wan,))...)


truepaths = hcat([4, 0, 4, 0, 4, 0, 4, 0, 4, 0], [4, 0, 4, 0, 4, 0, 4, 0, 4, 1], [4, 0, 4, 0, 4, 0, 4, 0, 3, 0])
