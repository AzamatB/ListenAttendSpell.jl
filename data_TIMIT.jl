using WAV
using MFCC
using Flux: onehotbatch
using JLD2
using Base.Iterators
using IterTools

# make dictionary to map from phones to class numbers
const PHONEMES, PHN2IDX = let
   phonemes = split("h# q eh dx iy r ey ix tcl sh ow z s hh aw m t er l w aa hv ae dcl y axr d kcl k ux ng gcl g ao epi ih p ay v n f jh ax en oy dh pcl ah bcl el zh uw pau b uh th ax-h em ch nx eng")
   phonemes = convert(Vector{String}, phonemes)
   phn2idx = Dict(phoneme => i for (i, phoneme) ∈ enumerate(phonemes))
   phn2idx["sil"] = phn2idx["h#"]
   phonemes, phn2idx
end

# make dictionary to perform class folding
const FOLDINGS = Dict(
   "ao" => "aa",
   "ax" => "ah",
   "ax-h" => "ah",
   "axr" => "er",
   "hv" => "hh",
   "ix" => "ih",
   "el" => "l",
   "em" => "m",
   "en" => "n",
   "nx" => "n",
   "eng" => "ng",
   "zh" => "sh",
   "pcl" => "sil",
   "tcl" => "sil",
   "kcl" => "sil",
   "bcl" => "sil",
   "dcl" => "sil",
   "gcl" => "sil",
   "h#" => "sil",
   "pau" => "sil",
   "epi" => "sil",
   "ux" => "uw"
)

const FRAME_LENGTH = 0.025 # ms
const FRAME_INTERVAL = 0.010 # ms

"""
   build_features(wavfile::AbstractString, phnfile::AbstractString)

Extracts Mel filterbanks and associated labels from `wavfile` and `phnfile`.
"""
function build_features(wavfile::AbstractString, phnfile::AbstractString, Δorder::Integer=2)
   samples, sampling_frequency = wavread(wavfile)
   samples = vec(samples)
   mfccs, _, _ = mfcc(samples, sampling_frequency, :rasta; wintime=FRAME_LENGTH, steptime=FRAME_INTERVAL)

   lines = readlines(phnfile)
   labels = similar(lines)
   boundaries = similar(lines, Int)
   for (i, line) ∈ enumerate(lines)
      _, boundary, labels[i] = split(line)
      boundaries[i] = parse(Int, boundary)
   end

   sampl_length = FRAME_LENGTH * sampling_frequency
   sampl_interval = FRAME_INTERVAL * sampling_frequency
   half_frame_length = FRAME_LENGTH / 2

   # begin generating sequence labels by looping through the MFCC frames
   nsegments = length(labels)
   label_idx = 1
   label_sequence = similar(labels, size(mfccs,1))
   label, boundary = labels[label_idx], boundaries[label_idx]
   for i ∈ axes(mfccs, 1)
      win_end = sampl_length + (i-1)sampl_interval
      # move on to next label if current frame of samples is more than half
      # way into next labeled section and there are still more labels to iterate through
      if (label_idx < nsegments) && (win_end - boundary > half_frame_length)
         label_idx += 1
         label, boundary = labels[label_idx], boundaries[label_idx]
      end
      label_sequence[i] = label
   end
   # get rid of the frames that were labeld as "q"
   idcs_to_keep = filter(i -> label_sequence[i] != "q", eachindex(label_sequence))
   label_sequence = label_sequence[idcs_to_keep]
   mfccs = mfccs[idcs_to_keep,:]
   # compute filterbank derivates, fitted over 2 consecutive frames
   Δmfccs = (Δmfcc for Δmfcc ∈ take(iterated(x -> deltas(x, 2), mfccs), Δorder))
   features = mapreduce(x -> x', vcat, (mfccs, Δmfccs...))
   # convert from double precision to single precision for Flux
   features = convert(Matrix{Float32}, features)
   # generate class numbers, there are 61 total classes, but only 39 are used after folding
   target = [PHN2IDX[label] for label ∈ label_sequence]
   return (features, target)
end

"""
   build_dataset(dir_data::AbstractString, path_out::AbstractString)

Extracts data from files in `dir_data`, builds features, concatenates produced outputs and saves results as `path_out`.
"""
function build_dataset(dir_data::AbstractString, path_out::AbstractString; Δorder::Integer=2)
   Xys = mapreduce(vcat, walkdir(dir_data)) do (root, _, files)
      print("$(root)\r")
      phnfiles = (file for file ∈ files if !startswith(file, "SA") && endswith(file, ".PHN"))
      wavfiles = (file for file ∈ files if !startswith(file, "SA") && endswith(file, ".wav"))

      map(wavfiles, phnfiles) do wavfile, phnfile
         wavfilepath = joinpath(root, wavfile)
         phnfilepath = joinpath(root, phnfile)
         build_features(wavfilepath, phnfilepath, Δorder)
      end
   end
   # Xs, ys = fieldarrays(StructArray(Xys))
   Xs = [X for (X, _) ∈ Xys]
   ys = [y for (_, y) ∈ Xys]

   ispath(path_out) || mkpath(dirname(path_out))
   JLD2.@save path_out Xs ys PHONEMES
   return Xs, ys
end


const DIR_DATA_TEST  = "/Users/Azamat/Projects/LAS/data/TIMIT/TIMIT_wav/TEST"
const PATH_OUT_TEST   = "/Users/Azamat/Projects/LAS/data/TIMIT/TIMIT_MFCC/data_test.jld"

const DIR_DATA_TRAIN = "/Users/Azamat/Projects/LAS/data/TIMIT/TIMIT_wav/TRAIN"
const PATH_OUT_TRAIN  = "/Users/Azamat/Projects/LAS/data/TIMIT/TIMIT_MFCC/data_train.jld"

@time build_dataset(DIR_DATA_TEST, PATH_OUT_TEST)
@time build_dataset(DIR_DATA_TRAIN, PATH_OUT_TRAIN)
