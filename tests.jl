using BenchmarkTools
using Test
using Traceur
# tests for levenshtein distance
using Random
using REPL

function compare(seq₁::AbstractString, seq₂::AbstractString)::Bool
   got = levendist(seq₁, seq₂)
   correct = REPL.levenshtein(seq₁, seq₂)
   if got != correct
      @show got, correct
      return false
   end
   return true
end

@test compare("αjosittingse", "[jokittense")
@test compare("josittingse", "jokittense")
@test compare("kitten", "sitting")
@test compare("azamat", "")
@test compare("azamat", "aza")
@test compare("aza", "tamaza")
@test compare("azamat", "azabemat")
@test compare("azabemat", "c")
@test compare("rosettacode", "raisethysword")
@test compare("azabemat", "b")
@test compare("Saturday", "Sunday")
@test compare("kittenaaaaaaaaaaaaaaaaa", "kitten")

seqlenrng = 1:1000
all(compare(randstring.(rand.((seqlenrng, seqlenrng)))...) for _ ∈ seqlenrng)
