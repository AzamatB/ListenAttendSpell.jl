include("LAS.jl")

using .ListenAttendSpell

ListenAttendSpell.main(n_epochs=300, saved_results=false, log_weights = false)
