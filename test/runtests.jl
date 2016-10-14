push!(LOAD_PATH, "$(homedir())/code/VinDsl.jl/src")  # needed if VB not a full module

using PDMats
import Base: ==
using FactCheck
using Distributions
using VinDsl
using Combinatorics
using StatsFuns

###################################################
# Utility functions for testing
###################################################
# use these equality definitions for testing purposes
==(x::PDMat, y::PDMat) = x.mat == y.mat
function =={D <: Distribution}(x::D, y::D)
    all(f -> getfield(x, f) == getfield(y, f), fieldnames(x))
end
function =={D <: Distribution}(x::RandomNode{D}, y::RandomNode{D})
    all(f -> getfield(x, f) == getfield(y, f), fieldnames(x))
end

###################################################
# Test files to run
###################################################
tests = [
    "typetests", # overwritten warning
    ## "constrainttests",
    ## "utiltests",
    ## "structuretests",
    ## "expressiontests",
    ## "hmmtests", # not summed to 1 warning
    #### "integrationtests",
    ## "distributiontests",
    ## "advitests",
]

print_with_color(:blue, "Running tests:\n")

srand(12345)

for t in tests
    test_fn = "$t.jl"
    print_with_color(:green, "* $test_fn\n")
    include(test_fn)
end
FactCheck.exitstatus()
