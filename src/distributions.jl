dists = [
    "constraints",
    "expfam",
    "HMM",
    "matrixnormal",
    "lkjdistribution"
]

for dname in dists
    include(joinpath("distributions", "$(dname).jl"))
end
