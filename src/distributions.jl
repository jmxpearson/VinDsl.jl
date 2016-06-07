dists = [
    "constraints",
    "expfam",
    "HMM",
    "matrixnormal"
]

for dname in dists
    include(joinpath("distributions", "$(dname).jl"))
end
