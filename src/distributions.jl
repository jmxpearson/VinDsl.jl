dists = [
    "constraints",
    "expfam",
    "HMM",
    "matrixnormal",
    "LKJ"
]

for dname in dists
    include(joinpath("distributions", "$(dname).jl"))
end
