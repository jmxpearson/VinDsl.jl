dists = [
    "expfam",
    "HMM"
]

for dname in dists
    include(joinpath("distributions", "$(dname).jl"))
end
