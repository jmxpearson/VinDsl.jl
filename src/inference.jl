# include various methods for updating nodes
strategies = [
    "conjugacy",
    "opt"
]

for sname in strategies
    include(joinpath("inference", "$(sname).jl"))
end
