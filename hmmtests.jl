push!(LOAD_PATH, ".")  # needed if VB not a full module
using FactCheck
using Distributions

include("HMM.jl")

srand(12345)

# set up some contants for test dataset
T = 500  # time points
K = 5  # states
dt = 1  # time step

# make transition probabilities
# we assume that the transition matrix A is right stochastic
pars = Dirichlet(K, 1)
A = Array{Float64}(K, K)
π0 = rand(pars)
for k in 1:K
    A[:, k] = rand(pars)
end

# draw a chain of states
z = zeros(K, T)
z_ints = Array{Int}(T)
init_state = rand(Categorical(π0))
z[init_state, 1] = 1
z_ints[1] = init_state
for t in 2:T
    newstate = rand(Categorical(A * z[:, t - 1]))
    z[newstate, t] = 1
    z_ints[t] = newstate
end

# observation model: Poisson
λ = 3 * collect(1:K)
N = Array{Int}(T)
for t in 1:T
    N[t] = rand(Poisson(λ[z_ints[t]]))
end

# logpdf of observations, conditioned on state
ψ = Array{Float64}(K, T)
for t in 1:T
    for k in 1:K
        ψ[k, t] = logpdf(Poisson(λ[k]), N[t])
    end
end

facts("Checking validity of test data.") do
    @fact sum(π0) --> roughly(1)
    @fact sum(A, 1) --> roughly(ones(1, K))
    @fact length(find(z)) --> T
    @fact sum(z .!= 0, 1) --> ones(1, T)
    @fact ψ .≤ 0 --> all

end
