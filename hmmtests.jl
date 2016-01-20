push!(LOAD_PATH, ".")  # needed if VB not a full module
using FactCheck
using Distributions

include("HMM.jl")
include("expfam.jl")

srand(12345)

facts("Checking HMM distribution") do
    K = 4
    T = 50
    pars = Dirichlet(K, 1)
    A = Array{Float64}(K, K)
    π0 = rand(pars)
    for k in 1:K
        A[:, k] = rand(pars)
    end
    ψ = rand(K, T)

    x = HMM(ψ, π0, A)

    context("Constructors") do
        @fact isa(x, HMM) --> true
        @fact isa(x, DiscreteMatrixDistribution) --> true
        @fact isa(x, Distribution) --> true
        @fact x.ψ --> ψ
        @fact x.A --> A
        @fact x.π0 --> π0
        @fact size(x.ξ) --> size(ψ)
        @fact size(x.Ξ) --> (K, K, T - 1)
    end

    context("Bad constructor inputs throw errors") do
        @fact_throws ErrorException HMM(ψ[2:end, :], π0, A)
        @fact_throws ErrorException HMM(ψ, π0[2:end], A)
        @fact_throws ErrorException HMM(ψ, π0, A[2:end, :])
        @fact_throws ErrorException HMM(ψ, π0, A[:, 2:end])
    end

    context("Check basic interface") do
        @fact nstates(x) --> K
        @fact size(x) --> (K, T)
        @fact length(x) --> K * T
    end

    context("Check mean") do
        ξ = mean(x)
        @fact ξ --> x.ξ
    end

    context("Check two-slice") do
        Ξ = cov(x)
        @fact Ξ --> x.Ξ
    end

    context("Check sampling") do
        z = rand(x)
        @fact size(z) --> (K, T)
        @fact length(find(z)) --> T
        @fact sum(z .!= 0, 1) --> ones(1, T)
    end

    context("Check logpdf") do
        # draw a chain of states
        z = zeros(K, T)
        init_state = rand(Categorical(π0))
        z[init_state, 1] = 1
        for t in 2:T
            newstate = rand(Categorical(A * z[:, t - 1]))
            z[newstate, t] = 1
        end
        lpdf = logpdf(x, z)

        @fact lpdf --> less_than_or_equal(0)
    end

    context("Check entropy") do
        @fact entropy(x) --> greater_than_or_equal(0)
    end
end

facts("Checking MarkovMatrix distribution") do
    d = 5

    pars = [Dirichlet(d, 1) for i in 1:d]
    parmat = hcat([rand(Dirichlet(d, 1)) for i in 1:d]...)
    parvec = [rand(Dirichlet(d, 1)) for i in 1:d]

    x = MarkovMatrix(pars)

    context("Inner constructor") do
        @fact isa(x, MarkovMatrix) --> true
        @fact isa(x, ContinuousMatrixDistribution) --> true
        @fact isa(x, Distribution) --> true
        @fact x.cols --> pars
    end

    context("Inner constructor consistency checks") do
        @fact_throws ErrorException MarkovMatrix(pars[2:end, :])
    end

    context("Outer constructor") do
        x = MarkovMatrix(parmat)
        @fact isa(x, MarkovMatrix) --> true

        y = MarkovMatrix(parvec)
        @fact isa(y, MarkovMatrix) --> true
    end

    context("Outer constructor consistency checks") do
        @fact_throws ErrorException MarkovMatrix(parmat[2:end, :])
    end

    context("Check basic interfact") do
        @fact nstates(x) --> d
        @fact size(x) --> (d, d)
        @fact length(x) --> d^2
    end

    context("Check mean") do
        m = mean(x)
        @fact size(m) --> (d, d)
        @fact sum(m, 1) --> roughly(ones(1, d))
    end

    context("Check Elog") do
        El = Elog(x)
        @fact size(El) --> (d, d)
    end

    context("Check sampling") do
        A = rand(x)
        @fact size(A) --> (d, d)
        @fact sum(A, 1) --> roughly(ones(1, d))
    end

    context("Check logpdf") do
        # draw a matrix
        A = rand(x)
        lpdf = logpdf(x, A)

        ll = sum([logpdf(x.cols[i], A[:, i]) for i in 1:nstates(x)])
        @fact lpdf --> roughly(ll)
    end

    context("Check entropy") do
        H = sum([entropy(c) for c in x.cols])
        @fact entropy(x) --> roughly(H)
    end
end

facts("Checking forward-backward algorithm.") do
    # set up some contants for test dataset
    T = 500  # time points
    U = 50  # number of observations at each time
    K = 5  # states
    dt = 1  # time step

    # make transition probabilities
    # we assume that the transition matrix A is left stochastic
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
    λ = 10 * collect(1:K)
    N = Array{Int}(U, T)
    for t in 1:T
        N[:, t] = rand(Poisson(λ[z_ints[t]]), U)
    end

    # logpdf of observations, conditioned on state
    logψ = Array{Float64}(K, T)
    for t in 1:T
        for k in 1:K
            logψ[k, t] = 0
            for u in 1:U
                logψ[k, t] += logpdf(Poisson(λ[k]), N[u, t])
            end
        end
    end

    # run forward-backward algorithm
    γ, logZ, Ξ = forwardbackward(π0, A, map(exp, logψ))

    context("Checking validity of test data") do
        @fact sum(π0) --> roughly(1)
        @fact sum(A, 1) --> roughly(ones(1, K))
        @fact length(find(z)) --> T
        @fact sum(z .!= 0, 1) --> ones(1, T)
        @fact logψ .≤ 0 --> all
    end

    context("Check return shapes.") do
        @fact size(γ) --> (K, T)
        @fact sum(γ, 1) --> roughly(ones(1, T))
        @fact size(logZ) --> ()
        @fact size(Ξ) --> (K, K, T - 1)
        @fact sum(Ξ, [1, 2]) --> roughly(ones(1, 1, T - 1))
    end

    context("Rescaling logψ compensated by logZ") do
        offset = randn(1, T)
        logψ_r = logψ .+ offset
        γ_r, logZ_r, Ξ_r = forwardbackward(π0, A, map(exp, logψ_r))

        @fact γ --> roughly(γ_r)
        @fact logZ_r --> roughly(logZ + sum(offset))
        @fact Ξ_r --> roughly(Ξ)
    end

    context("Rescaling π0 compensated by logZ") do
        offset = maximum(log(π0))
        π0_r = exp(log(π0) + offset)
        γ_r, logZ_r, Ξ_r = forwardbackward(π0_r, A, map(exp, logψ))

        @fact γ --> roughly(γ_r)
        @fact logZ_r --> roughly(logZ + offset)
        @fact Ξ_r --> roughly(Ξ)
    end

    context("Rescaling A compensated by logZ") do
        offset = maximum(log(A))
        A_r = exp(log(A) + offset)
        γ_r, logZ_r, Ξ_r = forwardbackward(π0, A_r, map(exp, logψ))

        @fact γ --> roughly(γ_r)
        @fact logZ_r --> roughly(logZ + (T - 1) * offset)
        @fact Ξ_r --> roughly(Ξ)
    end

    context("Test entropy positive") do
        # H = E_q[-log q] logZ - LL ≥ 0
        emission_piece = sum(γ .* logψ)
        initial_piece = sum(γ[:, 1] .* log(π0))
        transition_piece = 0.
        for t in 1:(T - 1)
            transition_piece += sum(Ξ[:, :, t] .* log(A))
        end

        LL = emission_piece + initial_piece + transition_piece

        @fact (logZ ≥ LL || logZ ≈ LL) --> true
    end

    context("Test entropy positive (subadditive parameters)") do
        # same test for positive entropy, but with subadditive A and π0
        logπ0_r = log(π0) - rand(K)
        logA_r = log(A) - rand(size(A))

        # run forward-backward algorithm
        γ, logZ, Ξ = forwardbackward(exp(logπ0_r), exp(logA_r), map(exp, logψ))

        # calculate log likelihood
        emission_piece = sum(γ .* logψ)
        initial_piece = sum(γ[:, 1] .* logπ0_r)
        transition_piece = 0.
        for t in 1:(T - 1)
            transition_piece += sum(Ξ[:, :, t] .* logA_r)
        end

        LL = emission_piece + initial_piece + transition_piece

        @fact (logZ ≥ LL || logZ ≈ LL) --> true
    end

    context("Test accuracy of inference") do
        tmin = 10
        @fact γ[:, tmin:end] --> roughly(z[:, tmin:end], atol=1e-3)
    end
end
