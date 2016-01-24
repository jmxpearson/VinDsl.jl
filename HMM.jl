using Distributions
import Base.size, Base.length, Base.rand, Base.mean, Distributions.logpdf, Distributions.entropy, Distributions.params

################### Hidden Markov Model distribution #####################
immutable HMM{N <: Number} <: DiscreteMatrixDistribution
    # convention: Matrices are state x time
    ψ::Matrix{N}  # matrix of conditional probabilites of symbol emission
    π0::Vector{N}  # vector of initial state probabilities
    A::Matrix{N}  # transition matrix (columns sum to 1)
    ξ::Matrix{N}  # mean of distribution
    Ξ::Array{N, 3}  # two-slice marginal of distribution (state x state x time)
    logZ::N  # log normalization constant for distribution

    function HMM(ψ, π0, A)
        size(ψ, 1) == length(π0) == size(A, 1) == size(A, 2) || error("A and π0 have incompatible shapes")

        M, T = size(ψ)

        sum(π0) ≈ 1 || error("Entries of π0 do not sum to 1.")

        sum(A, 1) ≈ ones(1, M) || error("Columns of A do not sum to 1.")

        ξ, logZ, Ξ = forwardbackward(π0, A, ψ)

        new(ψ, π0, A, ξ, Ξ)
    end
end

HMM{N <: Number}(ψ::Matrix{N}, π0::Vector{N}, A::Matrix{N}) = HMM{N}(ψ, π0, A)

nstates(d::HMM) = length(d.π0)
size(d::HMM) = size(d.ψ)
length(d::HMM) = prod(size(d))

mean(d::HMM) = d.ξ
cov(d::HMM) = d.Ξ
params(d::HMM) = (d.ψ, d.π0, d.A)

function rand(d::HMM)
    M, T = size(d)
    z = zeros(M, T)
    init_state = rand(Categorical(d.ξ[:, 1]))
    z[init_state, 1] = 1
    for t in 2:T
        pvec = d.Ξ[:, findfirst(z[:, t - 1]), t - 1]
        pvec /= sum(pvec)  # normalize, since Ξ is joint, not conditional
        newstate = rand(Categorical(pvec))
        z[newstate, t] = 1
    end
    z
end

function logpdf{N <: Number}(d::HMM{N}, x::Matrix{N})
    size(x) == size(d) || error("Input matrix x has wrong size.")

    T = size(x, 2)

    emission_piece = sum(x .* log(d.ψ))
    initial_piece = sum(x[:, 1] .* log(d.π0))
    transition_piece = 0.
    for t in 1:(T - 1)
        transition_piece += (x[:, t + 1]' * log(d.A) * x[:, t])[1]
    end

    emission_piece + initial_piece + transition_piece - d.logZ
end

function entropy(d::HMM)
    _, T = size(d)

    emission_piece = sum(d.ξ .* log(d.ψ))
    initial_piece = sum(d.ξ[:, 1] .* log(d.π0))
    transition_piece = 0.
    for t in 1:(T - 1)
        transition_piece += sum(d.Ξ[:, :, t] .* log(d.A))
    end

    d.logZ -(emission_piece + initial_piece + transition_piece)
end

function naturals(d::HMM)
    (log(d.ψ), log(d.π0), log(d.A))
end

function naturals_to_params{N <: Number}(η, ::Type{HMM{N}})
    map(exp, η)
end

################### Markov Matrix distribution #####################
immutable MarkovMatrix <: ContinuousMatrixDistribution
    cols::Vector{Dirichlet}  # each column is a Dirichlet distribution

    function MarkovMatrix(cols)
        length(cols) == length(cols[1]) || error("Input is not a square matrix.")

        new(cols)
    end
end

function MarkovMatrix{N <: Number}(A::Matrix{N})
    r, c = size(A)
    r == c || error("Input matrix must be square.")

    MarkovMatrix([Dirichlet(A[:, i]) for i in 1:c])
end

function MarkovMatrix{N <: Number}(cols::Vector{Vector{N}})
    p = length(cols)
    all(x -> length(x) == p, cols) || error("Probability vector lengths do not match.")

    MarkovMatrix([Dirichlet(c) for c in cols])
end

nstates(d::MarkovMatrix) = length(d.cols)
size(d::MarkovMatrix) = (length(d.cols), length(d.cols))
length(d::MarkovMatrix) = length(d.cols)^2

function mean(d::MarkovMatrix)
    p = nstates(d)
    m = Array{Float64}(p, p)
    for i in 1:p
        m[:, i] = mean(d.cols[i])
    end
    m
end

function Elog(d::MarkovMatrix)
    p = nstates(d)
    m = Array{Float64}(p, p)
    for i in 1:p
        m[:, i] = Elog(d.cols[i])
    end
    m
end

function rand(d::MarkovMatrix)
    p = nstates(d)
    x = Array{Float64}(p, p)
    for i in 1:p
        x[:, i] = rand(d.cols[i])
    end
    x
end

function logpdf{N <: Number}(d::MarkovMatrix, x::Matrix{N})
    size(x) == size(d) || error("Input matrix x has wrong size.")
    s = 0.
    for i in 1:nstates(d)
        s += logpdf(d.cols[i], x[:, i])
    end
    s
end

function entropy(d::MarkovMatrix)
    s = 0.
    for c in d.cols
        s += entropy(c)
    end
    s
end

function naturals(d::MarkovMatrix)
    p = nstates(d)
    nats = Array{Float64}(p, p)
    for i in 1:p
        nats[:, i] = naturals(d.cols[i])[1]  # naturals returns a tuple
    end
    (nats,)
end

function params(d::MarkovMatrix)
    p = nstates(d)
    nats = Array{Float64}(p, p)
    for i in 1:p
        nats[:, i] = params(d.cols[i])[1]  # naturals returns a tuple
    end
    (nats,)
end

function naturals_to_params(η, ::Type{MarkovMatrix})
    nats = η[1]
    p, _ = size(nats)
    pars = similar(nats)
    for i in 1:p
        pars[:, i] = naturals_to_params(nats[:, i], Dirichlet)[1]  # n2par returns a tuple
    end
    (pars,)
end

"""
Implement the forward-backward inference algorithm.
A is a matrix of transition probabilities that acts to the right:
new_state = A * old_state, so that columns of A sum to 1
ψ is the vector of evidence: p(y_t|z_t); it does not need to be
normalized, but the lack of normalization will be reflected in logZ
such that the end result using the given ψ will be properly normalized
when using the returned value of logZ.
"""
function forwardbackward(π0, A, ψ)
    if any(π0 .> 1) throw(ArgumentError()) end
    if any(A .> 1) throw(ArgumentError()) end

    # get shape parameters
    M, T = size(ψ)

    # allocate empty variables
    α = Array{Float64}(M, T)
    β = Array{Float64}(M, T)
    γ = Array{Float64}(M, T)
    Ξ = Array{Float64}(M, M, T - 1)

    # initialize
    a = ψ[:, 1] .* π0
    α[:, 1] = a / sum(a)
    logZ = log(sum(a))
    β[:, T] = 1

    # forward pass
    for t in 2:T
        asum = 0.
        for i in 1:M
            a[i] = 0.
            for j in 1:M
                a[i] += ψ[i, t] * A[i, j] * α[j, t - 1]
            end
            asum += a[i]
        end

        for i in 1:M
            α[i, t] = a[i] / asum
        end

        logZ += log(asum)
    end

    # backward pass
    for t in T:-1:2
        asum = 0.
        for i in 1:M
            a[i] = 0.
            for j in 1:M
                a[i] += β[j, t] * ψ[j, t] * A[j, i]
            end
            asum += a[i]
        end

        for i in 1:M
            β[i, t - 1] = a[i] / asum
        end
    end

    # calculate posterior
    for t in 1:T
        γsum = 0.
        for i in 1:M
            γ[i, t] = α[i, t] * β[i, t]
            γsum += γ[i, t]
        end

        for i in 1:M
            γ[i, t] /= γsum
        end
    end

    # calculate two-slice marginals
    for t in 1:(T - 1)
        xsum = 0.
        for i in 1:M
            for j in 1:M
                Ξ[i, j, t] = β[i, t + 1] * ψ[i, t + 1]
                Ξ[i, j, t] *= α[j, t] * A[i, j]
                xsum += Ξ[i, j, t]
            end
        end

        #normalize
        for i in 1:M
            for j in 1:M
                Ξ[i, j, t] /= xsum
            end
        end
    end

    γ, logZ, Ξ

end
