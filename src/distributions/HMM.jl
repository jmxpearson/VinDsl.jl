using Distributions
import Base.size, Base.length, Base.rand, Base.mean, Base.convert, Distributions.logpdf, Distributions.entropy, Distributions.params

################### Hidden Markov Model distribution #####################
immutable HMM{S <: Real} <: DiscreteMatrixDistribution
    # convention: Matrices are state x time
    ψ::Matrix{S}  # matrix of conditional probabilites of symbol emission
    π0::Vector{S}  # vector of initial state probabilities
    A::Matrix{S}  # transition matrix (columns sum to 1)
    ξ::Matrix{S}  # mean of distribution
    Ξ::Array{S, 3}  # two-slice marginal of distribution (state x state x time)
    logZ::S  # log normalization constant for distribution

    function HMM(ψ, π0, A)
        size(ψ, 1) == length(π0) == size(A, 1) == size(A, 2) || error("A and π0 have incompatible shapes")

        M, T = size(ψ)

        sum(π0) ≈ 1 || warn("Entries of π0 do not sum to 1.")

        sum(A, 1) ≈ ones(1, M) || warn("Columns of A do not sum to 1.")

        ξ, logZ, Ξ = forwardbackward(π0, A, ψ)

        new(ψ, π0, A, ξ, Ξ)
    end
end

HMM{S <: Real}(ψ::Matrix{S}, π0::Vector{S}, A::Matrix{S}) = HMM{S}(ψ, π0, A)

#### Conversions
convert{S <: Real, V <: Real}(::Type{HMM{S}}, ψ::Matrix{V}, π0::Vector{V}, A::Matrix{V}) = HMM(convert(Matrix{S}, ψ), convert(Vector{S}, π0), convert(Matrix{S}, A))
convert{S <: Real, V <: Real}(::Type{HMM{S}}, d::HMM{V}) = HMM(convert(Matrix{S}, d.ψ), convert(Vector{S}, d.π0), convert(Matrix{S}, d.A))

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

function logpdf{S <: Real}(d::HMM{S}, x::Matrix{S})
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

function naturals_to_params{D <: HMM}(η, ::Type{D})
    map(exp, η)
end

################### Markov Chain distribution #####################
immutable MarkovChain{S <: Real} <: DiscreteMatrixDistribution
    # convention: Matrices are state x time
    π0::Vector{S}  # vector of initial state probabilities
    A::Matrix{S}  # transition matrix (columns sum to 1)
    T::Integer

    function MarkovChain(π0, A, T)
        isa(T, Int) || error("Supplied chain length is not an integer.")
        length(π0) == size(A, 1) == size(A, 2) || error("A and π0 have incompatible shapes")

        sum(π0) ≈ 1 || warn("Entries of π0 do not sum to 1.")

        sum(A, 1) ≈ ones(1, size(A, 1)) || warn("Columns of A do not sum to 1.")

        new(π0, A, T)
    end
end

MarkovChain{S <: Real}(π0::Vector{S}, A::Matrix{S}, T) = MarkovChain{S}(π0, A, T)
function MarkovChain{S <: Real, V <: Real}(π0::Vector{S}, A::Matrix{V}, T)
    W = promote_type(eltype(π0), eltype(A))
    MarkovChain(convert(Vector{W}, π0), convert(Matrix{W}, A), T)
end

#### Conversions
convert{S <: Real, V <: Real}(::Type{MarkovChain{S}}, π0::Vector{V}, A::Matrix{V}, T) = MarkovChain(convert(Vector{S}, π0), convert(Matrix{S}, A), T)
convert{S <: Real, V <: Real}(::Type{MarkovChain{S}}, d::MarkovChain{V}) = MarkovChain(convert(Vector{S}, d.π0), convert(Matrix{S}, d.A), d.T)

nstates(d::MarkovChain) = length(d.π0)
size(d::MarkovChain) = (nstates(d), d.T)
length(d::MarkovChain) = prod(size(d))

function mean(d::MarkovChain)
    M, T = size(d)
    m = zeros(M, T)
    m[:, 1] = d.π0
    for i in 2:T
        m[:, i] = d.A * m[:, i - 1]
    end
    m
end

params(d::MarkovChain) = (d.π0, d.A)

function rand(d::MarkovChain)
    M, T = size(d)
    z = zeros(M, T)
    init_state = rand(Categorical(d.π0))
    z[init_state, 1] = 1
    for t in 2:T
        pvec = d.A * z[:, t - 1]
        pvec /= sum(pvec)  # normalize, since Ξ is joint, not conditional
        newstate = rand(Categorical(pvec))
        z[newstate, t] = 1
    end
    z
end

function logpdf{S <: Real}(d::MarkovChain{S}, x::Matrix{S})
    size(x) == size(d) || error("Input matrix x has wrong size.")

    T = size(x, 2)

    initial_piece = sum(x[:, 1] .* log(d.π0))
    transition_piece = 0.
    for t in 1:(T - 1)
        transition_piece += (x[:, t + 1]' * log(d.A) * x[:, t])[1]
    end

    initial_piece + transition_piece
end

function entropy(d::MarkovChain)
    _, T = size(d)

    initial_piece = sum(d.π0 .* log(d.π0))
    transition_piece = 0.
    for t in 1:(T - 1)
        transition_piece += sum(d.A .* log(d.A))
    end

    -(initial_piece + transition_piece)
end

function naturals(d::MarkovChain)
    (log(d.π0), log(d.A))
end

function naturals_to_params{D <: MarkovChain}(η, ::Type{D})
    map(exp, η)
end
################### Markov Matrix distribution #####################
immutable MarkovMatrix{T <: Real} <: ContinuousMatrixDistribution
    cols::Vector{Dirichlet{T}}  # each column is a Dirichlet distribution

    function MarkovMatrix(cols)
        length(cols) == length(cols[1]) || error("Input is not a square matrix.")

        new(cols)
    end
end

MarkovMatrix{T <: Real}(cols::Vector{Dirichlet{T}}) = MarkovMatrix{T}(cols)

function MarkovMatrix{T <: Real}(A::Matrix{T})
    r, c = size(A)
    r == c || error("Input matrix must be square.")

    MarkovMatrix([Dirichlet(A[:, i]) for i in 1:c])
end

function MarkovMatrix{T <: Real}(cols::Vector{Vector{T}})
    p = length(cols)
    all(x -> length(x) == p, cols) || error("Probability vector lengths do not match.")

    MarkovMatrix([Dirichlet(c) for c in cols])
end

#### Conversions
convert{T <: Real, S <: Real}(::Type{MarkovMatrix{T}}, cols::Vector{Dirichlet{S}}) = MarkovMatrix([convert(Dirichlet{T}, c) for c in cols])
convert{T <: Real, S <: Real}(::Type{MarkovMatrix{T}}, A::Matrix{S}) = MarkovMatrix(convert(Matrix{T}, A))
convert{T <: Real, S <: Real}(::Type{MarkovMatrix{T}}, cols::Vector{Vector{S}}) = MarkovMatrix([convert(Vector{T}, c) for c in cols])
convert{T <: Real, S <: Real}(::Type{MarkovMatrix{T}}, d::MarkovMatrix{S}) = MarkovMatrix([convert(Dirichlet{T}, c) for c in d.cols])

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

function logpdf{T <: Real}(d::MarkovMatrix, x::Matrix{T})
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

function naturals_to_params{D <: MarkovMatrix}(η, ::Type{D})
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
