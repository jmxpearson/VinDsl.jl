using Distributions
import Base.size, Base.length, Base.rand

immutable HMM{N <: Number} <: DiscreteMatrixDistribution
    ψ::Matrix{N}
    π0::Vector{N}  # vector of initial state probabilities
    A::Matrix{N}  # transition matrix (columns sum to 1)
    ξ::Matrix{N}  # mean of distribution
    Ξ::Array{N, 3}  # two-slice marginal of distribution
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

nstates(x::HMM) = length(x.π0)
size(x::HMM) = size(x.ψ)
length(x::HMM) = prod(size(x))

mean(x::HMM) = x.ξ

rand(x::HMM) = error("Not implemented!")

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
