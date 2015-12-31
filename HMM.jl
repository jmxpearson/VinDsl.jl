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
    logZ = Array{Float64}(T)
    Ξ = Array{Float64}(M, M, T - 1)

    # initialize
    a = ψ[:, 1] .* π0
    α[:, 1] = a / sum(a)
    logZ[1] = log(sum(a))
    β[:, T] = 1/M

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

        logZ[t] = log(asum)
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

    γ, sum(logZ), Ξ

end
