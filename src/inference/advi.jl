import Base.randn!

function advi_rand{T}(d::MvNormal{T})
    x = Array{T}(length(d))
    Distributions.add!(PDMats.unwhiten!(d.Σ, randn!(x)), d.μ)
end
function advi_rand{T}(d::MvNormal{T}, n::Int)
    x = Array{T}(length(d), n)
    Distributions.add!(PDMats.unwhiten!(d.Σ, randn!(x)), d.μ)
end
advi_rand(d::Normal) = d.μ + d.σ * randn()
function advi_rand{T}(d::Normal{T}, n::Int)
    x = Array(T, n)
    randn!(x)
    scale!(d.σ, x)
    x[:] += d.μ
end

function randn!{T<:ForwardDiff.Dual}(A::AbstractArray{T})
    for i in eachindex(A)
        @inbounds A[i] = randn()
    end
    A
end

"""
Calculate the entropy of a (multivariate) Normal distribution based on
a vector of unconstrained parameters for the mean and covariance.
"""
function H{T<:Real}(x::Vector{T}, full=false)
    if full
        # number of parameters: will throw InexactError if not an integer
        p = Int((-3 + sqrt(9 + 8 * length(x)))/2)

        # ½ logdet(Σ) = ∑ log L_ii, with L the Cholesky factor
        # but log L_ii is just the unconstrained parameter
        offset = p + 1  # first element of covariance part
        ldet = 0.
        skip = p
        while offset ≤ length(x)
            ldet += x[offset]
            offset += skip
            skip -= 1
        end
    else
        # return a multivariate normal with diagonal covariance
        p = Int(length(x)/2)
        ldet = sum(x[p+1:end])/2  # ½ ∑ log(σ)
    end
    p * (log2π + 1)/2 + ldet
end
