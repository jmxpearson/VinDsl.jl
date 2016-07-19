import Distributions.partype
import PDMats.dim
import Base.Random.rand
doc"""
    LKJ(η, p)

The *LKJ correlation matrix* with shape parameter \eta.

```julia
LKJ(eta, p)        # LKJ distribution with shape parameter \eta and dimension p

dim(d)            # Get the dimensions of LKJ correlation matrix
size(d)           # Get the size of correlation matrix as a tuple
params(d)         # Get the parameters, i.e. (mu, sig)

rand(d, n)          # Draw random samples from LKJ distribution, by default n=1
meanlogdet(d)       # Expected log determinant
meanloglkj(d, h)
entropy(d)
```

"""

immutable LKJ{T<:Real} <: ContinuousMatrixDistribution
    η::T
    p::Int

    function LKJ(η, p)
        η > zero(η) || error("η should be greater than zero.")
        new(η, p)
    end
end

LKJ(η::Real, p::Integer) = LKJ{typeof(η)}(η, p)
LKJ(η::Integer, p::Integer) = LKJ{Float64}(Float64(η), p)

@inline partype{T<:Real}(d::LKJ{T}) = T


"""
Draw n random samples with replacement from the LKJ distribution (n = 1 by default).
"""
function rand(d::LKJ, n::Integer)
    η = d.η
    p = d.p
    idxvec = [η + (p - i)/2. for i in 2:p] # generate Beta distribution indices
    L = zeros(p, p)
    for j in 1:length(idxvec)
        L[j+1:end, j] = 2 * rand(Distributions.Beta(idxvec[j], idxvec[j]), p - j) - 1
    end
    L = _cpc_to_corr(L, p)
    S = PDMats.PDMat(Base.LinAlg.Cholesky(full(L), :L)) # consistent with Stan Lib
    rand(S.mat, n)
end

rand(d::LKJ) = rand(d::LKJ, 1)

function _cpc_to_corr(L::AbstractMatrix, p::Real)  # consistent with Stan Lib
    #p = dim(d)
    LL = eye(p)
    for j in 1:p
        for i in j+1:p
            sumsqs = 0 # faster than dot(vec(L[i, 1:j]), vec(L[i, 1:j]))
            for ij in 1:j
                sumsqs += LL[i, ij]^2
            end
            LL[i, j] = L[i, j] * sqrt(1 - sumsqs)
        end
        sumsqs = 0
        for ij in 1:j-1
            sumsqs += LL[j, ij]^2
        end
        LL[j, j] = sqrt(1 - sumsqs)
    end
    LowerTriangular(LL)
end


dim(d::LKJ) = d.p
size(d::LKJ) = (d.p, d.p)
params(d::LKJ) = (d.η, d.p)


"""
E[log |A|] where
A ~ LKJ(eta) of dimension p
"""
function meanlogdet(d::LKJ)
    η = d.η
    p = d.p
    idxvec = [η + (p - i)/2. for i in 2:p]
    ldsum = 0
    for i in 1:length(idxvec)
        ldsum += (digamma(idxvec[i]) - digamma(2 * idxvec[i])) * (p - i)
    end
    ldsum
end

"""
E_q[log p(x)] where
p(x) = LKJ(h, p)  # p = matrix dimension
q(x) = LKJ(eta, p)
"""
function meanloglkj(d::LKJ, h::Real)
    η = d.η
    p = d.p
    betaq = [η + (p - i)/2. for i in 2:p]
    betap = [h + (p - i)/2. for i in 2:p]
    Elogx = 0
    for i in 1:length(betaq)
        Elogx += (p - i) * (2(betap[i] - 1) .* (digamma(betaq[i]) - digamma(2betaq[i])) - (2lgamma(betap[i]) - lgamma(2betap[i])))
    end
    Elogx
end

"""
Compute entropy for LKJ distribution by Dirichelet entropy
"""
function entropy(d::LKJ)
    η = d.η
    p = d.p
    idxvec = [η + (p - i)/2. for i in 2:p]
    H = 0
    for i in 1:length(idxvec)
        H += (p - i) * (2*lgamma(idxvec[i]) - lgamma(2idxvec[i]) + 2(idxvec[i] - 1) .* digamma(2idxvec[i]) - 2 * (idxvec[i] - 1) .* digamma(idxvec[i]))
    end
    H
end
