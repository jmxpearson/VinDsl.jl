doc"""
    LKJcorr(η, p)

The *LKJ correlation matrix* with shape parameter \eta.

```julia
LKJ(eta, p)        # LKJ distribution with shape parameter \eta and dimension p

dim(d)
size(d)

params(d)         # Get the parameters, i.e. (mu, sig)
mean(d)           # Get the mean, i.e. mu
std(d)            # Get the standard deviation, i.e. sig
```

"""

immutable LKJcorr <: ContinuousMatrixDistribution
    η::Float64
    p::Float64
end


"""
Random draw from the LKJ distribution with parameter eta and dimension d.
"""
function LKJcorr(η::Real, p::Real)
    η > zero(η) || error("η should be greater than zero.")
    idxvec = [η + (p - i)/2. for i in 2:p] # generate Beta distribution indices
    L = zeros(p, p)
    for j in 1:length(idxvec)
        L[j+1:end, j] = 2 * rand(Distributions.Beta(idxvec[j], idxvec[j]), p - j) - 1
    end
    L = _cpc_to_corr(L, p)
    PDMats.PDMat(Base.LinAlg.Cholesky(full(L), :L)) # consistent with Stan Lib
#    β = _lkj_to_beta_pars(η, d)  # consistent with John's Python code
#    cpcs = zeros(length(β))
#    for i in 1:length(β)
#        cpcs[i] = 2 * rand(Distributions.Beta(β[i], β[i])) - 1
#    end
#    cpcs
#    LL = _cpc_to_corr(cpcs, d)
#    LL + transpose(LL) + diagm(ones(d))
end

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
    LL
end

#"""
#Transform LKJ distribution with parameter eta for matrix of dimension d
#to vector of beta distribution parameters:
#p_{i >= 1, j>i; 1...i-1} ~ Beta(b_i, b_i)
#b_i = eta + (d - 1 - i)/2
#"""
#function _lkj_to_beta_pars(η::Real, d::Int)
#    idxmat = (d - 2)/2. * ones(d)
#    for i in 2:d
#        idxmat = hcat(idxmat, (d - i - 1)/2. * ones(d))
#    end
#    bmat = η .+ idxmat
#    VinDsl.flatten(LowerTriangular(bmat[2:d, 1:d-1]))
#end


#"""
#Given a vector of canonical partial correlations (taken from the
#upper triangle by rows), return a vector of correlations in the
#same format.
#Makes use of the relation between partial correlations
#r_{ij;L} = \sqrt{(1 - r_{ik;L}^2)(1 - r_{jk;L}^2)} r_{ij;kL} + r_{ik;L} r_{jk;L}
#"""
#function _cpc_to_corr(x::Vector, d::Int)
#    L = hcat(vcat(zeros(d-1)', LowerTriangular(x)), zeros(d))
#    LL = zeros(similar(L))
#    LL[:, 1] = L[:, 1]
#    for j in 2:d
#        for i in j+1:d
#            rho = L[i, j]
#            for k in j-1:-1:1
#                rho = rho * sqrt((1 - L[i, k]^2) * (1 - L[j, k]^2)) + L[i, k] * L[j, k]
#            end
#            LL[i, j] = rho
#        end
#    end
#    LowerTriangular(LL)
#end

dim(d::LKJcorr) = d.p
size(d::LKJcorr) = (p = dim(d); (p, p))
params(d::LKJcorr) = (d.η, d.p)

"""
E[log |A|] where
A ~ LKJ(eta) of dimension d
"""
function logdetjac(d::LKJcorr)
    η = d.η
    p = dim(d)
    idxvec = [η + (p - i)/2. for i in 2:p]
    ldsum = 0
    for i in 1:length(idxvec)
        ldsum += (digamma(idxvec[i]) - digamma(2 * idxvec[i])) * (p - i)
    end
    ldsum
end


"""
Compute entropy for LKJ distribution by Dirichelet entropy
"""
function entropy(d::LKJcorr)
    η = d.η
    p = dim(d)
    idxvec = [η + (p - i)/2. for i in 2:p]
    H = 0
    for i in 1:length(idxvec)
        H += (p - i) * (2*lgamma(idxvec[i]) - lgamma(2idxvec[i]) + 2(idxvec[i] - 1) .* digamma(2idxvec[i]) - 2 * (idxvec[i] - 1) .* digamma(idxvec[i]))
    end
    ldsum
end
