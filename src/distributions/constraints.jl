###################################################
# Random Variable Type Hierarchy
###################################################
#=
Arguably, much of this is duplicating insupport/support from distributions,
but is more usable for our purposes of mapping between constrained and
unconstrained model parameters.
=#
abstract RVType

abstract RScalar <: RVType
abstract RVector <: RVType
abstract RMatrix <: RVType

ndims(x::RScalar) = 1
nfree(x::RScalar) = 1

storage_type(rv::RScalar, T) = T
storage_type(rv::RVector, T) = Vector{T}
storage_type(rv::RMatrix, T) = Matrix{T}
###################################################
# random variable concrete types
###################################################
"""
Real-valued scalar.
"""
immutable RReal  <: RScalar
end
constrain(rv::RReal, x::Real) = x
unconstrain(rv::RReal, x::Real) = x
logdetjac(rv::RReal, x::Real) = 0.

"""
Positively constrained variable with (optional) lower bound.
"""
immutable RPositive{T <: Real}  <: RScalar
    lb::T
end
RPositive{T <: Real}(lb::T=0.) = RPositive{T}(lb)
constrain(rv::RPositive, x::Real) = exp(x) + rv.lb
unconstrain(rv::RPositive, x::Real) = log(x - rv.lb)
logdetjac(rv::RPositive, x::Real) = x

"""
Real-valued scalar.
"""
immutable RRealVec  <: RVector
    d::Int
end
ndims(x::RRealVec) = x.d
nfree(x::RRealVec) = ndims(x)
constrain(::RRealVec, x::Vector) = x
unconstrain(::RRealVec, x::Vector) = x
logdetjac(::RRealVec, x::Vector) = 0.

"""
Random Cholesky factor (lower triangular matrix with positive diagonal).
"""
immutable RCholFact <: RMatrix
    d::Int
end

ndims(x::RCholFact) = x.d
nfree(x::RCholFact) = (p = ndims(x); p * (p + 1) ÷ 2)

function constrain(rv::RCholFact, x::Vector)
    L = LowerTriangular(x)
    for j in 1:ndims(rv)
        L[j, j] = exp(L[j, j])  # Cholesky factor must have positive diagonals
    end
    L
end

function unconstrain(::RCholFact, S::LowerTriangular)
    L = copy(S)
    for j in 1:dim(S)
        L[j, j] = log(L[j, j])  # diagonal of Cholesky is positive, so take log
    end
    flatten(L)
end

logdetjac(rv::RCholFact, x::Vector) = sum(diag(LowerTriangular(x)))

"""
Random covariance matrix (symmetric, positive-definite).
"""
immutable RCovMat <: RMatrix
    d::Int
end

ndims(x::RCovMat) = x.d
nfree(x::RCovMat) = (p = ndims(x); p * (p + 1) ÷ 2)

function constrain(rv::RCovMat, x::Vector)
    L = LowerTriangular(x)
    for j in 1:ndims(rv)
        L[j, j] = exp(L[j, j])  # Cholesky factor must have positive diagonals
    end
    PDMat(Base.LinAlg.Cholesky(full(L), :L))
end

function unconstrain(::RCovMat, S::PDMat)
    L = copy(S.chol[:L])
    for j in 1:dim(S)
        L[j, j] = log(L[j, j])  # diagonal of Cholesky is positive, so take log
    end
    flatten(L)
end

function logdetjac(rv::RCovMat, x::Vector)
    d = ndims(rv)
    d * logtwo + (d + 1:-1:2) ⋅ diag(LowerTriangular(x))
end

constrain(pars, d::Distribution) = map(constrain, parsupp(d), pars)
unconstrain(d::Distribution) = map(unconstrain, parsupp(d), params(d))

"""
Number of free parameters needed for (multivariate) normal approximation
to the posterior over unconstrained parameters in ADVI.
"""
function num_pars_advi(rv::RVType, full=false)
    p = nfree(rv)
    full ? p * (p + 3) ÷ 2 : 2p
end
