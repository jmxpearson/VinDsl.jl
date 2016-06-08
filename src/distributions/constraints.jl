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
# size(x::RVector) = (D,)
# size(x::RMatrix) = (P, Q)

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
constrain(rv::RRealVec, x::Vector) = x
unconstrain(rv::RRealVec, x::Vector) = x
logdetjac(rv::RRealVec, x::Vector) = 0.

"""
Random covariance matrix (symmetric, positive-definite).
"""
immutable RCovMat <: RMatrix
    d::Int
end

ndims(x::RCovMat) = x.d
nfree(x::RCovMat) = (p = ndims(x); p * (p + 1) ÷ 2)

function constrain(rv::RCovMat, x::Vector)
    U = UpperTriangular(x)
    for j in 1:ndims(rv)
        U[j, j] = exp(U[j, j])  # Cholesky factor must have positive diagonals
    end
    PDMat(Base.LinAlg.Cholesky(full(U), :U))
end

function unconstrain(rv::RCovMat, S::PDMat)
    U = copy(S.chol[:U])
    for j in 1:dim(S)
        U[j, j] = log(U[j, j])  # diagonal of Cholesky is positive, so take log
    end
    flatten(U)
end

function logdetjac(rv::RCovMat, x::Vector)
    d = ndims(rv)
    d * logtwo + (d + 1:-1:2) ⋅ diag(UpperTriangular(x))
end

constrain(pars, d::Distribution) = map(constrain, parsupp(d), pars)
unconstrain(d::Distribution) = map(unconstrain, parsupp(d), params(d))

"""
Number of free parameters needed for (multivariate) normal approximation
to the posterior over unconstrained parameters in ADVI.
"""
function num_pars_advi(d::Distribution, full=false)
    p = nfree(supp(d))
    full ? p * (p + 3) ÷ 2 : 2p
end
