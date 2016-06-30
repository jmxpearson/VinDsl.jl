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
 Constrained variable with (optional) upper bound.
"""
immutable RNegative{T <: Real}  <: RScalar
    ub::T
end
RNegative{T <: Real}(ub::T=0.) = RNegative{T}(ub)
constrain(rv::RNegative, x::Real) = rv.ub - exp(x)
unconstrain(rv::RNegative, x::Real) = log(rv.ub - x)
logdetjac(rv::RNegative, x::Real) = x

"""
Constrained variable with (optional) upper and lower bound.
"""
immutable RBounded{T <: Real}  <: RScalar
    lb::T
    ub::T
end

RBounded{T <: Real}(lb::T = 0., ub::T = 1.0) = RBounded{T}(lb, ub)

function constrain(rv::RBounded, x::Real)
    if x > 0
        invlogitx = 1 / (1 + exp(-x))
        if x < Inf && invlogitx == 1
            invlogitx = 1 - eps()
        else
            invlogitx = 1
        end
    else
        invlogitx = 1 - 1 / (1 + exp(x))
        if x > -Inf && invlogitx == 0
            invlogitx = eps()
        else
            invlogitx = 0
        end
        rv.lb + (rv.ub - rv.lb) * invlogitx
    end
end

function unconstrain(rv::RBounded, x::Real)
    if rv.lb < x < rv.ub
        logitx = log((x - rv.lb) / (rv.ub - x))
    elseif x == rv.lb
        logitx = -Inf
    elseif x == rv.ub
        logitx = Inf
    end
    logitx
end

logdetjac(rv::RBounded, x::Real) = log(rv.ub - rv.lb) - x - 2log(1 + exp(-x))

"""
Probability constrained value.
"""
immutable RProbability <: RScalar
end

constrain(rv::RProbability, x::Real) = 1 / (1 + exp(-x))
unconstrain(rv::RProbability, x::Real) = log(x / (1 - x))
logdetjac(rv::RProbability, x::Real) = - x - 2log(1 + exp(-x))

"""
Correlation constrained value.
"""
immutable RCorrelation <: RScalar
end

constrain(rv::RCorrelation, x::Real) = (exp(2 * x) - 1) / (exp(2 * x) + 1)
unconstrain(rv::RCorrelation, x::Real) = .5log((1 + x) / (1 - x))
logdetjac(rv::RCorrelation, x::Real) = log(4) + 2 * x - 2 * log(1 + exp(2 * x))


"""
Real-valued vector.
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
Unit length vector.
"""
immutable RUnitVec  <: RVector
    d::Int
end
ndims(x::RUnitVec) = x.d
nfree(x::RUnitVec) = ndims(x)
constrain(::RUnitVec, x::Vector) = x ./ sqrt(dot(x, x))
unconstrain(::RUnitVec, x::Vector) = x .* sqrt(dot(x, x))
logdetjac(::RUnitVec, x::Vector) = - .5 * dot(x, x)

"""
Unit length vector.
"""
immutable RUnitVec  <: RVector
    d::Int
end
ndims(x::RUnitVec) = x.d
nfree(x::RUnitVec) = ndims(x)
constrain(::RUnitVec, x::Vector) = x ./ sqrt(dot(x, x))
unconstrain(::RUnitVec, x::Vector) = x .* sqrt(dot(x, x))
logdetjac(::RUnitVec, x::Vector) = - .5 * dot(x, x)

"""
Ordered Constraint
"""
immutable ROrdered  <: RVector
    d::Int
end
ndims(x::ROrdered) = x.d
nfree(x::ROrdered) = ndims(x)
function constrain(rv::ROrdered, x::Vector)
    y = x
    for j in 1:(ndims(rv) - 1)
        y[j + 1] = y[j] + exp(x[j + 1])
    end
    y
end

function unconstrain(::ROrdered, x::Vector)
    y = x
    for j in 1:(ndims(rv) - 1)
        y[j + 1] = log(x[j + 1] - x[j])
    end
    y
end

logdetjac(::ROrdered, x::Vector) = sum(x)

"""
Positive Ordered Constraint
"""
immutable RPosOrdered  <: RVector
    d::Int
end
ndims(x::RPosOrdered) = x.d
nfree(x::RPosOrdered) = ndims(x)
function constrain(rv::RPosOrdered, x::Vector)
    y = x
    y[1] = exp(x[1])
    for j in 1:(ndims(rv) - 1)
        y[j + 1] = y[j] + exp(x[j + 1])
    end
    y
end

function unconstrain(::RPosOrdered, x::Vector)
    y = x
    for j in 1:(ndims(rv) - 1)
        y[j + 1] = log(x[j + 1] - x[j])
    end
    y[1] = log(x[1])
    y
end

logdetjac(::RPosOrdered, x::Vector) = sum(x)


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
