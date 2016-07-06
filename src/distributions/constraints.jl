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
constrain(rv::RBounded, x::Real) = rv.lb + (rv.ub - rv.lb) * StatsFuns.logistic(x)
unconstrain(rv::RBounded, x::Real) = logit((x - rv.lb) / (rv.ub - rv.lb))
logdetjac(rv::RBounded, x::Real) = log(rv.ub - rv.lb) - x - 2 * StatsFuns.log1pexp(-x)

"""
Probability constrained value.
"""
immutable RProbability <: RScalar
end

constrain(rv::RProbability, x::Real) = StatsFuns.logistic(x)
unconstrain(rv::RProbability, x::Real) = StatsFuns.logit(x)
logdetjac(rv::RProbability, x::Real) = - x - 2log1pexp(-x)

"""
Correlation constrained value.
"""
immutable RCorrelation <: RScalar
end

constrain(rv::RCorrelation, x::Real) = tanh(x)
unconstrain(rv::RCorrelation, x::Real) = atanh(x)
logdetjac(rv::RCorrelation, x::Real) = log(4) + 2x - 2 * StatsFuns.log1pexp(2x)


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

function unconstrain(rv::ROrdered, x::Vector)
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

function unconstrain(rv::RPosOrdered, x::Vector)
    y = x
    for j in 1:(ndims(rv) - 1)
        y[j + 1] = log(x[j + 1] - x[j])
    end
    y[1] = log(x[1])
    y
end

logdetjac(::RPosOrdered, x::Vector) = sum(x)

"""
Simplex Constraint (entries sum to one)
"""
immutable RSimplex  <: RVector
    d::Int
end
ndims(x::RSimplex) = x.d
nfree(x::RSimplex) = ndims(x)
function constrain(rv::RSimplex, x::Vector)
    y = x
    stick_len = 1
    for j in 1:ndims(rv)
        y[j] = stick_len * logistic(x[j] - log(ndims(rv) - (j - 1)))
        stick_len -= y[j]
    end
    y! = [y; stick_len]
end

function unconstrain(rv::RSimplex, x::Vector)
    y = zeros(ndims(rv))
    stick_len = 1
    for j in 1:ndims(rv)
        y[j] = logit(x[j] / stick_len) + log(ndims(rv) - (j - 1))
        stick_len -= x[j]
    end
    y
end

function logdetjac(rv::RSimplex, x::Vector)
    y = x
    stick_len = 1
    for j in 1:ndims(rv)
        constan = x[j] - log(ndims(rv) - (j - 1))
        y[j] = stick_len * logistic(constan)
        logdetx += log(stick_len) - StatsFuns.log1pexp(-constan) + StatsFuns.log1pexp(constan)
        stick_len -= y[j]
    end
    logdetx
end


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
Cholesky Correlation Matrix (lower triangular matrix with elements between -1 and 1).
"""
immutable RCholCorr <: RMatrix
    d::Int
end

ndims(x::RCholCorr) = x.d
nfree(x::RCholCorr) = (p = ndims(x); p * (p - 1) ÷ 2)

function constrain(rv::RCholCorr, x::Vector)
    z = tanh(x)
    k = 1
    L = eye(ndims(rv))
    for i in 2:ndims(rv)
        L[i, 1] = z[k]
        k += 1
        sumsqs = L[i, 1]^2
        for j in 2:i-1
            L[i, j] = z[k] * sqrt(1 - sumsqs)
            k += 1
            sumsqs += L[i, j]^2
        end
        L[i, i] = sqrt(1 - sumsqs)
    end
    LowerTriangular(L)
end

function unconstrain(::RCholCorr, S::LowerTriangular)
    L = copy(S)
    k = 1
    for i in 2:dim(S)
        z[k] = L[i, 1]
        k += 1
        sumsqs = L[i, 1]^2
        for j in 2:i
            z[k] = L[i, j] / sqrt(1 - sumsqs)
            k += 1
            sumsqs += L[i, j]^2
        end
    end
    atanh(z)
end

function logdetjac(rv::RCholCorr, x::Vector)
    z = tanh(x)
    k = 1
    L = eye(ndims(rv))
    logdetL = sum(log(4) + 2x - 2 * StatsFuns.log1pexp(2x))
    for i in 2:ndims(rv)
        L[i, 1] = z[k]
        k += 1
        sumsqs = L[i, 1]^2
        for j in 2:i-1
            logdetL += 0.5 * log1p(-sumsqs)
            L[i, j] = z[k] * sqrt(1 - sumsqs)
            k += 1
            sumsqs += L[i, j]^2
        end
        L[i, i] = sqrt(1 - sumsqs)
    end
    logdetL
end

"""
Random Correlation matrix (symmetric, positive-definite).
"""
immutable RCorrMat <: RMatrix
    d::Int
end

ndims(x::RCorrMat) = x.d
nfree(x::RCorrMat) = (p = ndims(x); p * (p - 1) ÷ 2)

function constrain(rv::RCorrMat, x::Vector)
    z = tanh(x)
    k = 1
    L = eye(ndims(rv))
    for i in 2:ndims(rv)
        L[i, 1] = z[k]
        k += 1
        sumsqs = L[i, 1]^2
        for j in 2:i-1
            L[i, j] = z[k] * sqrt(1 - sumsqs)
            k += 1
            sumsqs += L[i, j]^2
        end
        L[i, i] = sqrt(1 - sumsqs)
    end
    lowerL = LowerTriangular(L)
    lowerL * transpose(lowerL)
end

function unconstrain(::RCorrMat, S::PDMat)
    L = copy(S.chol[:L])
    k = 1
    for i in 2:dim(S)
        z[k] = L[i, 1]
        k += 1
        sumsqs = L[i, 1]^2
        for j in 2:i
            z[k] = L[i, j] / sqrt(1 - sumsqs)
            k += 1
            sumsqs += L[i, j]^2
        end
    end
    atanh(z)
end


###### Need revise for the following logdetjac!!!
function logdetjac(rv::RCorrMat, x::Vector)
    d = ndims(rv)
    z = tanh(x)
    logdetL = d * log(4) + 2 * sum(x) - 2 * sum(log1pexp(2x))
    k = 1
    L = eye(ndims(rv))
    for i in 2:ndims(rv)
        L[i, 1] = z[k]
        k += 1
        sumsqs = L[i, 1]^2
        for j in 2:i-1
            logdetL += 0.5 * sumsqs
            L[i, j] = z[k] * sqrt(1 - sumsqs)
            k += 1
            sumsqs += L[i, j]^2
        end
        L[i, i] = sqrt(1 - sumsqs)
    end
    logdetL
end

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
