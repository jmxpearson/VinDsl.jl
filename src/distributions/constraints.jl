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
RPositive() = RPositive(0.)
constrain(rv::RPositive, x::Real) = exp(x) + rv.lb
unconstrain(rv::RPositive, x::Real) = log(x - rv.lb)
logdetjac(rv::RPositive, x::Real) = x

"""
Constrained variable with (optional) upper bound.
"""
immutable RNegative{T <: Real}  <: RScalar
    ub::T
end
RNegative() = RNegative(0.)
constrain(rv::RNegative, x::Real) = rv.ub - exp(x)
unconstrain(rv::RNegative, x::Real) = log(rv.ub - x)
logdetjac(rv::RNegative, x::Real) = x

"""
Constrained variable with (optional) upper and lower bound.
Using logistic transformation.
"""
immutable RBounded{T <: Real}  <: RScalar
    lb::T
    ub::T
end

RBounded() = RBounded(0., 1.)
constrain(rv::RBounded, x::Real) = rv.lb + (rv.ub - rv.lb) * StatsFuns.logistic(x)
unconstrain(rv::RBounded, x::Real) = logit((x - rv.lb) / (rv.ub - rv.lb))
logdetjac(rv::RBounded, x::Real) = log(rv.ub - rv.lb) - x - 2 * StatsFuns.log1pexp(-x)

"""
Probability constrained value.
Using logistic transformation.
"""
immutable RProbability <: RScalar
end

constrain(rv::RProbability, x::Real) = StatsFuns.logistic(x)
unconstrain(rv::RProbability, x::Real) = StatsFuns.logit(x)
logdetjac(rv::RProbability, x::Real) = - x - 2 * StatsFuns.log1pexp(-x)

"""
Correlation constrained value.
Using tanh(x) function.
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
unconstrain(::RUnitVec, x::Vector, len::Real) = len * x  # need to provide the norm of original vector to free x
logdetjac(::RUnitVec, x::Vector) = - .5 * dot(x, x)


"""
Ordered Constraint.
"""
immutable ROrdered  <: RVector
    d::Int
end
ndims(x::ROrdered) = x.d
nfree(x::ROrdered) = ndims(x)
function constrain(rv::ROrdered, x::Vector)
    y = zeros(length(x))
    y[:] = x
    # Or use y = copy(x); drawback needed!
    for j in 1:(ndims(rv) - 1)
        y[j + 1] = y[j] + exp(x[j + 1]) # make sure the latter is greater than previous.
    end
    y
end

function unconstrain(rv::ROrdered, x::Vector)
    y = copy(x)
    for j in 1:(ndims(rv) - 1)
        y[j + 1] = log(x[j + 1] - x[j])
    end
    y
end

logdetjac(::ROrdered, x::Vector) = sum(x[2:end])

"""
Positive Ordered Constraint
"""
immutable RPosOrdered  <: RVector
    d::Int
end
ndims(x::RPosOrdered) = x.d
nfree(x::RPosOrdered) = ndims(x)
function constrain(rv::RPosOrdered, x::Vector)
    y = copy(x)
    y[1] = exp(x[1]) # Ensure the first is positive
    for j in 1:(ndims(rv) - 1)
        y[j + 1] = y[j] + exp(x[j + 1])
    end
    y
end

function unconstrain(rv::RPosOrdered, x::Vector)
    y = copy(x)
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
    y = copy(x)
    stick_len = 1 # entries sum to 1
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
    y = copy(x)
    d = length(x)
    stick_len = 1
    logdetx = 0
    for j in 1:ndims(rv)
        constan = x[j] - log(d - (j - 1))
        y[j] = stick_len * logistic(constan)
        logdetx += log(stick_len) - StatsFuns.log1pexp(-constan) - StatsFuns.log1pexp(constan)
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
    U = UpperTriangular(x)
    for j in 1:ndims(rv)
        U[j, j] = exp(U[j, j])  # Cholesky factor must have positive diagonals
    end
    transpose(U)
end

function unconstrain(rv::RCholFact, S::LowerTriangular)
    U = transpose(S)
    for j in 1:ndims(rv)
        U[j, j] = log(U[j, j])  # diagonal of Cholesky is positive, so take log
    end
    flatten(U)
end

logdetjac(rv::RCholFact, x::Vector) = sum(diag(UpperTriangular(x)))


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
    pos = 1
    U = eye(ndims(rv))
    for j in 1:ndims(rv)
        for i in 1:j-1
            sumsqs = 0 # faster than dot(vec(U.data[1:i, j]), vec(U.data[1:i, j]))
            for ij in 1:i
                sumsqs += U[ij, j]^2
            end
            U[i, j] = z[pos] * sqrt(1 - sumsqs)
            pos += 1
        end
        sumsqs = 0 # compute dot(U[1:j-1, j], U[1:j-1, j]) in loop
        for ij in 1:j-1
            sumsqs += U[ij, j]^2
        end
        U[j, j] = sqrt(1 - sumsqs)
    end
    transpose(UpperTriangular(U)) # faster than LowerTriangular(transpose(U))
end

function unconstrain(rv::RCholCorr, S::LowerTriangular)
    U = transpose(S)
    pos = 1
    z = zeros(Int(.5 * ndims(rv) * (ndims(rv) - 1)))
    for j in 2:ndims(rv)
        for i in 1:j-1
            sumsqs = 0  # faster than dot(vec(U.data[1:i-1, j]), vec(U.data[1:i-1, j]))
            for ij in 1:i-1
                sumsqs += U[ij, j]^2
            end
            z[pos] = U[i, j] / sqrt(1 - sumsqs)
            pos += 1
        end
    end
    atanh(z)
end


function logdetjac(rv::RCholCorr, x::Vector)
    z = tanh(x)
    pos = 2 # z[1] has no contribution to logdetjac
    U = eye(ndims(rv))
    logdetL = sum(log(4) + 2x - 2 * StatsFuns.log1pexp.(2x))
    for j in 3:ndims(rv)
        for i in 1:j-1
            sumsqs = 0
            for ij in 1:i
                sumsqs += U[ij, j]^2
            end
            logdetL += 0.5 * log1p(-sumsqs)
            U[i, j] = z[pos] * sqrt(1 - sumsqs)
            pos += 1
        end
        sumsqs = 0 # compute dot(U[1:j-1, j], U[1:j-1, j]) in loop
        for ij in 1:j-1
            sumsqs += U[ij, j]^2
        end
        U[j, j] = sqrt(1 - sumsqs)
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
    pos = 1
    L = eye(ndims(rv))
    for j in 1:ndims(rv)
        for i in j+1:ndims(rv)
            sumsqs = 0 # faster than dot(vec(L[i, 1:j]), vec(L[i, 1:j]))
            for ij in 1:j
                sumsqs += L[i, ij]^2
            end
            L[i, j] = z[pos] * sqrt(1 - sumsqs)
            pos += 1
        end
        sumsqs = 0
        for ij in 1:j-1
            sumsqs += L[j, ij]^2
        end
        L[j, j] = sqrt(1 - sumsqs)
    end
    PDMat(Base.LinAlg.Cholesky(full(L), :L))
end

function unconstrain(::RCorrMat, S::PDMat)
    L = copy(S.chol[:L])
    K = dim(S)
    z = zeros(Int(.5 * K * (K - 1)))
    z[1:K-1] = L.data[2:K, 1]
    pos = K
    for j in 2:K
        for i in j+1:K
            sumsqs = 0
            for ij in 1:j-1
                sumsqs += L[i, ij]^2
            end
            z[pos] = L[i, j] / sqrt(1 - sumsqs)
            pos += 1
        end
    end
    atanh(z)
end


function logdetjac(rv::RCorrMat, x::Vector)
    z = tanh(x)
    pos = 1
    val = zeros(length(x))
    logdetL = length(x) * log(4) + 2sum(x) - 2sum(StatsFuns.log1pexp.(2x))
    for k in 1:ndims(rv)-2
        for i in k+1:ndims(rv)
            val[pos] = (ndims(rv) - k - 1) * log1p(-z[pos]^2) # from Eq (11) of LKJ paper
            pos += 1
        end
    end
    logdetL += .5 * sum(val)
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
    U = UpperTriangular(x)
    for j in 1:ndims(rv)
        U[j, j] = exp(U[j, j])  # Cholesky factor must have positive diagonals
    end
    PDMat(Base.LinAlg.Cholesky(full(U), :U))
end

function unconstrain(::RCovMat, S::PDMat)
    U = copy(S.chol[:U])
    for j in 1:dim(S)
        U[j, j] = log(U[j, j])  # diagonal of Cholesky is positive, so take log
    end
    flatten(U)
end

function logdetjac(rv::RCovMat, x::Vector)
    d = ndims(rv)
    d * logtwo + (d+1:-1:2) ⋅ diag(LowerTriangular(x))
end

"""
Random LKJ covariance matrix (symmetric, positive-definite).
"""
immutable RCovLKJ <: RMatrix
    d::Int
end

ndims(x::RCovLKJ) = x.d
nfree(x::RCovLKJ) = (p = ndims(x); p * (p + 1) ÷ 2)

function constrain(rv::RCovLKJ, x::Vector)
    d = ndims(rv)
    K = Int(.5d * (d - 1))  # if K not an integer, should throw an error message
    z = tanh(x[1:K])
    diagz = exp(x[K + 1:end])
    pos = 1
    L = eye(ndims(rv))
    for j in 1:ndims(rv)
        for i in j+1:ndims(rv)
            sumsqs = 0
            for ij in 1:j
                sumsqs += L[i, ij]^2
            end
            #sumsqs = dot(vec(L[i, 1:j]), vec(L[i, 1:j]))
            L[i, j] = z[pos] * sqrt(1 - sumsqs)
            pos += 1
        end
        sumsqs = 0
        for ij in 1:j-1
            sumsqs += L[j, ij]^2
        end
        L[j, j] = sqrt(1 - sumsqs)
    end
    PDMat(Base.LinAlg.Cholesky(full(Base.LinAlg.Diagonal(diagz) * L), :L))
end


function unconstrain(::RCovLKJ, S::PDMat)
    K = dim(S)
    z = zeros(Int(.5K * (K + 1)))
    z[Int(.5K * (K - 1))+1:end] = .5log(diag(S)) # identify the σ^2 variances
    L = S.chol[:L] ./ sqrt(diag(S))
    z[1:K-1] = L[2:K, 1]
    pos = K
    for j in 2:K
        for i in j+1:K
            sumsqs = 0
            for ij in 1:j-1
                sumsqs += L[i, ij]^2
            end
            z[pos] = L[i, j] / sqrt(1 - sumsqs)
            pos += 1
        end
    end
    z[1:Int(.5K * (K - 1))] = atanh(z[1:Int(.5K * (K - 1))])
    z
end

function logdetjac(rv::RCovLKJ, x::Vector)
    d = ndims(rv)
    K = Int(.5d * (d - 1))
    z = tanh(x[1:K])
    pos = 1
    val = zeros(length(x))
    logdetL = K * log(4) + 2sum(x[1:K]) - 2sum(StatsFuns.log1pexp.(2x[1:K]))
    logdetL += sum(x[K+1:end]) # log determinant for positive constraint
    logdetL += d .* (sum(x[K+1:end]) + log(2)) # correction from correlation to covariance
    for k in 1:d-2
        for i in k+1:d
            val[pos] = (d - k - 1) * log1p(-z[pos]^2)
            pos += 1
        end
    end
    logdetL += .5 * sum(val) # log determinant for correlation constraint
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
