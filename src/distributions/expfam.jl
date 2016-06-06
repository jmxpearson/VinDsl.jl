# let's define some useful exponential family functions

using Distributions

naturals_to_params(η, d::Distribution) = naturals_to_params(η, typeof(d).name.primary)

################# Normal ####################
function naturals(d::Normal)
    μ, σ = params(d)
    (μ/σ^2, -1/(2σ^2))
end

function naturals_to_params{D <: Normal}(η, ::Type{D})
    σ = sqrt(-1/(2η[2]))
    (η[1] * σ^2, σ)
end

function uparams(d::Normal)
    μ, σ = params(d)
    (μ, log(σ))
end

function constrain{D <: Normal}(pars, ::Type{D})
    (pars[1], exp(pars[2]))
end

################# Gamma ####################
function naturals(d::Gamma)
    a, θ = params(d)
    (a - 1, -1/θ)
end

function naturals_to_params{D <: Gamma}(η, ::Type{D})
    (η[1] + 1, -1/η[2])
end

function uparams(d::Gamma)
    a, θ = params(d)
    (log(a), log(θ))
end

function constrain{D <: Gamma}(pars, ::Type{D})
    (exp(pars[1]), exp(pars[2]))
end

function Elog(d::Gamma)
    a, θ = params(d)
    digamma(a) + log(θ)
end

# the following overwrites logpdf in Distributions
# this can be removed when Distributions has a pure Julia implementation
function logpdf(d::Gamma, x::Real)
    α, θ = params(d)
    (α - 1) * log(x) - (x/θ) - α * log(θ) - lgamma(α)
end

################# Dirichlet ####################
function naturals(d::Dirichlet)
    (d.alpha - 1,)
end

function naturals_to_params{D <: Dirichlet}(η, ::Type{D})
    (η[1] + 1,)
end

function Elog(d::Dirichlet)
    α = d.alpha
    digamma(α) - digamma(sum(α))
end

################# Poisson ####################
# the following overwrites logpdf in Distributions
# this can be removed when Distributions has a pure Julia implementation
function logpdf(d::Poisson, x::Int)
    (λ,) = params(d)
    x * log(λ) - λ - lgamma(x + 1)
end
logpdf(d::Poisson, X::Array{Int}) = map(x -> logpdf(d, x), X)

################# MvNormalCanon ####################
# exponential family representation of the multivariate normal
function naturals(d::MvNormalCanon)
    # h = Λ * μ
    # J = Λ
    (d.h, -d.J.mat/2)
end

function naturals_to_params{D <: MvNormalCanon}(η, ::Type{D})
    # since the parameterization is already exponential family,
    # no need to convert
    (η[1], -2η[2])
end

function uparams(d::MvNormalCanon)
    μ, h, J = params(d)
    U = J.chol[:U]
    (h, flatten(U))
end

function constrain{D <: MvNormalCanon}(pars, ::Type{D})
    U = UpperTriangular(pars[2])
    # could do much better here by creating PDMat directly using
    # U and not doing Cholesky all over again, but this is fine for now
    (pars[1], U' * U)
end

################# Wishart ####################
function naturals(d::Wishart)
    ((d.df - size(d)[1] - 1)/2, -inv(d.S).mat/2)
end

function naturals_to_params{D <: Wishart}(η, ::Type{D})
    (2η[1] + size(η[2], 1) + 1, -inv(η[2])/2)
end

function uparams(d::Wishart)
    df, S, _ = params(d)
    U = S.chol[:U]
    d = dim(S)
    (log(df - d + 1), flatten(U))
end

function constrain{D <: Wishart}(pars, ::Type{D})
    U = UpperTriangular(pars[2])
    d = size(U, 1)
    (exp(pars[1]) + d - 1, U' * U)
end

Elogdet(d::Wishart) = Distributions.meanlogdet(d)
