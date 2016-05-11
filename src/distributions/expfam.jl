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
    (η[1] * σ, σ)
end

################# Gamma ####################
function naturals(d::Gamma)
    a, θ = params(d)
    (a, 1/θ)
end

function naturals_to_params{D <: Gamma}(η, ::Type{D})
    (η[1] + 1, -η[2])
end

function Elog(d::Gamma)
    a, θ = params(d)
    digamma(a) + log(θ)
end

################# Dirichlet ####################
function naturals(d::Dirichlet)
    (d.alpha - 1,)
end

function naturals_to_params{D <: Dirichlet}(η, ::Type{D})
    (η + 1,)
end

function Elog(d::Dirichlet)
    α = d.alpha
    digamma(α) - digamma(sum(α))
end

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

################# Wishart ####################
function naturals(d::Wishart)
    (-inv(d.S).mat/2, (d.df - size(d)[1] - 1)/2)
end

function naturals_to_params{D <: Wishart}(η, ::Type{D})
    (-inv(η[1])/2, 2η[2] + size(eta[1], 1) + 1)
end

Elogdet(d::Wishart) = Distributions.meanlogdet(d)
