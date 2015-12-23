# let's define some useful exponential family functions

using Distributions

naturals_to_params(η, d::Distribution) = naturals_to_params(η, typeof(d))

################# Normal ####################
function naturals(d::Normal)
    μ, σ = params(d)
    (μ/σ^2, -1/(2σ^2))
end

function naturals_to_params(η, ::Type{Normal})
    σ = sqrt(-1/(2η[2]))
    (η[1] * σ, σ)
end

################# Gamma ####################
function naturals(d::Gamma)
    a, θ = params(d)
    (a, 1/θ)
end

function naturals_to_params(η, ::Type{Gamma})
    (η[1] + 1, -η[2])
end

function Elog(d::Gamma)
    a, θ = params(d)
    digamma(a) + log(θ)
end

################# MvNormalCanon ####################
# exponential family representation of the multivariate normal
function naturals(d::MvNormalCanon)
    # h = Λ * μ
    # J = Λ
    (d.h, -d.J/2)
end

function naturals_to_params(η, ::Type{MvNormalCanon})
    # since the parameterization is already exponential family,
    # no need to convert
    (η[1], -2η[2])
end

convert(::Type{DiagNormalCanon}, v::Vector{Normal}) = begin
    J = 1 ./ map(var, v)
    μ = map(mean, v)
    DiagNormalCanon(J .* μ, J)
end

convert(::Type{IsoNormalCanon}, v::Normal) = begin
    J = 1 / var(v)
    μ = mean(v)
    DiagNormalCanon(J * μ, J)
end
