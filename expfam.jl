# let's define some useful exponential family functions

using Distributions

function naturals(d::Normal)
    μ, σ = params(d)
    (μ/σ^2, -1/(2σ^2))
end

function naturals(d::Gamma)
    a, θ = params(d)
    (a, 1/θ)
end

function naturals_to_params(η, ::Type{Normal})
    σ = sqrt(-1/(2η[2]))
    (η[1] * σ, σ)
end

function naturals_to_params(η, ::Type{Gamma})
    (η[1] + 1, -η[2])
end

naturals_to_params(η, d::Distribution) = naturals_to_params(η, typeof(d))

function Elog(d::Gamma)
    a, θ = params(d)
    digamma(a) + log(θ)
end


