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

function Elog(d::Gamma)
    a, θ = params(d)
    digamma(a) + log(θ)
end