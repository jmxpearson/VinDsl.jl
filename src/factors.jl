# pre-defined factors

@deffactor EntropyFactor [x] H(x)

@deffactor LogNormalFactor [x, μ, τ] begin
    -(1/2) * ((E(τ) * ( V(x) + V(μ) + (E(x) - E(μ))^2 ) + log(2π) - Elog(τ)))
end

@deffactor LogGammaCanonFactor [x, α, β] begin
    (E(α) - 1) * Elog(x) - E(x) * E(β) + E(α) * Elog(β) - Eloggamma(α)
end

@deffactor LogGammaFactor [x, α, θ] begin
    (E(α) - 1) * Elog(x) - E(x)/E(θ) - E(α) * Elog(θ) - Eloggamma(α)
end

@deffactor LogMvNormalCanonFactor [x, μ, Λ] begin
    δ = E(x) - E(μ)
    EΛ = E(Λ)
    -(1/2) * (trace(EΛ * (C(x) .+ C(μ) .+ δ * δ')) + length(x) * log(2π) - Elogdet(Λ))
end

@deffactor LogMvNormalDiagCanonFactor [x, μ, τ] begin
    δ = E(x) - E(μ)
    Eτ = E(τ)
    -(1/2) * (sum(Eτ .* (V(x) .+ V(μ) .+ δ.^2 )) + length(x) * log(2π) - Elogdet(τ))
end

@deffactor LogDirichletFactor [x, α] begin
    dot(E(α) - 1, Elog(x)) - ElogB(α)
end

@deffactor LogWishartFactor [X ν V] begin
    EX = E(X)
    p = size(EX, 1)
    n = E(ν)
    0.5 * (n - p - 1) * Elogdet(X) - 0.5 * n * Elogdet(V) - 0.5 * trace(Einv(V) * EX) - 0.5 * n * p * log(2) - Elogmvgamma(p, ν/2)
end

################### Dealing with HMMs in factors #####################
@deffactor LogMarkovChainFactor [z, π0, A] begin
    dot(E(z)[:, 1], Elog(π0)) + sum(C(z) .* Elog(A))
end

@deffactor LogMarkovMatrixFactor [x, A] begin
    sum((E(A) - 1) .* Elog(x) .- ElogB(A))
end
