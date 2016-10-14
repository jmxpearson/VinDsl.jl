# definitions of natural parameter messages from factors to nodes

@defnaturals LogNormalFactor x Normal begin
    Eμ, Eτ = E(μ), E(τ)
    (Eμ * Eτ, -Eτ/2)
end

@defnaturals LogNormalFactor μ Normal begin
    Ex, Eτ = E(x), E(τ)
    (Ex * Eτ, -Eτ/2)
end

@defnaturals LogNormalFactor τ Gamma begin
    v = V(x) + V(μ) + (E(x) - E(μ))^2
    (1/2, -v/2)
end

@defnaturals LogGammaFactor x Gamma begin
    (E(α) - 1, -1/E(θ))
end

@defnaturals LogGammaFactor x Gamma begin
    (E(α) - 1, -1/E(θ))
end

@defnaturals LogGammaCanonFactor x Gamma begin
    (E(α) - 1, -E(β))
end

nats_mvn{T <: Number}(μ::Vector{T}, Λ::Matrix{T}, x::MvNormalCanon) =
    (Λ * μ, -Λ/2)
nats_mvn{T <: Number}(μ::Number, Λ::Matrix{T}, x::MvNormalCanon) =
    (μ * sum(Λ, 2), -Λ/2)
nats_mvn{T <: Number}(μ::Vector{T}, τ::Vector{T}, x::MvNormalCanon) =
    (τ .* μ, -diagm(τ)/2)
nats_mvn{T <: Number}(μ::Number, τ::Vector{T}, x::MvNormalCanon) =
    (τ * μ, -diagm(τ)/2)
nats_mvn{T <: Number}(μ::Vector{T}, τ::Number, x::MvNormalCanon) =
    (τ * μ, -τ * eye(length(μ))/2)
nats_mvn(μ::Number, τ::Number, x::MvNormalCanon) =
    (μ * τ * ones(length(x)), -τ * eye(length(x))/2)
nats_mvn{T <: Number}(μ::Vector{T}, Λ::Matrix{T}, x::Normal) =
    (sum(Λ * μ), -sum(Λ)/2)
nats_mvn{T <: Number}(μ::Number, Λ::Matrix{T}, x::Normal) =
    (μ * sum(Λ), -sum(Λ)/2)
nats_mvn{T <: Number, D <: Normal}(μ::Vector{T}, τ::Vector{T}, x::Vector{D}) =
    [nats_mvn(m, t) for (m, t) in zip(μ, τ)]
nats_mvn{T <: Number, D <: Normal}(μ::Vector{T}, τ::Number, x::Vector{D}) =
    [nats_mvn(m, τ) for m in μ]
nats_mvn{T <: Number, D <: Normal}(μ::Number, τ::Vector{T}, x::Vector{D}) =
    [nats_mvn(μ, t) for t in τ]
nats_mvn(μ, τ, x::Normal) =
    reduce(add_nats, nats_mvn(μ, τ, [x]))
nats_mvn(μ::Number, τ::Number) = (μ * τ, -τ/2)
nats_mvn{D <: Gamma}(v::Vector, x::Vector{D}) =
    Tuple{Float64, Float64}[(1/2, vv/2) for vv in v]
nats_mvn(v::Vector, x::Gamma) =
    reduce(add_nats, nats_mvn(v, [x]))

@defnaturals LogMvNormalCanonFactor x MvNormalCanon begin
    nats_mvn(E(μ), E(Λ), x)
end

@defnaturals LogMvNormalCanonFactor μ MvNormalCanon begin
    nats_mvn(E(x), E(Λ), μ)
end

@defnaturals LogMvNormalCanonFactor x Normal begin
    nats_mvn(E(μ), E(Λ), x)
end

@defnaturals LogMvNormalCanonFactor μ Normal begin
    nats_mvn(E(x), E(Λ), μ)
end

@defnaturals LogMvNormalCanonFactor Λ Wishart begin
    δ = E(x) - E(μ)
    v = C(x) .+ C(μ) .+ δ * δ'
    (1/2, -v/2)
end

@defnaturals LogMvNormalDiagCanonFactor x Normal begin
    nats_mvn(E(μ), E(τ), x)
end

@defnaturals LogMvNormalDiagCanonFactor x MvNormalCanon begin
    nats_mvn(E(μ), E(τ), x)
end

@defnaturals LogMvNormalDiagCanonFactor μ Normal begin
    nats_mvn(E(x), E(τ), μ)
end

@defnaturals LogMvNormalDiagCanonFactor μ MvNormalCanon begin
    nats_mvn(E(x), E(τ), μ)
end

@defnaturals LogMvNormalDiagCanonFactor τ Gamma begin
    δ = E(x) - E(μ)
    v = V(x) + V(μ) + δ.^2
    nats_mvn(v, τ)
end

@defnaturals LogWishartFactor X Wishart begin
    n = E(ν)
    Vinv = Einv(V)
    p = size(Vinv, 1)
    ((n - p - 1)/2, -Vinv/2)
end

@defnaturals LogMarkovChainFactor π0 Dirichlet begin
    (E(z)[:, 1], )
end

@defnaturals LogMarkovChainFactor A MarkovMatrix begin
    (view(sum(C(z), 3), :, :), )
end

@defnaturals LogMarkovChainFactor z HMM begin
    (zero_like(z.ψ), Elog(π0), Elog(A))
end

@defnaturals LogDirichletFactor x Dirichlet begin
    (E(α) - 1,)
end

@defnaturals LogMarkovMatrixFactor x MarkovMatrix begin
    (E(A) - 1,)
end
