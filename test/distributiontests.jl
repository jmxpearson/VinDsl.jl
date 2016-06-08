# tests for distributions defined in VinDsl
using FactCheck
using Distributions
using PDMats

using VinDsl
srand(12349)

facts("Checking MatrixNormal distribution") do
    n, p = 5, 7
    LU = rand(n, n)
    LV = rand(p, p)
    U = PDMat(LU * LU')
    V = PDMat(LV * LV')
    M = rand(n, p)

    X = MatrixNormal(M, U, V)

    context("Constructors: 3-param") do
        @fact isa(X, MatrixNormal) --> true
        @fact isa(X, ContinuousMatrixDistribution) --> true
        @fact isa(X, Distribution) --> true
        @fact X.M --> M
        @fact X.U --> U
        @fact X.V --> V
    end

    context("Constructors: 2-param, same type") do
        XX = MatrixNormal(U, V)
        @fact isa(XX, MatrixNormal) --> true
        @fact XX.M --> zeros(dim(U), dim(V))
    end

    context("Constructors: 2-param, different type") do
        UU = ScalMat(n, 2.4)
        XX = MatrixNormal(U, V)
        @fact isa(XX, MatrixNormal) --> true
        @fact XX.M --> zeros(dim(UU), dim(V))
    end

    context("Constructors: 2-param, matrix and vector") do
        XX = MatrixNormal(U.mat, V.mat)
        @fact isa(XX, MatrixNormal) --> true

        XX = MatrixNormal(U, V.mat)
        @fact isa(XX, MatrixNormal) --> true

        XX = MatrixNormal(U.mat, V)
        @fact isa(XX, MatrixNormal) --> true

        XX = MatrixNormal(diag(U.mat), V)
        @fact isa(XX, MatrixNormal) --> true

        XX = MatrixNormal(diag(U.mat), V.mat)
        @fact isa(XX, MatrixNormal) --> true
    end

    context("Bad constructor inputs throw errors") do
        @fact_throws DimensionMismatch MatrixNormal(M[2:end, :], U, V)
        @fact_throws DimensionMismatch MatrixNormal(M[:, 2:end], U, V)
    end

    context("Check basic interface") do
        @fact size(X) --> (n, p)
        @fact params(X) --> (M, U, V)
    end

    context("Check mean") do
        m = mean(X)
        @fact m --> X.M
    end

    context("Check cov and var") do
        @fact cov(X) --> kron(V.mat, U.mat)
        @fact var(X) --> diag(U.mat) * diag(V.mat)'
        @fact invcov(X) --> roughly(invcov(convert(MvNormal, X)))
        @fact logdetcov(X) --> roughly(logdet(kron(full(V), full(U))))
    end

    context("Check sampling") do
        z = rand(X)
        @fact size(z) --> size(X)
    end

    context("Check logpdf") do
        z = rand(X)
        lpdf = logpdf(X, z)
        m = convert(MvNormal, X)

        @fact lpdf --> roughly(logpdf(m, vec(z)))
    end

    context("Check entropy") do
        @fact entropy(X) --> entropy(convert(MvNormal, X))
    end

end

facts("Checking exponential family interface") do
    context("Normal") do
        μ = 1.1
        σ = 2.5
        τ = 1/σ^2
        d = Normal(μ, σ)
        @fact naturals(d) --> (μ * τ, -τ/2)
        @fact naturals_to_params(naturals(d), Normal) --> (μ, σ)
        @fact Normal(constrain(unconstrain(d), d)...) --> d
        @fact supp(d) --> RReal()
    end

    context("Gamma") do
        a = 1.1
        θ = 2.5
        d = Gamma(a, θ)
        @fact naturals(d) --> (a - 1, -1/θ)
        @fact naturals_to_params(naturals(d), Gamma) --> (a, θ)
        @fact Gamma(constrain(unconstrain(d), d)...) --> d
        @fact supp(d) --> RPositive()
    end

    context("Dirichlet") do
        a = rand(5)
        α = a / sum(a)
        d = Dirichlet(α)
        @fact naturals(d) --> (α - 1,)
        @fact naturals_to_params(naturals(d), Dirichlet)[1] --> roughly(α)
        # @fact Dirichlet(constrain(unconstrain(d), Dirichlet)...) --> d
    end

    context("MvNormalCanon") do
        p = 7
        h = rand(p)
        jj = rand(p, p)
        J = jj * jj'
        d = MvNormalCanon(h, J)
        @fact naturals(d) --> (h, -J/2)
        @fact naturals_to_params(naturals(d), MvNormalCanon) --> (h, J)
        dd =  MvNormalCanon(constrain(unconstrain(d), d)...)
        @fact dd.h --> h
        @fact dd.J.mat --> roughly(J)
        @fact supp(d) --> RRealVec(length(d))
    end

    context("Wishart") do
        p = 7
        ss = rand(p, p)
        S = ss * ss'
        df = p + 10
        d = Wishart(df, S)
        nats = naturals(d)
        @fact nats[1] --> (df - p - 1)/2
        @fact nats[2] --> roughly(-inv(S)/2)
        n2p =  naturals_to_params(naturals(d), Wishart)
        @fact n2p[1] --> df
        @fact n2p[2] --> roughly(S)
        dd = Wishart(constrain(unconstrain(d), d)...)
        @fact dd.df --> df
        @fact dd.S.mat --> roughly(S)
        @fact supp(d) --> RCovMat(dim(d))
    end
end
