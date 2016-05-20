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
    #
    # context("Check natural parameters") do
    #     nats = naturals(x)
    #     pars = naturals_to_params(nats, typeof(x))
    #
    #     @fact length(nats) --> 3
    #     @fact size(nats[1]) --> size(x)
    #     @fact length(pars) --> 3
    #     @fact size(pars[1]) --> size(x)
    #     @fact nats[1] --> roughly(log(ψ))
    #     @fact nats[2] --> roughly(log(π0))
    #     @fact nats[3] --> roughly(log(A))
    #     @fact pars[1] --> roughly(ψ)
    #     @fact pars[2] --> roughly(π0)
    #     @fact pars[3] --> roughly(A)
    # end
    #
end
