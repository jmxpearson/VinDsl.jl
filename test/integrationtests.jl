# integration tests of entire models

facts("Conjugate Normal model") do
    dims = (20, 6)

    # note: it won't matter much how we initialize here
    μ[j] ~ Normal(zeros(dims[2]), ones(dims[2]))
    τ[j] ~ Gamma(1.1 * ones(dims[2]), ones(dims[2]))
    μ0[j] ~ Const(zeros(dims[2]))
    τ0[j] ~ Const(2 * ones(dims[2]))
    a0[j] ~ Const(1.1 * ones(dims[2]))
    b0[j] ~ Const(ones(dims[2]))

    y[i, j] ~ Const(rand(dims))

    # make factors
    obs = @factor LogNormalFactor y μ τ
    μ_prior = @factor LogNormalFactor μ μ0 τ0
    τ_prior = @factor LogGammaCanonFactor τ a0 b0

    m = VBModel([μ, τ, μ0, τ0, a0, b0, y], [obs, μ_prior, τ_prior])

    @fact Set([n.name for n in m.nodes]) --> Set([:μ, :τ, :μ0, :τ0, :a0, :b0, :y])
    @fact Set([typeof(f) for f in m.factors]) --> Set([VinDsl.LogNormalFactor{2}, VinDsl.LogNormalFactor{1}, VinDsl.LogGammaCanonFactor{1}])
    @fact length(m.graph) --> 7
    @fact check_conjugate(τ, m) --> true
    @fact check_conjugate(μ, m) --> true

    update!(m)
end

facts("Univariate ⟷ multivariate naturals extraction") do
    context("vector mean, full covariance") do
        d = 5
        μ[i] ~ MvNormalCanon(zeros(d), diagm(ones(d)))
        Λ[i, i] ~ Wishart(float(d), diagm(ones(d)))
        x[i, j] ~ MvNormalCanon([randn(d) for x in 1:20], [diagm(ones(d)) for x in 1:20])
        f = @factor LogMvNormalCanonFactor x μ Λ

        @fact value(f) --> isfinite
        @fact map(size, naturals(f, μ)[1]) --> ((d,), (d, d))
        @fact map(size, naturals(f, Λ)[1]) --> ((), (d, d))
    end

    context("vector mean, diagonal covariance") do
        d = 5
        μ[i] ~ MvNormalCanon(zeros(d), diagm(ones(d)))
        τ[i] ~ Gamma(1.1 * ones(d), ones(d))
        x[i, j] ~ MvNormalCanon([randn(d) for x in 1:20], [diagm(ones(d)) for x in 1:20])
        f = @factor LogMvNormalDiagCanonFactor x μ τ

        @fact value(f) --> isfinite
        @fact map(size, naturals(f, μ)[1]) --> ((d,), (d, d))
        @fact size(naturals(f, τ)) --> (d,)
    end

    context("vector mean, scalar covariance") do
        d = 5
        μ[i] ~ MvNormalCanon(zeros(d), diagm(ones(d)))
        τ ~ Gamma(1.1, 1.)
        x[i, j] ~ MvNormalCanon([randn(d) for x in 1:20], [diagm(ones(d)) for x in 1:20])
        f = @factor LogMvNormalDiagCanonFactor x μ τ

        @fact value(f) --> isfinite
        @fact map(size, naturals(f, μ)[1]) --> ((d,), (d, d))
        @fact map(size, naturals(f, τ)[1]) --> ((), ())
    end

    context("scalar mean, full covariance") do
        d = 5
        μ ~ Normal(0, 1)
        Λ[i, i] ~ Wishart(float(d), diagm(ones(d)))
        x[i, j] ~ MvNormalCanon([randn(d) for x in 1:20], [diagm(ones(d)) for x in 1:20])
        f = @factor LogMvNormalCanonFactor x μ Λ

        @fact value(f) --> isfinite
        @fact map(size, naturals(f, μ)[1]) --> ((), ())
        @fact map(size, naturals(f, Λ)[1]) --> ((), (d, d))
    end

    context("scalar mean, diagonal covariance") do
        d = 5
        μ ~ Normal(0, 1)
        τ[i] ~ Gamma(1.1 * ones(d), ones(d))
        x[i, j] ~ MvNormalCanon([randn(d) for x in 1:20], [diagm(ones(d)) for x in 1:20])
        f = @factor LogMvNormalDiagCanonFactor x μ τ

        @fact value(f) --> isfinite
        @fact map(size, naturals(f, μ)[1]) --> ((), ())
        @fact map(size, naturals(f, τ)[1]) --> ((), ())
    end

    context("vector-of-scalars mean, diagonal covariance") do
        d = 5
        N = 20
        μ[i] ~ Normal(zeros(d), ones(d))
        τ[i] ~ Gamma(1.1 * ones(d), ones(d))
        x[i, j] ~ MvNormalCanon([randn(d) for x in 1:N], [diagm(ones(d)) for x in 1:N])
        f = @factor LogMvNormalDiagCanonFactor x μ τ

        @fact value(f) --> isfinite
        @fact size(naturals(f, μ)) --> (d,)
        @fact size(naturals(f, τ)) --> (d,)
        @fact size(naturals(f, x)) --> (N,)
        @fact map(size, naturals(f, μ)[1]) --> ((), ())
        @fact map(size, naturals(f, τ)[1]) --> ((), ())
        @fact map(size, naturals(f, x)[1]) --> ((d,), (d, d))
    end

    context("scalar mean, scalar covariance") do
        d = 5
        N = 20
        μ ~ Normal(0, 1)
        τ ~ Gamma(1.1, 1)
        x[i, j] ~ MvNormalCanon([randn(d) for x in 1:N], [diagm(ones(d)) for x in 1:N])
        f = @factor LogMvNormalDiagCanonFactor x μ τ

        @fact value(f) --> isfinite
        @fact map(size, naturals(f, μ)[1]) --> ((), ())
        @fact map(size, naturals(f, τ)[1]) --> ((), ())
        @fact size(naturals(f, x)) --> (N,)
        @fact map(size, naturals(f, x)[1]) --> ((d,), (d, d))
    end
end

facts("Basic Hidden Markov Model") do
    d = 5
    T = 100
    pars = [Dirichlet(d, 1) for i in 1:d]

    A_par = rand(MarkovMatrix(pars))
    π0_par = rand(Dirichlet(d, 1))
    ψ_par = rand(d, T)

    z[i, t] ~ HMM(ψ_par, π0_par, A_par)
    A ~ MarkovMatrix(pars)
    A_0 ~ Const(A_par)
    π0 ~ Dirichlet(π0_par)
    π0_0 ~ Const(π0_par)
    ψ[i, t] ~ Const(rand(d, T))

    f = @factor LogMarkovChainFactor z π0 A
    π_prior = @factor LogDirichletFactor π0 π0_0
    A_prior = @factor LogMarkovMatrixFactor A A_0
    π_nats = naturals(f, π0)
    A_nats = naturals(f, A)
    z_nats = naturals(f, z)

    @fact value(f) --> isfinite
    @fact value(π_prior) --> isfinite
    @fact value(A_prior) --> isfinite
    @fact map(size, naturals(f, π0)[1]) --> ((d,), )
    @fact map(size, naturals(f, A)[1]) --> ((d, d), )
    @fact map(size, naturals(f, z)[1]) --> ((d, T), (d,), (d, d))
    @fact map(size, naturals(π_prior, π0)[1]) --> ((d,), )
    @fact map(size, naturals(A_prior, A)[1]) --> ((d, d), )
end

facts("Linear combination expression node") do
    context("Univariate") do
        dims = (20, 6)
        μ[j] ~ Normal(zeros(dims[2]), ones(dims[2]))
        ν[j] ~ Normal(zeros(dims[2]), ones(dims[2]))
        τ[j] ~ Gamma(1.1 * ones(dims[2]), ones(dims[2]))
        @exprnode w (μ + ν)

        y[i, j] ~ Const(rand(dims))

        # make factors
        obs = @factor LogNormalFactor y w τ
        @fact value(obs) --> isfinite
    end

    context("Multivariate") do
        d = 5
        μ[i] ~ MvNormalCanon(zeros(d), diagm(ones(d)))
        ν[i] ~ MvNormalCanon(zeros(d), diagm(ones(d)))
        Λ[i, i] ~ Wishart(float(d), diagm(ones(d)))
        x[i, j] ~ MvNormalCanon([randn(d) for x in 1:20], [diagm(ones(d)) for x in 1:20])
        @exprnode w (μ + ν)
        f = @factor LogMvNormalCanonFactor x w Λ

        @fact value(f) --> isfinite
    end

    context("Matrix linear combination") do
        d = 5
        q = 7
        μ[i, k] ~ MvNormalCanon([zeros(d) for _ in 1:q], [diagm(ones(d)) for _ in 1:q])
        ν[k] ~ MvNormalCanon(zeros(q), diagm(ones(q)))
        Λ[i, i] ~ Wishart(float(d), diagm(ones(d)))
        x[i, j] ~ MvNormalCanon([randn(d) for _ in 1:20], [diagm(ones(d)) for _ in 1:20])
        @exprnode w (dot(μ[k], ν[k]))
        f = @factor LogMvNormalCanonFactor x w Λ

        # @fact value(f) --> isfinite
    end
end

facts("Updating via explicit optimization") do
    dims = (20, 6)

    # note: it won't matter much how we initialize here
    μ[j] ~ Normal(zeros(dims[2]), ones(dims[2]))
    τ[j] ~ Gamma(1.1 * ones(dims[2]), ones(dims[2]))
    μ0[j] ~ Const(zeros(dims[2]))
    τ0[j] ~ Const(2 * ones(dims[2]))
    a0[j] ~ Const(1.1 * ones(dims[2]))
    b0[j] ~ Const(ones(dims[2]))

    y[i, j] ~ Const(rand(dims))

    # make factors
    obs = @factor LogNormalFactor y μ τ
    μ_prior = @factor LogNormalFactor μ μ0 τ0
    τ_prior = @factor LogGammaCanonFactor τ a0 b0

    m = VBModel([μ, τ, μ0, τ0, a0, b0, y], [obs, μ_prior, τ_prior])

    context("L-BFGS") do
        update!(μ, m, Val{:l_bfgs})
    end
end
