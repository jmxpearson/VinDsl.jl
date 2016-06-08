
facts("Random variable types") do
    context("Basic type assertions") do
        @fact RScalar <: RVType --> true
        @fact RVector <: RVType --> true
        @fact RMatrix <: RVType --> true
    end

    context("RReal type interface") do
        @fact RReal <: RVType --> true
        @fact RReal <: RScalar --> true

        @fact ndims(RReal()) --> 1
        @fact nfree(RReal()) --> 1

        rv = RReal()
        @fact constrain(rv, -2) --> -2
        @fact logdetjac(rv, -2.5) --> 0.
    end

    context("RPositive type interface") do
        @fact RPositive <: RVType --> true
        @fact RPositive <: RScalar --> true

        @fact ndims(RPositive()) --> 1
        @fact nfree(RPositive()) --> 1

        @fact RPositive().lb --> 0.
        @fact RPositive().lb --> 0.
        @fact RPositive(2).lb --> 2

        rv = RPositive(1.5)
        @fact constrain(rv, -2) ≥ rv.lb --> true
        @fact logdetjac(rv, -2.5) --> -2.5
    end

    context("RRealVec type interface") do
        @fact RRealVec <: RVType --> true
        @fact RRealVec <: RVector --> true

        @fact ndims(RRealVec(3)) --> 3
        @fact nfree(RRealVec(3)) --> 3

        d = 3
        rv = RRealVec(d)
        vv = rand(d)
        @fact constrain(rv, vv) --> vv
        @fact logdetjac(rv, vv) --> 0.
    end

    context("RCovMat type interface") do
        @fact RCovMat <: RVType --> true
        @fact RCovMat <: RMatrix --> true

        @fact ndims(RCovMat(5)) --> 5
        @fact nfree(RCovMat(5)) --> 15

        rv = RCovMat(5)
        vv = randn(5 * (5 + 1) ÷ 2)
        @fact isa(constrain(rv, vv), PDMat) --> true
        @fact isfinite(logdetjac(rv, vv)) --> true
    end
end


facts("Checking parameter number calculations") do
    context("Normal") do
        d = Normal(1)
        @fact VinDsl.num_pars_advi(d) --> 2
        @fact VinDsl.num_pars_advi(d, true) --> 2
    end

    context("Gamma") do
        d = Gamma(1)
        @fact VinDsl.num_pars_advi(d) --> 2
    end

    # context("Dirichlet") do
    #     d = Dirichlet(rand(3))
    #     @fact VinDsl.num_pars_advi(d) --> 9
    # end

    context("MvNormal") do
        d = MvNormal(rand(3))
        @fact VinDsl.num_pars_advi(d) --> 6
        @fact VinDsl.num_pars_advi(d, true) --> 9
    end

    context("Wishart") do
        U = rand(10, 10)
        S = U' * U
        d = Wishart(10, S)
        # a 10-d covariance matrix has 55 free parameters
        # a multivariate normal on this has 55 (mean) + 55 * 56 / 2 (cov) pars
        @fact VinDsl.num_pars_advi(d) --> 110
        @fact VinDsl.num_pars_advi(d, true) --> 1595
    end
end
