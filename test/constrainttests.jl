
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
        @fact VinDsl.num_pars_advi(RReal()) --> 2

        rv = RReal()
        @fact constrain(rv, -2) --> -2
        @fact logdetjac(rv, -2.5) --> 0.
    end

    context("RPositive type interface") do
        @fact RPositive <: RVType --> true
        @fact RPositive <: RScalar --> true

        @fact ndims(RPositive()) --> 1
        @fact nfree(RPositive()) --> 1
        @fact VinDsl.num_pars_advi(RPositive()) --> 2

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
        @fact VinDsl.num_pars_advi(RRealVec(3)) --> 6
        @fact VinDsl.num_pars_advi(RRealVec(3), true) --> 9

        d = 3
        rv = RRealVec(d)
        vv = rand(d)
        @fact constrain(rv, vv) --> vv
        @fact logdetjac(rv, vv) --> 0.
    end

    context("RCholFact type interface") do
        @fact RCholFact <: RVType --> true
        @fact RCholFact <: RMatrix --> true

        @fact ndims(RCholFact(5)) --> 5
        @fact nfree(RCholFact(5)) --> 15
        @fact VinDsl.num_pars_advi(RCholFact(5)) --> 30
        @fact VinDsl.num_pars_advi(RCholFact(5), true) --> 135

        rv = RCholFact(5)
        vv = randn(5 * (5 + 1) ÷ 2)
        @fact isa(constrain(rv, vv), LowerTriangular) --> true
        @fact logdetjac(rv, vv) --> vv[1] + vv[6] + vv[10] + vv[13] + vv[15]
    end

    context("RCovMat type interface") do
        @fact RCovMat <: RVType --> true
        @fact RCovMat <: RMatrix --> true

        @fact ndims(RCovMat(5)) --> 5
        @fact nfree(RCovMat(5)) --> 15
        @fact VinDsl.num_pars_advi(RCovMat(5)) --> 30
        @fact VinDsl.num_pars_advi(RCovMat(5), true) --> 135

        rv = RCovMat(5)
        vv = randn(5 * (5 + 1) ÷ 2)
        @fact isa(constrain(rv, vv), PDMat) --> true
        @fact isfinite(logdetjac(rv, vv)) --> true
    end
end
