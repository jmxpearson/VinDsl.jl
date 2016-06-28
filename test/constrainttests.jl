
facts("Random variable types") do
    context("Basic type assertions") do
        @fact RScalar <: RVType --> true
        @fact RVector <: RVType --> true
        @fact RMatrix <: RVType --> true

        @fact VinDsl.storage_type(RReal(), Float64) --> Float64
        @fact VinDsl.storage_type(RRealVec(5), Float64) --> Array{Float64, 1}
        @fact VinDsl.storage_type(RCovMat(5), Float64) --> Array{Float64, 2}
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
        @fact logdetjac(rv, 2.5) --> 2.5
    end

    context("RNegative type interface") do
        @fact RNegative <: RVType --> true
        @fact RNegative <: RScalar --> true

        @fact ndims(RNegative()) --> 1
        @fact nfree(RNegative()) --> 1
        @fact VinDsl.num_pars_advi(RNegative()) --> 2

        @fact RNegative().ub --> 0.
        @fact RNegative().ub --> 0.
        @fact RNegative(-2).ub --> -2

        rv = RNegative(-1.5)
        @fact constrain(rv, 2) ≤ rv.ub --> true
        @fact logdetjac(rv, -2.5) --> -2.5
    end

    context("RBounded type interface") do
        @fact RBounded <: RVType --> true
        @fact RBounded <: RScalar --> true

        @fact ndims(RBounded()) --> 1
        @fact nfree(RBounded()) --> 1
        @fact VinDsl.num_pars_advi(RBounded()) --> 2

        @fact RBounded().ub --> 1
        @fact RBounded().lb --> 0.
        @fact RBounded(-3, 2).ub --> 2
        @fact RBounded(-3, 2).lb --> -3

        rv = RBounded(-3, 2)
        @fact constrain(rv, 1) ≤ rv.ub --> true
        @fact constrain(rv, 1) ≥ rv.lb --> true
        @fact logdetjac(rv, 1) --> log(2 - (-3)) - 1 - 2 * log(1 + exp(-1))
    end

    context("RProbability type interface") do
        @fact RProbability <: RVType --> true
        @fact RProbability <: RScalar --> true

        @fact ndims(RProbability()) --> 1
        @fact nfree(RProbability()) --> 1
        @fact VinDsl.num_pars_advi(RProbability()) --> 2

        rv = RProbability()
        @fact constrain(rv, 2) --> 1 / (1 + exp(-2))
        @fact logdetjac(rv, 2) --> - 2 - 2 * log(1 + exp(-2))
    end

    context("RCorrelation type interface") do
        @fact RCorrelation <: RVType --> true
        @fact RCorrelation <: RScalar --> true

        @fact ndims(RCorrelation()) --> 1
        @fact nfree(RCorrelation()) --> 1
        @fact VinDsl.num_pars_advi(RCorrelation()) --> 2

        rv = RCorrelation()
        @fact constrain(rv, 3) --> (exp(6) - 1) / (exp(6) + 1)
        @fact logdetjac(rv, 3) --> log(4) + 6 - 2log(1 + exp(6))
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
