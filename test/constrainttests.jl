
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

        rv = RReal()
        @fact constrain(rv, -2) --> -2
        @fact logdetjac(rv, -2.5) --> 0.
    end

    context("RPositive type interface") do
        @fact RPositive <: RVType --> true
        @fact RPositive <: RScalar --> true

        @fact ndims(RPositive()) --> 1
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

        d = 3
        rv = RRealVec(d)
        vv = rand(d)
        @fact constrain(rv, vv) --> vv
        @fact logdetjac(rv, vv) --> 0.
    end

    context("RCholCov type interface") do
        @fact RCholCov <: RVType --> true
        @fact RCholCov <: RMatrix --> true

        @fact ndims(RCholCov(5)) --> 5

        rv = RCholCov(5)
        vv = randn(5 * (5 + 1) ÷ 2)
        @fact isa(constrain(rv, vv), PDMat) --> true
        @fact isfinite(logdetjac(rv, vv)) --> true
    end
end
