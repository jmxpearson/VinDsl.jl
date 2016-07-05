
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
        @fact logdetjac(rv, 1) --> log(2 - (-3)) - 1 - 2log(1 + exp(-1))
    end

    context("RProbability type interface") do
        @fact RProbability <: RVType --> true
        @fact RProbability <: RScalar --> true

        @fact ndims(RProbability()) --> 1
        @fact nfree(RProbability()) --> 1
        @fact VinDsl.num_pars_advi(RProbability()) --> 2

        rv = RProbability()
        @fact constrain(rv, 2) --> 1 / (1 + exp(-2))
        @fact logdetjac(rv, 2) --> - 2 - 2StatsFuns.log1pexp(-2) # Actually - 2 - 2log(1 + exp(-2)) has higher precision
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

    context("RUnitVec type interface") do
        @fact RUnitVec <: RVType --> true
        @fact RUnitVec <: RVector --> true

        @fact ndims(RUnitVec(3)) --> 3
        @fact nfree(RUnitVec(3)) --> 3
        @fact VinDsl.num_pars_advi(RUnitVec(3)) --> 6
        @fact VinDsl.num_pars_advi(RUnitVec(3), true) --> 9

        d = 3
        rv = RUnitVec(d)
        vv = rand(d)
        @fact vecnorm(constrain(rv, vv)) --> 1
        @fact logdetjac(rv, vv) --> - .5 * (vv[1]^2 + vv[2]^2 + vv[3]^2)
    end

    context("ROrdered type interface") do
        @fact ROrdered <: RVType --> true
        @fact ROrdered <: RVector --> true

        @fact ndims(ROrdered(3)) --> 3
        @fact nfree(ROrdered(3)) --> 3
        @fact VinDsl.num_pars_advi(ROrdered(3)) --> 6
        @fact VinDsl.num_pars_advi(ROrdered(3), true) --> 9

        d = 3
        rv = ROrdered(d)
        vv = rand(d)
        #vv0 = vv
        #@fact constrain(rv, vv) --> [vv0[1], vv0[1] + exp(vv0[2]), (vv0[1] + exp(vv0[2])) + exp(vv0[3])]
        @fact constrain(rv, vv) --> [vv[1], vv[2], vv[3]]
        @fact logdetjac(rv, vv) --> vv[1] + vv[2] + vv[3]
    end

    context("RPosOrdered type interface") do
        @fact RPosOrdered <: RVType --> true
        @fact RPosOrdered <: RVector --> true

        @fact ndims(RPosOrdered(3)) --> 3
        @fact nfree(RPosOrdered(3)) --> 3
        @fact VinDsl.num_pars_advi(RPosOrdered(3)) --> 6
        @fact VinDsl.num_pars_advi(RPosOrdered(3), true) --> 9

        d = 3
        rv = RPosOrdered(d)
        vv = rand(d)
        #@fact constrain(rv, vv) --> [vv0[1], vv0[1] + exp(vv0[2]), (vv0[1] + exp(vv0[2])) + exp(vv0[3])]
        @fact constrain(rv, vv)[1] > 0 --> true
        vvresult = constrain(rv, vv)
        @fact vvresult[3] > vvresult[2] > vvresult[1] --> true
        @fact logdetjac(rv, vv) --> vv[1] + vv[2] + vv[3]
    end

    context("RSimplex type interface") do
        @fact RSimplex <: RVType --> true
        @fact RSimplex <: RVector --> true

        @fact ndims(RSimplex(3)) --> 3
        @fact nfree(RSimplex(3)) --> 3
        @fact VinDsl.num_pars_advi(RSimplex(3)) --> 6
        @fact VinDsl.num_pars_advi(RSimplex(3), true) --> 9

        d = 3
        rv = RSimplex(d)
        #vv = rand(d)
        #@fact constrain(rv, vv) --> [vv0[1], vv0[1] + exp(vv0[2]), (vv0[1] + exp(vv0[2])) + exp(vv0[3])]
        #@fact constrain(rv, vv)[1] > 0 --> true
        #vvresult = constrain(rv, vv)
        #@fact vvresult[3] > vvresult[2] > vvresult[1] --> true
        #@fact logdetjac(rv, vv) --> vv[1] + vv[2] + vv[3]
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
        L1 = constrain(rv, vv)
        #println(L1)
        @fact isa(L1, LowerTriangular) --> true
        @fact logdetjac(rv, vv) --> vv[1] + vv[6] + vv[10] + vv[13] + vv[15]
    end

    context("RCholCorr type interface") do
        @fact RCholCorr <: RVType --> true
        @fact RCholCorr <: RMatrix --> true

        @fact ndims(RCholCorr(5)) --> 5
        @fact nfree(RCholCorr(5)) --> 10
        @fact VinDsl.num_pars_advi(RCholCorr(5)) --> 20
        @fact VinDsl.num_pars_advi(RCholCorr(5), true) --> 65

        rv = RCholCorr(5)
        vv = randn(5 * (5 - 1) ÷ 2)
        Lmatrix = constrain(rv, vv)
        #println(Lmatrix)
        @fact isa(Lmatrix, LowerTriangular) --> true
        @fact countnz(abs(Lmatrix) .<= 1) --> ndims(rv)^2
        #@fact logdetjac(rv, vv) --> vv[1] + vv[6] + vv[10] + vv[13] + vv[15]
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
