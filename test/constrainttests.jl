
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

        rv = RPositive(2)
        y1 = constrain(rv, -1)
        x1 = unconstrain(rv, y1)
        lp = logdetjac(rv, -1)
        @fact y1 ≥ rv.lb --> true
        @fact x1 --> -1
        @fact lp --> -1
        #println("lb constraint: ", y1, " free: ", x1, "\t", lp)
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

        rv = RNegative(2)
        y2 = constrain(rv, -1)
        x2 = unconstrain(rv, y2)
        lp = logdetjac(rv, -1)
        @fact y2 ≤ rv.ub --> true
        @fact x2 --> -1
        @fact lp --> -1
        #println("ub constraint: ", y2, " free: ", x2, "\t", lp)
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

        rv = RBounded(2, 4)
        y3 = constrain(rv, -1)
        x3 = unconstrain(rv, y3)
        lp = logdetjac(rv, -1)
        @fact y3 ≤ rv.ub --> true
        @fact y3 ≥ rv.lb --> true
        @fact y3 --> 2.5378828427399904
        @fact round(x3, 8) --> -1
        @fact lp --> -0.9333761944765002
        #println("lub constraint: ", y3, " free: ", x3, "\t", lp)
        #@fact logdetjac(rv, 1) --> log(4 - 2) - 1 - 2log(1 + exp(-1))
    end

    context("RProbability type interface") do
        @fact RProbability <: RVType --> true
        @fact RProbability <: RScalar --> true

        @fact ndims(RProbability()) --> 1
        @fact nfree(RProbability()) --> 1
        @fact VinDsl.num_pars_advi(RProbability()) --> 2

        rv = RProbability()
        y4 = constrain(rv, -1)
        x4 = unconstrain(rv, y4)
        lp = logdetjac(rv, -1)
        @fact y4 --> 0.2689414213699951
        @fact x4 --> -1
        @fact lp --> -1.6265233750364456
        #println("prob constraint: ", y4, " free: ", x4, "\t", lp)
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
        y5 = constrain(rv, -1)
        x5 = unconstrain(rv, y5)
        lp = logdetjac(rv, -1)
        @fact y5 --> -0.7615941559557649
        @fact round(x5, 8) --> -1
        @fact lp --> -0.8675616609660544
        #println("correlation constraint: ", y5, " free: ", x5, "\t", lp)
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
        vv = [1, -1, 2]
        #vv = rand(d)
        yy1 = constrain(rv, vv)
        xx1 = unconstrain(rv, yy1, sqrt(dot(vv,vv)))
        lp = logdetjac(rv, vv)
        @fact yy1 --> [0.4082482904638631,-0.4082482904638631,0.8164965809277261]
        @fact xx1 --> [1.0,-1.0,2.0]
        @fact lp --> -3
        #println("Unit vector constraint: ", yy1)
        #println("free: ", xx1)
        #println("log determinant: ", lp)
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
        #vv = rand(d)
        vv = [1., -1., 2.]
        yy2 = constrain(rv, vv)
        xx2 = unconstrain(rv, yy2)
        lp = logdetjac(rv, vv)
        @fact yy2 --> [1.0,1.3678794411714423,8.756935540102093]
        @fact xx2 --> [1.0,-1.0,2.0]
        @fact lp --> 1
        #println("Ordered vector constraint: ", yy2)
        #println("free: ", xx2)
        #println("log determinant: ", lp)
        @fact constrain(rv, vv) --> [vv[1], vv[1] + exp(vv[2]), (vv[1] + exp(vv[2])) + exp(vv[3])]
        @fact logdetjac(rv, vv) --> vv[2] + vv[3]
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
        #vv = rand(d)
        vv = [1., -1., 2.]
        yy3 = constrain(rv, vv)
        xx3 = unconstrain(rv, yy3)
        lp = logdetjac(rv, vv)
        @fact yy3 --> [2.718281828459045,3.0861612696304874,10.475217368561138]
        @fact xx3 --> [1.0,-1.0,2.0]
        @fact lp --> 2
        #println("Positive ordered constraint: ", yy3)
        #println("free: ", xx3 )
        #println("log determinant: ", lp)
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
        vv = [1.0, -1.0, 2.0]
        #vv = rand(d)
        yy4 = constrain(rv, vv)
        xx4 = unconstrain(rv, yy4)
        lp = logdetjac(rv, vv)
        @fact yy4 --> [0.4753668864186718,0.08150826148009065,0.3903030749101513,0.05282177719108627]
        @fact round(xx4, 8) --> [1, -1, 2]
        @fact lp --> -7.132382729651196
        #println("Simplex constraint: ", yy4)
        #println("free: ", xx4)
        #println("log determinant: ", lp)
        @fact length(yy4) --> d + 1
        @fact sum(yy4) --> 1
    end

    context("RCholFact type interface") do
        @fact RCholFact <: RVType --> true
        @fact RCholFact <: RMatrix --> true

        @fact ndims(RCholFact(5)) --> 5
        @fact nfree(RCholFact(5)) --> 15
        @fact VinDsl.num_pars_advi(RCholFact(5)) --> 30
        @fact VinDsl.num_pars_advi(RCholFact(5), true) --> 135

        rv = RCholFact(2)
        vv = [1., 2., 3., 4., 5., 6.]
        L1 = constrain(rv, vv)
        x1 = unconstrain(rv, L1)
        lp = logdetjac(rv, vv)
        @fact L1 --> [2.718281828459045 0.0 0.0; 2.0 54.598150033144236 0.0; 3.0 5.0 6.0]
        @fact x1 --> [1.0,2.0,3.0,4.0,5.0,6.0]
        @fact lp --> 11
        #println("Chol factor constraint: ", L1)
        #println("free: ", x1)
        #println("log determinant: ", lp)
        @fact isa(L1, LowerTriangular) --> true
        #vv = randn(5 * (5 + 1) ÷ 2)
        #@fact logdetjac(rv, vv) --> vv[1] + vv[6] + vv[10] + vv[13] + vv[15]
    end

    context("RCholCorr type interface") do
        @fact RCholCorr <: RVType --> true
        @fact RCholCorr <: RMatrix --> true

        @fact ndims(RCholCorr(5)) --> 5
        @fact nfree(RCholCorr(5)) --> 10
        @fact VinDsl.num_pars_advi(RCholCorr(5)) --> 20
        @fact VinDsl.num_pars_advi(RCholCorr(5), true) --> 65

        rv = RCholCorr(4)
        vv = [-0.202733, 0.549306, -0.361359, 0.867301, -0.287743, 1.35484]
        #vv = randn(5 * (5 - 1) ÷ 2)
        L2 = constrain(rv, vv)
        x2 = unconstrain(rv, L2)
        lp = logdetjac(rv, vv)
        @fact L2 --> [1.0 0.0 0.0 0.0;
 -0.20000042810804294 0.9797958097259855 0.0 0.0;
 0.49999989174945103 -0.3000003289343604 0.8124037856200652 0.0;
 0.7000002408759531 -0.2000000289044577 0.6000003991366605 0.33166123114960505]
        #@fact x2 --> [-0.202733, 0.549306, -0.361359, 0.867301, -0.287743, 1.35484]
        @fact lp --> -3.5216454246105022
        println("Chol corr constraint: ", L2)
        println("free: ", x2)
        println("log determinant: ", lp)
        @fact isa(L2, LowerTriangular) --> true
        @fact countnz(abs(L2) .<= 1) --> ndims(rv)^2
        #@fact logdetjac(rv, vv) --> vv[1] + vv[6] + vv[10] + vv[13] + vv[15]
    end

    context("RCorrMat type interface") do
        @fact RCorrMat <: RVType --> true
        @fact RCorrMat <: RMatrix --> true

        @fact ndims(RCorrMat(5)) --> 5
        @fact nfree(RCorrMat(5)) --> 10
        @fact VinDsl.num_pars_advi(RCorrMat(5)) --> 20
        @fact VinDsl.num_pars_advi(RCorrMat(5), true) --> 65

        rv = RCorrMat(4)
        #vv = randn(5 * (5 - 1) ÷ 2)
        vv = [-1., 2., 0., 1., 3., -1.5]
        Lmatrix = constrain(rv, vv)
        println(Lmatrix)
        @fact countnz(abs(Lmatrix) .<= 1) --> ndims(rv)^2
        @fact round(diag(Lmatrix), 8) == ones(ndims(rv)) --> true
        logdetL = logdetjac(rv, vv)
        println(logdetL)
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
