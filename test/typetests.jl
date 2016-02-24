
facts("Can create basic node types using constructors.") do

    context("Constant nodes: explicit constructor call") do
        cc = ConstantNode(:c, [:i, :j], rand(3, 4))

        @fact isa(cc, ConstantNode) --> true
        @fact cc.outerinds --> [:i, :j]
        @fact cc.innerinds --> []
        @fact size(cc.data) --> (3, 4)
    end

    context("Constant nodes: no name provided") do
        cc = ConstantNode(rand(3, 4), [:i, :j])

        @fact isa(cc, ConstantNode) --> true
        @fact cc.outerinds --> [:i, :j]
        @fact cc.innerinds --> []
        @fact size(cc.data) --> (3, 4)
    end

    @fact_throws AssertionError ConstantNode(rand(3, 4), [:i])

    context("Random nodes: explicit constructor calls") do
        aa = RandomNode(:a, [:i, :j], Normal, rand(3, 3), rand(3, 3))

        @fact isa(aa, RandomNode) --> true
    end

    context("Tilde node definitions") do
        a[i, j] ~ Normal(rand(4, 4), rand(4, 4))
        b ~ Normal(0, 1)
        c[i, j] ~ Const(rand(4, 4))
        d ~ Const(5.)

        @fact isa(a, RandomNode) --> true
        @fact isa(b, RandomNode) --> true
        @fact isa(c, ConstantNode) --> true
    end

    context("Error handling for indices") do
        @fact_throws AssertionError a[i] ~ Normal(rand(3, 3), rand(3, 3))
        @fact_throws AssertionError a[i, j, k] ~ Normal(rand(3, 3), rand(3, 3))
    end

    #=
    TODO: check for duplicate indices
    make sure duplicate inners have same size
    duplicate inner/outers should throw error
    =#

    context("Multivariate nodes with no outer indices") do
        d = 5
        x ~ MvNormal(zeros(d), diagm(ones(d)))
        @fact size(x.data) --> (1,)

        z[i] ~ MvNormal(zeros(d), diagm(ones(d)))
        @fact size(z.data) --> (1,)

        w ~ Wishart(float(d), diagm(rand(d)))
        @fact size(w.data) --> (1,)

        y[p, q] ~ Wishart(float(d), diagm(rand(d)))
        @fact size(w.data) --> (1,)
    end

    context("Multivariate nodes") do
        dims = (5, 3)
        m = [rand(dims[1]) for x in 1:dims[2]]
        VV = [diagm(rand(dims[1])) for x in 1:dims[2]]
        d[i, j] ~ MvNormal(m, VV)

        @fact isa(d, RandomNode) --> true
        @fact d.innerinds --> [:i]
        @fact d.outerinds --> [:j]
    end

end

facts("Expression nodes") do
    p = 5
    q = 8
    x[i] ~ Normal(rand(p), ones(p))
    y[i] ~ Gamma(1.1 * ones(p), ones(p))
    m[j] ~ MvNormal(ones(q), eye(q))
    a[k] ~ Normal(rand(q), ones(q))
    c ~ Const(rand(5, 5))
    d[k] ~ Dirichlet(q, 1)

    context("Outer constructor") do
        ex = :(x + 5 * y)
        z = ExprNode(:z, ex, Node[x, y])
        w = ExprNode(:w, :(x + m * y), Node[x, y, m])
        v = ExprNode(:v, :(c * d), Node[c, d])

        @fact isa(z, Node) --> true
        @fact isa(z, ExprNode) --> true
        @fact z.name --> :z
        @fact z.outerinds --> [:i]
        @fact z.innerinds --> []
        @fact z.dims --> [p]
        @fact size(z) --> (p,)

        @fact w.innerinds --> [:j]
        @fact w.outerinds --> [:i]
        @fact size(w) --> (p,)

        @fact v.outerinds --> [:scalar]
        @fact v.innerinds --> [:k]
    end

    context("Macro constructor") do
        @exprnode z (x + 5y * m)
        @fact isa(z, Node) --> true
        @fact isa(z, ExprNode) --> true
        @fact z.name --> :z
        @fact z.outerinds --> [:i]
        @fact z.innerinds --> [:j]
        @fact z.dims --> [p]
        @fact size(z) --> (p,)
    end

    context("RHS indices made internal") do
        @exprnode z (x + 5y * a[k])
        @fact isa(z, Node) --> true
        @fact isa(z, ExprNode) --> true
        @fact z.name --> :z
        @fact z.outerinds --> [:i]
        @fact z.innerinds --> [:k]
        @fact z.dims --> [p]
        @fact size(z) --> (p,)
    end

    context("Projection") do
        u = ExprNode(:u, :(x + a), Node[x, a])

        @fact u.outerinds --> [:i, :k]
        @fact u.innerinds --> []
        @fact size(u) --> (p, q)
        @fact project(:x, u, (2, 3)) --> x[2]
        @fact project(:a, u, (2, 3)) --> a[3]
    end

    context("getindex") do
        u = ExprNode(:u, :(x + a), Node[x, a])
        u23 = u[2, 3]

        @fact isa(u23, Distribution) --> true
        @fact isa(u23, ExprDist{Val{:u}}) --> true
        @fact Set(keys(u23.nodedict)) --> Set([:x, :a])
        @fact all(x -> isa(x, Distribution), values(u23.nodedict)) --> true
    end

    context("Expectation") do
        @exprnode u (x + 3a)

        @fact E(u[2, 3]) --> E(x[2]) + 3 * E(a[3])
    end

    context("Factors involving ExprNode") do
        v[i, k] ~ Const(rand(p, q))
        w[i] ~ Const(rand(p))
        @exprnode z (x + 3a)

        f = @factor LogGammaFactor v w z

        @fact Set(f.inds.indices) --> Set([:i, :k])
        @fact value(f) --> isfinite
    end
end
