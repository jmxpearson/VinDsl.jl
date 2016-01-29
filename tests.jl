push!(LOAD_PATH, ".")  # needed if VB not a full module
using FactCheck
using VB
using Distributions
using PDMats
import Base: ==

srand(12345)

# use these equality definitions for testing purposes
==(x::PDMat, y::PDMat) = x.mat == y.mat
function =={D <: Distribution}(x::D, y::D)
    all(f -> x.(f) == y.(f), fieldnames(x))
end
function =={D <: Distribution}(x::RandomNode{D}, y::RandomNode{D})
    all(f -> x.(f) == y.(f), fieldnames(x))
end

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

    context("Projection") do
        u = ExprNode(:u, :(x + a), Node[x, a])

        @fact u.outerinds --> [:i, :k]
        @fact u.innerinds --> []
        @fact size(u) --> (p, q)
        @fact project(u, :x, (2, 3)) --> x[2]
        @fact project(u, :a, (2, 3)) --> a[3]
    end

    context("getindex") do
        u = ExprNode(:u, :(x + a), Node[x, a])
        u23 = u[2, 3]

        @fact isa(u23, Distribution) --> true
        @fact isa(u23, ExprDist{Val{:u}}) --> true
        @fact Set(keys(u23.nodedict)) --> Set([:x, :a])
        @fact all(x -> isa(x, Distribution), values(u23.nodedict)) --> true
    end
end

facts("Inferring Factor structure") do

    context("All indices outer") do
        a[i, j] ~ Normal(rand(5, 5), rand(5, 5))
        b[j, k] ~ Gamma(rand(5, 3), rand(5, 3))

        nodes = Node[a, b]
        fi = get_structure(nodes...)

        # order should not be important, so compare Sets
        @fact Set(fi.indices) --> Set([:i, :j, :k])
        @fact Set(zip(fi.indices, fi.maxvals)) --> Set([(:i, 5), (:j, 5), (:k, 3)])

        # check definition of inds_in_factor
        @fact Set(fi.indices[fi.inds_in_factor[:a]]) --> Set([:i, :j])
        @fact Set(fi.indices[fi.inds_in_factor[:b]]) --> Set([:k, :j])

        # check definition of inds_in_node
        @fact Set(a.outerinds[fi.inds_in_node[:a]]) --> Set([:i, :j])
        @fact Set(b.outerinds[fi.inds_in_node[:b]]) --> Set([:j, :k])
    end

    context("Some indices inner") do
        dims = (5, 3)
        m = [rand(dims[1]) for x in 1:dims[2]]
        VV = [diagm(rand(dims[1])) for x in 1:dims[2]]

        d[i, j] ~ MvNormal(m, VV)
        a[i, k] ~ Normal(rand(5, 4), rand(5, 4))

        nodes = Node[a, d]
        fi = get_structure(nodes...)

        # order should not be important, so compare Sets
        @fact Set(fi.indices) --> Set([:j, :k])
        @fact Set(zip(fi.indices, fi.maxvals)) --> Set([(:j, 3), (:k, 4)])

        # check definition of inds_in_factor
        @fact Set(fi.indices[fi.inds_in_factor[:a]]) --> Set([:k])
        @fact Set(fi.indices[fi.inds_in_factor[:d]]) --> Set([:j])

        # check definition of inds_in_node
        @fact Set(a.outerinds[fi.inds_in_node[:a]]) --> Set([:k])
        @fact Set(d.outerinds[fi.inds_in_node[:d]]) --> Set([:j])
    end

    context("Dimension mismatch throws error") do
        a[i, j] ~ Normal(rand(5, 5), rand(5, 5))
        b[j, k] ~ Gamma(rand(4, 3), rand(4, 3))

        nodes = Node[a, b]

        @fact_throws ErrorException get_structure(nodes...)
    end

    context("Inner indices optional") do
        dims = (5, 3)
        m = [rand(dims[1]) for x in 1:dims[2]]
        VV = [diagm(rand(dims[1])) for x in 1:dims[2]]
        LL = rand(3)

        d[j] ~ MvNormal(m, VV)
        L[j] ~ Const(LL)

        nodes = Node[L, d]
        fi = get_structure(nodes...)
        inds = fi.indices
        maxvals = fi.maxvals

        @fact Set(inds) --> Set([:j])
    end

    context("Inner indices arbitrary") do
        dims = (5, 3)
        m = [rand(dims[1]) for x in 1:dims[2]]
        VV = [diagm(rand(dims[1])) for x in 1:dims[2]]
        LL = rand(5, 5, 3)

        d[a, j] ~ MvNormal(m, VV)
        L[a, a, j] ~ Const(LL)

        nodes = Node[L, d]
        fi = get_structure(nodes...)
        inds = fi.indices
        maxvals = fi.maxvals

        @fact Set(inds) --> Set([:j])
    end

end

facts("Basic factor construction") do

    context("Simple univariate nodes") do
        dims = (10, 2)

        a[j] ~ Normal(rand(dims[2]), ones(dims[2]))
        b ~ Gamma(1, 1)
        y[i, j] ~ Normal(rand(dims), ones(dims))

        f = @factor LogNormalFactor y a b;
        inds = f.inds.indices
        maxvals = f.inds.maxvals

        @fact Set(inds) --> Set([:i, :j, :scalar])

        # internals use names defined for factor type, not node names
        @fact project_inds(f, :x, inds) --> [:i, :j]
        @fact project_inds(f, :μ, inds) --> [:j]
        @fact project_inds(f, :τ, inds) --> [:scalar]
        @fact isa(project(f, :x, maxvals), Normal) --> true
        @fact isa(project(f, :μ, maxvals), Normal) --> true
        @fact isa(project(f, :τ, maxvals), Gamma) --> true
        @fact value(f) --> isfinite
    end

    context("Multivariate nodes in factor") do
        dims = (5, 3)
        m = [rand(dims[1]) for x in 1:dims[2]]
        VV = [diagm(rand(dims[1])) for x in 1:dims[2]]

        y[i, j] ~ MvNormal(m, VV)
        μ[i] ~ Normal(zeros(dims[1]), ones(dims[1]))
        Λ[j] ~ Const([diagm(rand(dims[1])) for x in 1:dims[2]])

        f = @factor LogMvNormalCanonFactor y μ Λ
        inds = f.inds.indices
        maxvals = f.inds.maxvals

        @fact Set(inds) --> Set([:j])
        @fact project_inds(f, :x, inds) --> [:j]
        @fact project_inds(f, :μ, inds) --> [Colon()]
        @fact project_inds(f, :Λ, inds) --> [:j]
        @fact isa(project(f, :x, maxvals), MvNormal) --> true
        @fact isa(project(f, :μ, maxvals), Vector{Normal}) --> true
        @fact isa(project(f, :Λ, maxvals), Matrix{Float64}) --> true
        @fact value(f) --> isfinite
    end

end

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
    τ_prior = @factor LogGammaFactor τ a0 b0

    m = VBModel([μ, τ, μ0, τ0, a0, b0, y], [obs, μ_prior, τ_prior])

    @fact Set([n.name for n in m.nodes]) --> Set([:μ, :τ, :μ0, :τ0, :a0, :b0, :y])
    @fact Set([typeof(f) for f in m.factors]) --> Set([VB.LogNormalFactor{2}, VB.LogNormalFactor{1}, VB.LogGammaFactor{1}])
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
        @fact map(size, naturals(f, Λ)[1]) --> ((d, d), ())
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
        @fact map(size, naturals(f, Λ)[1]) --> ((d, d), ())
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

facts("Unrolling and rerolling parameters") do
    context("Distributions") do
        dists = [Gamma(1.1, 1.), Normal(),
                 MvNormal(ones(5), eye(5)),
                 Wishart(5, eye(2)),]

        for d in dists
            par_sizes = get_par_sizes(d)
            npars = mapreduce(prod, +, par_sizes)
            x = unroll_pars(d)
            dd = reroll_pars(d, par_sizes, x)
            @fact isa(x, Vector) --> true
            @fact length(x) --> npars
            @fact dd --> d
        end
    end
    context("Nodes") do
        p, d = 5, 7
        x[i] ~ Gamma(1.1 * ones(p), ones(p))
        y[i] ~ MvNormalCanon([ones(d) for p in 1:p], [eye(d) for p in 1:p])

        node_list = [x, y]

        for n in node_list
            # get parameter sizes for single distribution
            par_sizes = get_par_sizes(n.data[1])
            npars = mapreduce(prod, +, par_sizes)

            # make a copy of the original node
            n_bak = deepcopy(n)

            # unroll parameters
            v = unroll_pars(n)
            @fact length(v) --> prod(size(n)) * npars
            @fact isa(v, Vector) --> true
            update_pars!(n, v)
            @fact n --> n_bak
        end
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
    τ_prior = @factor LogGammaFactor τ a0 b0

    m = VBModel([μ, τ, μ0, τ0, a0, b0, y], [obs, μ_prior, τ_prior])

    context("L-BFGS") do
        update!(μ, m, Val{:l_bfgs})
    end
end

facts("E calculus") do
    context("Get symbols in expression") do
        @fact get_all_syms(5) --> Set{Symbol}([])
        @fact get_all_syms(:x) --> Set([:x])
        @fact get_all_syms(:(x + 1)) --> Set([:x])
        @fact get_all_syms(:(x + y)) --> Set([:x, :y])
        @fact get_all_syms(:(x + (y * x))) --> Set([:x, :y])
        @fact get_all_syms(:(2x + (y * x))) --> Set([:x, :y])
    end

    context("Basic identities") do
        @fact _expandE(1) --> 1
        @fact _expandE(ones(5)) --> ones(5)
        @fact _expandE(:x) --> :(x)
        @fact _expandE(:(E(x))) --> :(E(x))
        @fact _expandE(:(x + y)) --> :(x + y)

        @fact _expand_wrapE(1) --> 1
        @fact _expand_wrapE(ones(5)) --> ones(5)
        @fact _expand_wrapE(:x) --> :(E(x))
        @fact _expand_wrapE(:(E(x))) --> :(E(x))
        @fact _expand_wrapE(:(x + y)) --> :(E(x) + E(y))
    end

    context("+ and -") do
        @fact _expandE(:(E(x + y))) --> :(E(x) + E(y))
        @fact _expandE(:(E(x + y + z))) --> :(E(x) + E(y) + E(z))
        @fact _expandE(:(E(x - y))) --> :(E(x) - E(y))
        @fact _expandE(:(E(x - y + z))) --> :(E(x) - E(y) + E(z))
        @fact _expandE(:(E(x .+ y))) --> :(E(x) .+ E(y))
    end

    context("*") do
        @fact _expandE(:(E(2x))) --> :(2 * E(x))
        @fact _expandE(:(E(2x * 3))) --> :((2 * E(x)) * 3)
        @fact _expandE(:(E(2x * y))) --> :((2 * E(x)) * E(y))
        @fact _expandE(:(E(2 * x * y))) --> :(2 * E(x) * E(y))
        @fact _expandE(:(E(2 * x * y * x))) --> :(2 * E(x * y * x))
        @fact _expandE(:(E(2 * x * y * x * z))) --> :(2 * E(x * y * x) * E(z))
        @fact _expandE(:(E((x * y) * x))) --> :(E((x * y) * x))
        @fact _expandE(:(E((x * y) * x * (w * z)))) --> :(E((x * y) * x) * (E(w) * E(z)))
    end

    context("macro expansion") do
        x ~ Normal(rand(), rand())
        y ~ Normal(rand(), rand())

        xy = @expandE E(x.data[1] + y.data[1])
        @fact xy --> E(x.data[1]) + E(y.data[1])

        xy = @expandE E(x.data[1] * y.data[1] + 5)
        @fact xy --> E(x.data[1]) * E(y.data[1]) + 5
    end
end

facts("Gamma-Poisson model") do
    U = 10  # units
    T = 50  # time points
    K = 3  # HMM factors

    A_0[k] ~ Const([[0.95 0.03 ; 0.05 0.97] for k in 1:K])
    π0_0[k] ~ Const([[0.5 0.5] for k in 1:K])


end
