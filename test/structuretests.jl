

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

    context("Some inner indices forced") do
        a[i, j] ~ Normal(rand(5, 5), rand(5, 5))
        b[j, k] ~ Gamma(rand(5, 3), rand(5, 3))

        nodes = Node[a, b]
        fi = get_structure([:j], nodes...)

        # order should not be important, so compare Sets
        # index j should be missing
        @fact Set(fi.indices) --> Set([:i, :k])
    end

end

facts("Basic factor construction") do

    context("Simple univariate nodes") do
        dims = (10, 2)

        a[j] ~ Normal(rand(dims[2]), ones(dims[2]))
        b ~ Gamma(1, 1)
        y[i, j] ~ Normal(rand(dims), ones(dims))

        f = @factor LogNormalFactor y a b
        inds = f.inds.indices
        maxvals = f.inds.maxvals

        @fact Set(inds) --> Set([:i, :j, :scalar])

        # internals use names defined for factor type, not node names
        @fact project_inds(:x, f, inds) --> [:i, :j]
        @fact project_inds(:μ, f, inds) --> [:j]
        @fact project_inds(:τ, f, inds) --> [:scalar]
        @fact isa(project(:x, f, maxvals), Normal) --> true
        @fact isa(project(:μ, f, maxvals), Normal) --> true
        @fact isa(project(:τ, f, maxvals), Gamma) --> true
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
        @fact project_inds(:x, f, inds) --> [:j]
        @fact project_inds(:μ, f, inds) --> [Colon()]
        @fact project_inds(:Λ, f, inds) --> [:j]
        @fact isa(project(:x, f, maxvals), MvNormal) --> true
        @fact isa(project(:μ, f, maxvals), Vector{Normal}) --> true
        @fact isa(project(:Λ, f, maxvals), Matrix{Float64}) --> true
        @fact value(f) --> isfinite
    end

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
