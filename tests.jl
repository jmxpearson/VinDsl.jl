push!(LOAD_PATH, ".")  # needed if VB not a full module
using FactCheck
using VB
using Distributions

srand(12345)

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
        c[i, j] ~ Const(rand(4, 4))

        @fact isa(a, RandomNode) --> true
        @fact isa(c, ConstantNode) --> true
    end

    context("Error handling for indices") do
        @fact_throws AssertionError a[i] ~ Normal(rand(3, 3), rand(3, 3))
        @fact_throws AssertionError a[i, j, k] ~ Normal(rand(3, 3), rand(3, 3))
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
    end
end
