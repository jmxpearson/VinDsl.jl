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

# facts("Inferring Factor structure") do
#     μ = rand(5, 5)
#     σ = rand(5, 5)
#     aa = RandomNode(:a, [:i, :j], Normal, μ, σ)
#     bb = RandomNode(:b, [:j, :k], Gamma, rand(5, 3), rand(5, 3))
# end
