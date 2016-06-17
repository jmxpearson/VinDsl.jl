using FactCheck
using Distributions
using PDMats

using VinDsl
srand(12357)

facts("Flattening matrices") do
    context("Basic flattens") do
        @fact flatten(5) --> 5
        @fact flatten([[1, 2, 3] [4, 5, 6]]) --> [1, 2, 3, 4, 5, 6]
    end
    context("Triangulars") do
        A = rand(5, 5)
        U = UpperTriangular(A)
        L = LowerTriangular(A)
        @fact UpperTriangular(flatten(U)) --> U
        @fact LowerTriangular(flatten(L)) --> L
    end
    context("PDMats") do
        A = Float64[[10, 2, 3] [2, 40, 5] [3, 5, 60]]
        P = PDMat(A)
        @fact flatten(P) --> Float64[10, 2, 40, 3, 5, 60]
    end
end

facts("Checking normal_from_unconstrained") do
    context("Univariate") do
        x = randn(2)
        d = VinDsl.normal_from_unconstrained(x)
        @fact isa(d, Normal) --> true
        @fact d.μ --> x[1]
        @fact d.σ --> roughly(exp(x[2]))
    end
    context("Multivariate: mean field") do
        p = 7
        l = 2p
        x = randn(l)
        d = VinDsl.mvnormal_from_unconstrained(x)
        @fact isa(d, MvNormal) --> true
        @fact d.μ --> x[1:p]
        @fact d.Σ.diag --> roughly(exp(2 .* x[p+1:end]))
    end
    context("Multivariate: full covariance") do
        p = 7
        l = p * (p + 3) ÷ 2
        x = randn(l)
        d = VinDsl.mvnormal_from_unconstrained(x, true)
        @fact isa(d, MvNormal) --> true
        @fact d.μ --> x[1:p]
        @fact unconstrain(RCovMat(p), d.Σ) --> roughly(x[p+1:end])
    end
end
