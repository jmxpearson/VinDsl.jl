using FactCheck
using Distributions
using PDMats

using VinDsl
srand(12357)

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
        d = VinDsl.normal_from_unconstrained(x)
        @fact isa(d, MvNormal) --> true
        @fact d.μ --> x[1:p]
        @fact d.Σ.diag --> roughly(exp(2 .* x[p+1:end]))
    end
    context("Multivariate: full covariance") do
        p = 7
        l = p * (p + 3) ÷ 2
        x = randn(l)
        d = VinDsl.normal_from_unconstrained(x, true)
        @fact isa(d, MvNormal) --> true
        @fact d.μ --> x[1:p]
        @fact unconstrain(RCovMat(p), d.Σ) --> roughly(x[p+1:end])
    end
end
