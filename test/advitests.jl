
using FactCheck
using Distributions
using PDMats
using StatsFuns

using VinDsl
srand(56778)

facts("Utility functions") do
    p = 10
    n = 6
    d = MvNormal(rand(p))
    x = VinDsl.advi_rand(d)
    @fact eltype(x) --> Float64
    @fact size(x) --> (p,)
    y = VinDsl.advi_rand(d, n)
    @fact size(y) --> (p, n)

    d = Normal(1, 2.)
    x = VinDsl.advi_rand(d)
    @fact size(x) --> ()
    y = VinDsl.advi_rand(d, n)
    @fact size(y) --> (n,)

    x = Array{ForwardDiff.Dual}(rand(10, 7))
    randn!(x)
    @fact eltype(x) <: ForwardDiff.Dual --> true

    p = 5
    vv = rand(p * (p + 3) ÷ 2)
    @fact H(vv, true) --> p * (log2π + 1)/2 + sum(vv[p + [1, 6, 10, 13, 15]])
    @fact H([1.1, 2.2]) --> (log2π + 1)/2 + 2.2/2
end
