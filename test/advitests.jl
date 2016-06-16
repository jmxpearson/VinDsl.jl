
using FactCheck
using Distributions
using PDMats

using VinDsl
srand(56778)

facts("Utility functions") do
    p = 10
    n = 6
    d = MvNormal(Array{BigFloat}(rand(p)))
    x = VinDsl.advi_rand(d)
    @fact eltype(x) --> BigFloat
    @fact size(x) --> (p,)
    y = VinDsl.advi_rand(d, n)
    @fact size(y) --> (p, n)

    d = Normal(1, 2.)
    x = VinDsl.advi_rand(d)
    @fact size(x) --> ()
    y = VinDsl.advi_rand(d, n)
    @fact size(y) --> (n,)

end
