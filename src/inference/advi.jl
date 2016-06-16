function advi_rand(d::MvNormal, n::Int=1)
    T = Array{partype(d)}
    x = n == 1 ? T(randn(length(d))) : T(randn(length(d), n))
    Distributions.add!(PDMats.unwhiten!(d.Σ, x), d.μ)
end

function advi_rand(d::Normal, n::Int=1)
    T = partype(d)
    x = n == 1 ? T(randn()) : Array{T}(randn(n))
    if n == 1
        x *= d.σ
        x += d.μ
    else
        scale!(d.σ, x)
        x[:] += d.μ
    end
    x
end
