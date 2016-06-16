import Base.randn!

function advi_rand{T}(d::MvNormal{T})
    x = Array{T}(length(d))
    Distributions.add!(PDMats.unwhiten!(d.Σ, randn!(x)), d.μ)
end
function advi_rand{T}(d::MvNormal{T}, n::Int)
    x = Array{T}(length(d), n)
    Distributions.add!(PDMats.unwhiten!(d.Σ, randn!(x)), d.μ)
end
advi_rand(d::Normal) = d.μ + d.σ * randn()
function advi_rand{T}(d::Normal{T}, n::Int)
    x = Array(T, n)
    randn!(x)
    scale!(d.σ, x)
    x[:] += d.μ
end

function randn!{T<:ForwardDiff.Dual}(A::AbstractArray{T})
    for i in eachindex(A)
        @inbounds A[i] = randn()
    end
    A
end
