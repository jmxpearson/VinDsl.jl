# things that don't easily fit elsewhere
import Base.UpperTriangular

zero_like(A::Array) = zeros(A)
zero_like(x::Number) = zero(x)
lB(x::Vector) = sum(lgamma(x)) - lgamma(sum(x))
lB(x::Matrix) = sum(lgamma(x), 1) - lgamma(sum(x, 1))  # columnwise

# define a convert method for Arrays to PDMats (positive definite matrices)
# if the array is not posdef, this will throw an exception
raw(Σ::PDMat) = Σ.mat
raw(Σ::PDSparseMat) = Σ.mat
raw(Σ::PDiagMat) = Σ.diag
raw(Σ::ScalMat) = diag(Σ)

convert{T<:AbstractPDMat}(::Type{T}, P::T) = P
convert{T <: Number}(::Type{PDMat{T, Matrix{T}}}, P::Matrix{T}) = PDMat(P)
convert{T<:AbstractPDMat, S<:AbstractPDMat}(::Type{T}, P::S) = convert(T, raw(P))

flatten(a::Number) = a
flatten(a::Array) = reshape(a, prod(size(a)))
flatten(a::AbstractPDMat) = flatten(UpperTriangular(full(a)))
function flatten(a::UpperTriangular)
    d = size(a, 1)
    l = div(d * (d + 1), 2)
    out = Array(eltype(a), l)
    idx = 1
    for c in 1:d
        for r in 1:d
            if r ≤ c
                out[idx] = a.data[r, c]
                idx += 1
            end
        end
    end
    out
end

function UpperTriangular(v::AbstractVector)
    l = length(v)
    d = (-1 + Int(sqrt(1 + 8l))) ÷ 2
    A = zeros(eltype(v), d, d)
    idx = 1
    for c in 1:d
        for r in 1:d
            if r ≤ c
                A[r, c] = v[idx]
                idx += 1
            end
        end
    end
    UpperTriangular(A)
end

"""
Generate a (multivariate) Normal distribution from a vector of unconstrained
parameters. Number of required parameters is
p (μ) + p(p + 1)/2 (Σ) = p(p + 3)/2
"""
function normal_from_unconstrained(x::Vector)
    # number of parameters: will throw InexactError if not an integer
    p = Int((-3 + sqrt(9 + 8 * length(x)))/2)
    if p == 1
        d = Normal(x[1], exp(x[2]))
    else
        d = MvNormal(x[1:p], constrain(RCovMat(p), x[p+1:end]))
    end
    d
end
