# things that don't easily fit elsewhere

zero_like(A::Array) = zeros(A)
zero_like(x::Number) = zero(x)
lB(x::Vector) = sum(lgamma(x)) - lgamma(sum(x))
lB(x::Matrix) = sum(lgamma(x), 1) - lgamma(sum(x, 1))  # columnwise

# define a convert method for Arrays to PDMats (positive definite matrices)
# if the array is not posdef, this will throw an exception
convert{T <: Number}(::Type{PDMat{T, Array{T, 2}}}, arr::Array{T, 2}) = PDMat(arr)

flatten(a::Number) = a
flatten(a::Array) = reshape(a, prod(size(a)))
flatten(a::AbstractPDMat) = flatten(full(a))
