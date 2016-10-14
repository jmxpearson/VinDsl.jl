using Distributions
import Base.size, Base.length, Base.rand, Base.mean, Base.convert

doc"""
    MatrixNormal(M, U, V)
The *Matrix Normal distribution* with mean `M` and row and column covariances `U` and `V` has probability density function
$f(\mathbf{X}; \mathbf{M}, \mathbf{U}, \mathbf{V}) = \frac{\exp\left(
    -\frac{1}{2} \mathrm{tr}\left[
        \mathbf{V}^{-1} (\mathbf{X} - \mathbf{M})^\top \mathbf{U}^{-1}
        (\mathbf{X} - \mathbf{M})
    \right]
    \right)}{(2\pi)^{np/2} |\mathbf{V}|^{n/2} |\mathbf{U}|^{p/2}}$
when $\mathbf{X}$ is $n \times p$.
```julia
MatrixNormal(U, V)      # Matrix Normal with mean 0 and covariances U and V
MatrixNormal(M, U, V)      # Matrix Normal with mean M and covariances U and V
params(d)        # Get the parameters, i.e. (M, U, V)
mean(d)          # Get the mean, i.e., M
cov(d)           # Get the covariance, i.e., V ⊗ U
var(d)           # Get the marginal variances for individual entries
invcov(d)        # inverse of covariance
logdetcov(d)     # log determinant of covariance
logpdf(d, X)     # log of the pdf at point X
```
External links
* [Matrix Normal distribution on Wikipedia](http://en.wikipedia.org/wiki/Matrix_normal_distribution)
"""
immutable MatrixNormal{Mean<:AbstractMatrix, CovU<:AbstractPDMat, CovV<:AbstractPDMat} <: ContinuousMatrixDistribution
    M::Mean
    U::CovU
    V::CovV
end

### Constructors
typealias CovLike Union{AbstractMatrix, AbstractVector, AbstractPDMat}
normalize(σ::AbstractVector) = PDiagMat(abs2(σ))
normalize(Σ::AbstractMatrix) = PDMat(Σ)
normalize(Σ::AbstractPDMat) = Σ
normalize(d::Int, σ::Real) = PDScalMat(d, abs2(σ))

function MatrixNormal{Mean <: AbstractMatrix, CovU<:AbstractPDMat, CovV<:AbstractPDMat}(M::Mean, U::CovU, V::CovV)
    size(M) == (dim(U), dim(V)) || throw(DimensionMismatch("The dimensions of M, U, and V are inconsistent."))
    MatrixNormal{Mean, CovU, CovV}(M, U, V)
end
function MatrixNormal{Mean<:AbstractMatrix, CovU<:CovLike, CovV<:CovLike}(M::Mean, U::CovU, V::CovV)
    MatrixNormal(M, normalize(U), normalize(V))
end

MatrixNormal{CovU<:AbstractPDMat, CovV<:AbstractPDMat}(U::CovU, V::CovV) = MatrixNormal(zeros(dim(U), dim(V)), U, V)
MatrixNormal{CovU<:CovLike, CovV<:CovLike}(U::CovU, V::CovV) = MatrixNormal(normalize(U), normalize(V))

### Conversions
convert(::Type{MvNormal}, d::MatrixNormal) = MvNormal(vec(mean(d)), cov(d))

### interface
size(d::MatrixNormal) = (dim(d.U), dim(d.V))
params(d::MatrixNormal) = (d.M, d.U, d.V)
mean(d::MatrixNormal) = d.M
cov(d::MatrixNormal) = kron(full(d.V), full(d.U))
var(d::MatrixNormal) = diag(d.U) * diag(d.V)'
invcov(d::MatrixNormal) = kron(full(inv(d.V)), full(inv(d.U)))
logdetcov(d::MatrixNormal) = ((n, p) = size(d); n * logdet(d.V) + p * logdet(d.U))

rand(d::MatrixNormal) = reshape(rand(convert(MvNormal, d)), size(d))
entropy(d::MatrixNormal) = entropy(convert(MvNormal, d))

function logpdf{T<:Real}(d::MatrixNormal, X::AbstractMatrix{T})
    size(X) == size(d) || error("Input matrix x has wrong size.")
    m = X - d.M
    n, p = size(d)
    invU = full(inv(d.U))
    invV = full(inv(d.V))
    lpdf = -0.5 * trace(invV * m' * invU * m)
    lpdf -= (n * p/2) * Distributions.log2π
    lpdf -= 0.5 * logdetcov(d)
    lpdf
end
