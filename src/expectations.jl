# code for working with expectations of nodes

# define an expectation method on Distributions
"Calculate the expected value of a Node x."
E(x) = x
E(x::Distribution) = mean(x)
V(x) = zero(x)
V(x::Distribution) = var(x)
C(x) = zero(x)
C(x::Distribution{Univariate}) = V(x)
C(x::AbstractMvNormal) = cov(x)
C(x::HMM) = cov(x)
C{D <: Distribution}(x::Vector{D}) = diagm(map(V, x))
H(x) = zero(x)
H(x::Distribution) = entropy(x)

# Define functions for nonrandom nodes.
# In each case, a specialized method is already defined for distributions.
Elog(x) = log(x)
Einv(x) = inv(x)
Eloggamma(x) = lgamma(x)
Elogmvgamma(p, x) = logmvgamma(p, x)
Elogbeta(x) = lbeta(x)
ElogB(x) = lB(x)
Elogdet(x) = logdet(x)
Elogdet{D <: Distribution{Univariate}}(x::Array{D}) = prod(Elog(x))
Elogdet(x::Distribution{Univariate}) = Elog(x)

# now make version of all these functions that work on nodes
# by working elementwise
macro make_mapped_version(fn)
    esc(:($fn{D <: Distribution}(n::Array{D}) = map($fn, n)))
end

to_map = [E, V, C, H, Einv, Elog, ElogB, Eloggamma, Elogmvgamma, Elogdet]
for fn in to_map
    @make_mapped_version fn
end
