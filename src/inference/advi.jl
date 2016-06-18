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

function advi_rand{T<:Real}(x::Vector{T}, full=false)
    if full
        p = Int((-3 + sqrt(9 + 8 * length(x)))/2)
        L = LowerTriangular(x[p+1:endof(x)])
        out = L * randn(p)
    else
        p = Int(length(x)/2)
        out = exp(x[p+1:endof(x)]) .* randn(p)
    end
    for i in 1:p
        @inbounds out[i] += x[i]
    end
    out
end

function advi_rand{T<:Real}(x::Vector{T}, n::Int, full=false)
    if full
        p = Int((-3 + sqrt(9 + 8 * length(x)))/2)
        L = LowerTriangular(x[p+1:endof(x)])
        out = L * randn(p, n)
    else
        p = Int(length(x)/2)
        out = scale(exp(x[p+1:endof(x)]), randn(p, n))
    end
    for i in 1:p
        @inbounds out[i] += x[i]
    end
    out
end

function randn!{T<:ForwardDiff.Dual}(A::AbstractArray{T})
    for i in eachindex(A)
        @inbounds A[i] = randn()
    end
    A
end

"""
Calculate the entropy of a (multivariate) Normal distribution based on
a vector of unconstrained parameters for the mean and covariance.
"""
function H{T<:Real}(x::Vector{T}, full=false)
    if full
        # number of parameters: will throw InexactError if not an integer
        p = Int((-3 + sqrt(9 + 8 * length(x)))/2)

        # ½ logdet(Σ) = ∑ log L_ii, with L the Cholesky factor
        # but log L_ii is just the unconstrained parameter
        offset = p + 1  # first element of covariance part
        ldet = 0.
        skip = p
        while offset ≤ length(x)
            ldet += x[offset]
            offset += skip
            skip -= 1
        end
    else
        # return a multivariate normal with diagonal covariance
        p = Int(length(x)/2)
        ldet = sum(x[p+1:end])/2  # ½ ∑ log(σ)
    end
    p * (log2π + 1)/2 + ldet
end

"""
Initialize unconstrained and constrained arrays of advi variables.
x is the vector of unconstrained parameters being optimized over
offset is the initial offset within x to begin reading from
rv is the random variable (isa(T, RVType) == true)
ζ is the array of (sampled) unconstrained variables
v is the array of (sampled) constrained variables
Returns a tuple containing the total ELBO contribution due to
    entropy plus Jacobians and the updated offset
"""
function advi_initialize!(x, offset, rv, ζ, v)
    L = zero(eltype(x))
    npars = VinDsl.num_pars_advi(rv)

    ctr = offset
    for i in eachindex(v)
        L += H(x[ctr:ctr+npars-1])
        if isa(rv, RScalar)
            ζ[i] = VinDsl.advi_rand(x[ctr:ctr+npars-1])[1]
        else
            ζ[i] = VinDsl.advi_rand(x[ctr:ctr+npars-1])
        end
        v[i] = constrain(rv, ζ[i])
        L += logdetjac(rv, ζ[i])
        ctr += npars
    end
    (L, ctr)
end

"""
Add an ADVI random variable to the model. Add contributions to the ELBO
for entropy and the log determinant of the Jacobian for the transformation from
unconstrained to constrained variables. Return the ELBO contribution, an
offset in the parameter vector marking the next unused parameter, and
a sample value of the constrained variable.
"""
function advi_variable(x, offset, rv, dims)
    C = VinDsl.storage_type(rv, eltype(x))
    ζ = Array{C}(dims...)
    v = Array{C}(dims...)

    L, new_offset = advi_initialize!(x, offset, rv, ζ, v)

    (L, new_offset, v)
end
function advi_variable(x, offset, rv)
    L = zero(eltype(x))
    npars = VinDsl.num_pars_advi(rv)

    L += H(x[offset:offset+npars-1])
    if isa(rv, RScalar)
        ζ = VinDsl.advi_rand(x[offset:offset+npars-1])[1]
    else
        ζ = VinDsl.advi_rand(x[offset:offset+npars-1])
    end
    v = constrain(rv, ζ)
    L += logdetjac(rv, ζ)

    (L, offset + npars, v)
end

macro advi_declarations(x)
    esc(_advi_declarations(x))
end

function _advi_declarations(ex::Expr, vars)
    if ex.head == :(::)
        vname = ex.args[1]
        typearg = ex.args[2]
        if isa(typearg, Expr)
            if typearg.head == :ref   # array of variables
                T = _convert_typename(typearg.args[1])
                dims = typearg.args[2:end]
                d_expr = :(
                begin
                    ΔL, ctr, $vname = VinDsl.advi_variable(x, ctr, $T, tuple($(dims...)))
                    L += ΔL
                end
                    )
            else  # single variable
                T = _convert_typename(typearg)
                d_expr = :(
                begin
                    ΔL, ctr, $vname = VinDsl.advi_variable(x, ctr, $T)
                    L += ΔL
                end
                    )
            end
            push!(vars, d_expr)
        end

    else
        for a in filter(x -> isa(x, Expr), ex.args)
            _advi_declarations(a, vars)
        end
    end
    Expr(:block, vars...)
end

_advi_declarations(ex::Expr) = _advi_declarations(ex, [])

function _convert_typename(ex)
    out = copy(ex)
    out.args[1] = Symbol("R", out.args[1])
    out
end

macro advi_model(x)
    esc(_advi_model(x))
end

function _advi_model(ex::Expr)
    # if we have a ~ expression, replace with increment of L
    if ex.head == :macrocall && ex.args[1] == Symbol("@~")
        out = :(L += logpdf($(ex.args[3]), $(ex.args[2])))
    else  # recursively parse
        out = copy(ex)
        for i in eachindex(out.args)
            out.args[i] = isa(out.args[i], Expr) ? _advi_model(out.args[i]) : out.args[i]
        end
    end
    out
end
