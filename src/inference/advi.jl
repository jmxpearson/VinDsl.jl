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

macro advi_declarations(adviparamlist)
    esc(_advi_declarations(adviparamlist))
end

function _advi_declarations(adviexpr::Expr, paramlist)
    if adviexpr.head == :(::)
        advivname = adviexpr.args[1]
        advitypearg = adviexpr.args[2]
        if isa(advitypearg, Expr)
            if advitypearg.head == :ref   # array of variables
                advitype = _convert_typename(advitypearg.args[1])
                paramdims = advitypearg.args[2:end]
                dimsexpr = :(
                begin
                    ΔLoweradvi, adviparamctr, $advivname = VinDsl.advi_variable(adviparamlist, adviparamctr, $advitype, tuple($(paramdims...)))
                    Loweradvi += ΔLoweradvi
                end
                    )
            else  # single variable
                advitype = _convert_typename(advitypearg)
                dimsexpr = :(
                begin
                    ΔLoweradvi, adviparamctr, $advivname = VinDsl.advi_variable(adviparamlist, adviparamctr, $advitype)
                    Loweradvi += ΔLoweradvi
                end
                    )
            end
            push!(param_list, dimsexpr)
        end

    else
        for aadiviex in filter(adviparamlist -> isa(adviparamlist, Expr), adviexpr.args)
            _advi_declarations(aadiviex, paramlist)
        end
    end
    Expr(:block, paramlist...)
end

_advi_declarations(adviexpr::Expr) = _advi_declarations(adviexpr, [])

function _convert_typename(adviexpr)
    adviexprout = copy(adviexpr)
    adviexprout.args[1] = Symbol("R", adviexprout.args[1])
    adviexprout
end

macro advi_model(adviparamlist)
    esc(_advi_model(adviparamlist))
end

function _advi_model(adviexpr::Expr)
    # if we have a ~ expression, replace with increment of L
    if adviexpr.head == :macrocall && adviexpr.args[1] == Symbol("@~")
        adviexprout = :(Loweradvi += logpdf($(adviexpr.args[3]), $(adviexpr.args[2])))
    else  # recursively parse
        adviexprout = copy(adviexpr)
        for indxadiviex in eachindex(adviexprout.args)
            adviexprout.args[indxadiviex] = isa(adviexprout.args[i], Expr) ? _advi_model(adviexprout.args[indxadiviex]) : adviexprout.args[indxadiviex]
        end
    end
    adviexprout
end

macro ELBO(adviexpr)
    adviexprout = quote
        function ELBO(adviparamlist::Vector)
            adviparamctr = 1
            Loweradvi = zero(eltype(adviparamlist))
            $adviexpr
            Loweradvi
        end
    end

    esc(adviexprout)
end
