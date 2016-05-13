# code for handling conjugacy, natural parameters, and conjugate updates

function check_conjugate{D}(n::RandomNode{D}, m::VBModel)
    is_conj = true
    for (f, s) in m.graph[n]
        ttype = Tuple{Type{typeof(f)}, Type{Val{s}}, Type{D.name.primary}}
        if !method_exists(natural_formula, ttype)
            is_conj = false
            break
        end
    end
    is_conj
end

function update!{D}(n::RandomNode{D}, m::VBModel, ::Type{Val{:conjugate}})
    # get natural parameter vectors for each factor
    messages = [naturals(f, n) for (f, _) in m.graph[n]]

    # update each distribution in the array
    for idx in eachindex(n.data)
        # get all messages corresponding to this element
        this_messages = Any[msg[idx] for msg in messages]

        # sum respective tuple elements
        natpars = map(x -> +(x...), this_messages)

        # convert natural parameters to Distributions.jl parameters
        n[idx] = D(naturals_to_params(natpars, D)...)
    end
end

function naturals(f::Factor, n::RandomNode)
    fsym = f.namemap[n.name]
    _naturals(f, Val{fsym}, n)
end

@generated function _naturals{N, S, D}(f::Factor{N}, fsym::Type{Val{S}}, n::RandomNode{D})
    vars = fieldnames(f)

    # get expression corresponding to the natural parameters
    nat_expr = natural_formula(f, Val{S}, D)

    quote
        # init array of natural parameter tuples
        # should have the same dimension as fsym
        # get the type of the naturals by calling the
        # function on the first element of the array
        nats = naturals(n[1])
        η_type = typeof(nats)
        η = Array{η_type}(size(n)...)
        for i in eachindex(η)
            η[i] = map(zero_like, nats)
        end

        @nloops $N i d -> 1:f.inds.maxvals[d] begin
            nats = @wrapvars $vars $nat_expr project f (@ntuple $N i)
            nat_tup = project_inds(S, f, (@ntuple $N i))
            η[nat_tup...] = add_nats(η[nat_tup...], nats)
        end
        η
    end
end

#=
The strategy for combining natural parameters is:
- If η and nats are both tuples, add elementwise
- If η and nats are arrays of the same size, add the tuples at each position elementwise
=#
@inline function add_nats{N}(η::NTuple{N}, nats::NTuple{N})
    map(+, η, nats)
end

@inline function add_nats{T <: NTuple, N}(η::Array{T, N}, nats::Array{T, N})
    map(add_nats, η, nats)
end

"""
Define natural parameters for a given factor type, variable within that
factor, and form of the conjugate distribution for that variable.
"""
macro defnaturals(factortype, varname, disttype, expr)
    nat_expr = Expr(:quote, expr)
    varsym = Expr(:quote, varname)
    ex = quote
        natural_formula{N}(::Type{$(factortype){N}}, ::Type{Val{$varsym}}, ::Type{$disttype}) = $nat_expr
    end
    esc(ex)
end
