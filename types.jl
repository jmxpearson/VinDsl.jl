# let's define some types
#################### Node #######################

abstract Node

immutable RandomNode{D <: Distribution} <: Node
    name::Symbol
    indices::Vector{Symbol}
    data::Array{D}

    function RandomNode(name, indices, data)
        @assert ndims(data) == length(indices) "Indices do not match data shape."
        new(name, indices, data)
    end
end
RandomNode{D <: Distribution}(name::Symbol, indices::Vector{Symbol}, ::Type{D}, pars...) = RandomNode{D}(name, indices, map(D, pars...))

immutable ConstantNode{T <: Number} <: Node
    name::Symbol
    indices::Vector{Symbol}
    data::Array{T}
end
ConstantNode{T <: Number}(data::Array{T}, indices::Vector{Symbol}) = 
    ConstantNode(gensym("const"), indices, data)
ConstantNode(x::Number) = ConstantNode(gensym("const"), [:scalar], [x])


#################### Factor #######################
"Defines a factor, a term in the variational objective."
abstract Factor{N}

immutable FactorInds
    indices::Vector{Symbol}
    ranges::Vector{Int}
    indexmap::Dict{Symbol, Vector{Int}}
end


macro factor(ftype, nodes...)
    local ex = quote
        fi = get_structure($(nodes...))
        $ftype{length(fi.indices)}(fi, $(nodes...))
    end
    esc(ex)
end


###################################################
# Functions to deal with factor structure
###################################################
"""
Given a list of nodes, return a FactorInds type variable that calculates
the unique indices in the nodes, their ranges, and the map from node symbols
to integer indices.
"""
function get_structure(nodes...)
    # first, get all unique indices
    allinds = union([n.indices for n in nodes]...)

    # now, map these indices to consecutive integers
    idx_to_int = [idx => idxnum for (idxnum, idx) in enumerate(allinds)]

    # initialize Dicts that will hold maps from index symbols to their 
    # sizes in various nodes (for checking) and from node names to 
    # the (integer) indices they contain
    idxdict = Dict{Symbol, Vector{Int}}([(idx => []) for idx in allinds])
    node_to_int_inds = Dict{Symbol, Vector{Int}}([(n.name => []) for n in nodes])

    # now loop over nodes, building these dicts
    for n in nodes
        for (d, idx) in enumerate(n.indices)
            push!(idxdict[idx], size(n.data, d))
            push!(node_to_int_inds[n.name], idx_to_int[idx])
        end
    end

    # lastly, check that the index ranges are the same for every node in 
    # which an index appears; build a vector of index ranges
    idxsizes = Integer[]
    for (idx, lengths) in idxdict
        if !all(x -> x == lengths[1], lengths)
            error("Index length mismatch in index $idx.")
        else
            push!(idxsizes, lengths[1])
        end
    end

    FactorInds(allinds, idxsizes, node_to_int_inds)
end

"""
Given a factor, a symbol naming a node in that factor, and a tuple of 
index ranges for the factor as a whole, return the elements of node
corresponding to the global range of indices.
"""
function project(f::Factor, name::Symbol, rangetuple)
    node = getfield(f, name)
    node_inds = f.inds.indexmap[name]
    node.data[rangetuple[node_inds]...]
end


_wrapvars(vars, x, y) = x

function _wrapvars(vars::Vector{Symbol}, ex::Expr, indtup)
    # copy AST
    new_ex = copy(ex)

    # recursively wrap variables
    for i in eachindex(ex.args)
        new_ex.args[i] = _wrapvars(vars, ex.args[i], indtup)
    end

    # return new expression
    new_ex
end

function _wrapvars(vars::Vector{Symbol}, s::Symbol, indtup)
    # if s is a variable in the approved list of vars
    if s in vars 
        sym = Expr(:quote, s)
        :(project(f, $sym, $indtup))
    else
        s
    end
end


"""
Recursively wrap variables in an expression so that in the new expression,
they are projected down to their index tuples in a factor with variables
given in vars.
"""
macro wrapvars(vars, ex, indtup)
    esc(_wrapvars(vars, ex, indtup))
end

# "Defines a Variational Bayes model."
# type VBModel  
#     # nodes maps symbols to the nodes/groups of nodes associated with them
#     nodes::Dict{Symbol, Node}

#     # all factors in the graph
#     factors::Vector{Factor}

#     # dictionary linking all random variables to a list of tuples
#     # each tuple gives a factor and the name of the random variable
#     # in that factor
#     graph::Dict{Distribution, Vector{Tuple{Factor, Symbol}}}

#     VBModel(nodes, factors) = begin
#         for f in factors
#             register(f)
#         end
#         new(nodes, factors)
#     end
# end

# function getwidth(nodes::Vector{Node})
#     allinds = Set([n.indices for n in nodes])
# end

# # register a factor with its associated nodes in the graph
# function register(f::Factor, m::VBModel) 
#     for var in fieldnames(f)
#         n = getfield(f, var)
#         if isa(n, Distribution)
#             push!(m.graph, n => (f, var))
#         end
#     end
# end

# function check_conjugate(n::Distribution, m::VBModel)
#     is_conj = Bool[method_exists(naturals, Tuple{typeof(f), Type{Val{s}}, typeof(n)}) for (f, s) in m.graph[n]]
#     all(is_conj)
# end

# # define some factors
# type EntropyFactor <: Factor
#     x::Node
# end

immutable LogNormalFactor{N} <: Factor{N}
    inds::FactorInds
    x::Node
    μ::Node  # mean
    τ::Node  # precision
end

# type LogGammaFactor <: Factor
#     x::Union{Node, Float64}
#     α::Union{Node, Float64}  # shape
#     β::Union{Node, Float64}  # rate
# end

# # define an expectation method on Distributions
# "Calculate the expected value of a Node x."
# E(x::Distribution) = mean(x)

# # Define functions for nonrandom nodes.
# # In each case, a specialized method is already defined for distributions.
# E(x) = x
# entropy(x) = zero(x)
# Elog(x) = log(x)
# Eloggamma(x) = lgamma(x)

# "Calculate the contribution of a Factor f to the objective function."
# value(f::LogNormalFactor) = -(1/2) * ((E(f.τ) * ( var(f.x) + var(f.μ) + 
#     (E(f.x) - E(f.μ))^2 ) + log(2π) + Elog(f.τ)))

# value(f::LogGammaFactor) = (E(f.α) - 1) * E(f.x) - E(f.β) * E(f.x) + 
#     E(f.α) * E(f.β) - Eloggamma(f.α)

# value(f::EntropyFactor) = entropy(f.x)


# "Return natural parameters from a Factor f viewed as a distribution for 
# a given symbol. The last parameter is a type check for conjugacy."
# naturals(f::LogNormalFactor, ::Type{Val{:x}}, ::Normal) = begin
#     μ, τ = E(f.μ), E(f.τ)
#     (μ .* τ, -τ/2)
# end
# naturals(f::LogNormalFactor, ::Type{Val{:μ}}, ::Normal) = begin
#     x, τ = E(f.x), E(f.τ)
#     (x .* τ, -τ/2)
# end
# naturals(f::LogNormalFactor, ::Type{Val{:τ}}, ::Gamma) = begin
#     v = var(f.x) + var(f.μ) + (E(f.x) - E(f.μ)).^2
#     (1/2, v/2)
# end
# naturals(f::LogGammaFactor, ::Type{Val{:x}}, ::Gamma) = (E(f.α) - 1, -E(f.β))

# "Update a RandomNode n."
# function update!{D}(n::RandomNode{D}, ::Type{Val{:conjugate}})
#     # get natural parameter vectors for each factor
#     nlist = [naturals(f, Val{s}, n) for (f, s) in n.factormap]

#     # sum all natural parameter vectors
#     # zip converts a list of natural parameter vectors for each factor into 
#     # a list of factors for each element of the natural parameter vector
#     # we then map + over each of these lists
#     totals = map(x -> +(x...), zip(nlist))

#     # update each distribution in the array
#     for idx in eachindex(n.data)
#         natpars = Any[par[idx] for par in totals]
#         n.data[idx] = D(naturals_to_params(natpars, D)...)
#     end
# end

