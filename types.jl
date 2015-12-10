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

    function ConstantNode(name, indices, data)
        @assert ndims(data) == length(indices) "Indices do not match data shape."
        new(name, indices, data)
    end
end
ConstantNode{T <: Number}(name::Symbol, indices::Vector{Symbol}, data::Array{T}) = ConstantNode{T}(name, indices, data)
ConstantNode{T <: Number}(data::Array{T}, indices::Vector{Symbol}) = 
    ConstantNode(gensym("const"), indices, data)
ConstantNode(x::Number) = ConstantNode(gensym("const"), [:scalar], [x])

"""
Create a node using formula syntax. E.g.
x[i, j, k] ~ Normal(μ, σ)
z ~ Gamma(a, b)
y[a, b] ~ Const(Y)  (for a constant node)
"""
macro ~(varex, distex)
    if isa(varex, Symbol)
        name = varex
        inds = :([:scalar])
    elseif varex.head == :ref
        name = varex.args[1]
        inds = Symbol[varex.args[2:end]...]        
    end
    qname = Expr(:quote, name)
    
    if distex.head == :call
        if distex.args[1] == :Const
            constr = :ConstantNode
            out = :($name = $constr($qname, $inds, $(distex.args[2])))
        else
            constr = :RandomNode
            dist = distex.args[1]
            distargs = distex.args[2:end]
            out = :($name = $constr($qname, $inds, $dist, $(distargs...)))
        end
    end
    esc(out)
end

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
        symdict = get_name_mapping($ftype, $(nodes...))
        $ftype{length(fi.indices)}($(nodes...), fi, symdict)
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

function get_node_size(f::Factor, n::Node)
    fi = f.inds
    syminds = fi.indexmap[n.name]
    fi.ranges[syminds]
end

function get_name_mapping{F <: Factor}(ftype::Type{F}, nodes...)
    nodenames = [n.name for n in nodes]
    Dict(zip(nodenames, fieldnames(ftype)))
end

"""
Given a factor, a symbol naming a node in that factor, and a tuple of 
index ranges for the factor as a whole, return the elements of node
corresponding to the global range of indices.
"""
function project(f::Factor, name::Symbol, rangetuple)
    node = getfield(f, name)
    node_inds = f.inds.indexmap[node.name]
    node.data[rangetuple[node_inds]...]
end

function project_inds(f::Factor, name::Symbol, rangetuple)
    node = getfield(f, name)
    node_inds = f.inds.indexmap[node.name]
    rangetuple[node_inds]
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

"""
Calculate the value of a factor. Relies on value methods taking factor 
types as arguments.
"""
@generated function value{N}(f::Factor{N})
    vars = fieldnames(f)
    val_expr = value(f)
    quote
        v = 0
        @nloops $N i d -> 1:f.inds.ranges[d] begin
            v += @wrapvars $vars $val_expr (@ntuple $N i)
        end
        v
    end
end

#################### Model #######################
"Defines a Variational Bayes model."
type VBModel  
    # nodes maps symbols to the nodes/groups of nodes associated with them
    nodes::Vector{Node}

    # all factors in the graph
    factors::Vector{Factor}

    # dictionary linking all random variables to a list of tuples;
    # each tuple gives a factor and the name of the random variable
    # in that factor
    graph::Dict{Node, Vector{Tuple{Factor, Symbol}}}

    VBModel(nodes, factors) = begin
        m = new(nodes, factors)

        # build an empty graph
        m.graph = Dict{Node, Vector{Tuple{Factor, Symbol}}}()
        for n in nodes
            push!(m.graph, n => [])
        end

        # register factors
        for f in factors
            register(f, m)
        end
        m
    end
end

# register a factor with its associated nodes in the graph
function register(f::Factor, m::VBModel) 
    for var in fieldnames(f)
        n = getfield(f, var)
        if isa(n, Node)
            push!(m.graph[n], (f, var))
        end
    end
end

# function check_conjugate(n::Distribution, m::VBModel)
#     is_conj = Bool[method_exists(naturals, Tuple{typeof(f), Type{Val{s}}, typeof(n)}) for (f, s) in m.graph[n]]
#     all(is_conj)
# end

###################################################
# Define some factors
###################################################
immutable EntropyFactor{N} <: Factor{N}
    x::Node
    inds::FactorInds
    namemap::Dict{Symbol, Symbol}
end

immutable LogNormalFactor{N} <: Factor{N}
    x::Node
    μ::Node  # mean
    τ::Node  # precision
    inds::FactorInds
    namemap::Dict{Symbol, Symbol}
end

immutable LogGammaFactor{N} <: Factor{N}
    x::Node
    α::Node  # shape
    β::Node  # rate
    inds::FactorInds
    namemap::Dict{Symbol, Symbol}
end

# define an expectation method on Distributions
"Calculate the expected value of a Node x."
E(x) = x
E(x::Distribution) = mean(x)
V(x) = zero(x)
V(x::Distribution) = var(x)
H(x) = zero(x)
H(x::Distribution) = entropy(x)

# Define functions for nonrandom nodes.
# In each case, a specialized method is already defined for distributions.
Elog(x) = log(x)
Eloggamma(x) = lgamma(x)

# "Calculate the contribution of a Factor f to the objective function."
value{N}(::Type{LogNormalFactor{N}}) = quote
    -(1/2) * ((E(τ) * ( V(x) + V(μ) + (E(x) - E(μ))^2 ) + log(2π) + Elog(τ)))
end

value{N}(::Type{LogGammaFactor{N}}) = quote
    (E(α) - 1) * Elog(x) - E(β) * E(x) + E(α) * E(β) - Eloggamma(α)
end

value{N}(::Type{EntropyFactor{N}}) = quote H(x) end

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
        η_type = typeof(naturals(n.data[1]))
        η = Array{η_type}(get_node_size(f, n)...) 
        for i in eachindex(η)
            η[i] = map(zero_like, η[i])
        end

        @nloops $N i d -> 1:f.inds.ranges[d] begin
            nats = @wrapvars $vars $nat_expr (@ntuple $N i)
            nat_tup = project_inds(f, S, (@ntuple $N i))
            η[nat_tup...] = map(.+, η[nat_tup...], nats)
        end
        η
    end
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

@defnaturals LogNormalFactor x Normal begin
    Eμ, Eτ = E(μ), E(τ)
    (Eμ * Eτ, -Eτ/2)
end

@defnaturals LogNormalFactor μ Normal begin
    Ex, Eτ = E(x), E(τ)
    (Ex * Eτ, -Eτ/2)
end

@defnaturals LogNormalFactor τ Gamma begin
    v = V(x) + V(μ) + (E(x) - E(μ))^2
    (1/2, v/2)
end

@defnaturals LogGammaFactor x Gamma begin
    (E(α) - 1, -E(β))
end

# "Return natural parameters from a Factor f viewed as a distribution for 
# a given symbol. The last parameter is a type check for conjugacy."
# naturals{N}(::Type{LogNormalFactor{N}}, ::Type{Val{:x}}, ::Type{Normal}) = quote
#     Eμ, Eτ = E(μ), E(τ)
#     (Eμ .* Eτ, -Eτ/2)
# end
# naturals{N}(::Type{LogNormalFactor{N}}, ::Type{Val{:μ}}, ::Type{Normal}) = quote
#     Ex, Eτ = E(x), E(τ)
#     (Ex .* Eτ, -Eτ/2)
# end
# naturals{N}(::Type{LogNormalFactor{N}}, ::Type{Val{:τ}}, ::Type{Gamma}) = quote
#     v = V(x) + V(μ) + (E(x) - E(μ)).^2
#     (1/2, v/2)
# end
# naturals{N}(::Type{LogGammaFactor{N}}, ::Type{Val{:x}}, ::Type{Gamma}) = quote
#     (E(α) - 1, -E(β))
# end


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


