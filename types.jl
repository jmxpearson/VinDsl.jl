# let's define some types
#################### Node #######################

abstract Node

immutable RandomNode{D <: Distribution} <: Node
    name::Symbol
    innerinds::Vector{Symbol}
    outerinds::Vector{Symbol}
    data::Array{D}

    function RandomNode(name, indices, data)
        # handle the case where node is scalar in outer indices
        if length(data) == 1
            inds = vcat(indices, :scalar)
        else
            inds = copy(indices)
        end

        # inner indices are assumed listed first
        ninds = length(inds)
        nouter = ndims(data)
        @assert nouter <= ninds "Indices do not match data shape"
        outerinds = inds[ninds - nouter + 1:end]
        innerinds = inds[1:ninds - nouter]
        ninner = length(innerinds)
        @assert nouter == length(outerinds) "Indices do not match data shape."
        if ninner > 0
            # length(size(data[1])) gives number of dimensions in
            # the random distribution D
            @assert ninner == length(size(data[1])) "Inner indices, if provided, must match distribution shape."
        end
        new(name, innerinds, outerinds, data)
    end
end
RandomNode{D <: Distribution}(name::Symbol, indices::Vector{Symbol}, ::Type{D}, pars...) = begin
    dims = map(size, pars)

    # if all pars have the same size, assume these are arrays of
    # params to be mapped over, otherwise, assume they are tuples to be
    # fed directly to D
    if length(unique(dims)) == 1
        RandomNode{D}(name, indices, map(D, pars...))
    else
        RandomNode{D}(name, indices, [D(pars...)])
    end
end

immutable ConstantNode{T} <: Node
    name::Symbol
    innerinds::Vector{Symbol}
    outerinds::Vector{Symbol}
    data::Array{T}

    function ConstantNode(name, indices, data)
        @assert ndims(data) == length(indices) "Indices do not match data shape."
        outerinds = indices
        innerinds = Symbol[]
        new(name, innerinds, outerinds, data)
    end
end
ConstantNode{T}(name::Symbol, indices::Vector{Symbol}, data::Array{T}) = ConstantNode{T}(name, indices, data)
ConstantNode(name::Symbol, indices::Vector{Symbol}, data::Number) = ConstantNode(name, [:scalar], [data])
ConstantNode{T}(data::Array{T}, indices::Vector{Symbol}) =
    ConstantNode(gensym("const"), indices, data)
ConstantNode(x::Number) = ConstantNode(gensym("const"), [:scalar], [x])

"""
Create a node using formula syntax. E.g.,
x[i, j, k] ~ Normal(μ, σ)
z ~ Gamma(a, b)
y[a, b] ~ Const(Y)  (for a constant node)
"""
macro ~(varex, distex)
    if isa(varex, Symbol)
        name = varex
        inds = :(Symbol[])
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
    indices::Vector{Symbol}  # all fully outer indices
    maxvals::Vector{Int}  # maximum value of each index
    #=
    The next two dicts take care of a pair of necessary mappings:
    - inds_in_factor maps a node symbol to the (integer) subset of factor
        indices involving this node
    - inds_in_node maps a node symbol to the (integer) subset of node indices
        involved in this factor
    =#
    inds_in_factor::Dict{Symbol, Vector{Int}}
    inds_in_node::Dict{Symbol, Vector{Int}}
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

TODO: Example showing how inner and outer indices get parsed and what the
inds_in_* mappings are.
"""
function get_structure(nodes...)
    # first, get all indices that are not inner for any node
    # these will be the nodes that need explicit summing
    outers = union([n.outerinds for n in nodes]...)
    inners = union([n.innerinds for n in nodes]...)
    allinds = setdiff(outers, intersect(outers, inners))

    # now, map these indices to consecutive integers
    idx_to_int = [idx => idxnum for (idxnum, idx) in enumerate(allinds)]

    #=
    initialize some Dicts that will be useful for working with factors:
    - idxdict: map index symbol to the size of each node along this
        dimension
    - node_to_int_inds: map node symbol to the integer codes within the
        factor of its fully outer indices
    - ints_within_node: map node symbol to the subset of *all* its indices
        corresponding to the fully outer indices
    =#
    idxdict = Dict{Symbol, Vector{Int}}([(idx => []) for idx in allinds])
    inds_in_factor = Dict{Symbol, Vector{Int}}([(n.name => []) for n in nodes])
    inds_in_node = Dict{Symbol, Vector{Int}}([(n.name => []) for n in nodes])

    # now loop over nodes, building these dicts
    for n in nodes
        for (d, idx) in enumerate(n.outerinds)
            # if this outer index is to be summed over...
            if idx in allinds
                push!(idxdict[idx], size(n.data, d))
                push!(inds_in_factor[n.name], idx_to_int[idx])
                push!(inds_in_node[n.name], indexin([idx], n.outerinds)[1])
            end
        end
    end

    # lastly, check that the index ranges are the same for every node in
    # which an index appears; build a vector of index ranges
    maxvals = Integer[]
    for idx in allinds
        lengths = idxdict[idx]
        if !all(x -> x == lengths[1], lengths)
            error("Index length mismatch in index $idx.")
        else
            push!(maxvals, lengths[1])
        end
    end

    FactorInds(allinds, maxvals, inds_in_factor, inds_in_node)
end

function get_node_size(f::Factor, n::Node)
    fi = f.inds
    syminds = fi.inds_in_factor[n.name]
    fi.maxvals[syminds]
end

function get_name_mapping{F <: Factor}(ftype::Type{F}, nodes...)
    nodenames = [n.name for n in nodes]
    Dict(zip(nodenames, fieldnames(ftype)))
end

"""
Given a factor, a symbol naming a variable in that factor, and a tuple of
index ranges for the factor as a whole, return the elements of node
corresponding to the global range of indices.
"""
function project(f::Factor, name::Symbol, rangetuple)
    node = getfield(f, name)
    node.data[project_inds(f, name, rangetuple)...]
end

function project_inds(f::Factor, name::Symbol, rangetuple)
    node = getfield(f, name)
    factor_inds = f.inds.inds_in_factor[node.name]
    node_inds = f.inds.inds_in_node[node.name]
    ninds = length(node.outerinds)

    outtuple = repmat(Any[Colon()], ninds)
    if !isempty(node_inds)
        for (ni, fi) in zip(node_inds, factor_inds)
            outtuple[ni] = rangetuple[fi]
        end
    end
    outtuple
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
        @nloops $N i d -> 1:f.inds.maxvals[d] begin
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

    # dictionary containing the update strategy for each node
    update_strategy::Dict{Node, Symbol}

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

        # calculate default update strategy
        m.update_strategy = Dict{Node, Symbol}()
        for n in nodes
            if isa(n, ConstantNode)
                m.update_strategy[n] = :constant
            elseif isa(n, RandomNode) && check_conjugate(n, m)
                m.update_strategy[n] = :conjugate
            end
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

function check_conjugate{D}(n::RandomNode{D}, m::VBModel)
    is_conj = true
    for (f, s) in m.graph[n]
        ttype = Tuple{Type{typeof(f)}, Type{Val{s}}, Type{D}}
        if !method_exists(natural_formula, ttype)
            is_conj = false
            break
        end
    end
    is_conj
end

###################################################
# Define some factors
###################################################
macro deffactor(typename, vars, valexpr)
    varlist = [:($v::Node) for v in vars.args]
    varblock = Expr(:block, varlist...)
    value_formula = Expr(:quote, valexpr)

    ex = quote
        immutable ($typename){N} <: Factor{N}
            $varblock
            inds::FactorInds
            namemap::Dict{Symbol, Symbol}
        end

        value{N}(::Type{($typename){N}}) = $value_formula
    end
    esc(ex)
end

@deffactor EntropyFactor [x] H(x)

@deffactor LogNormalFactor [x, μ, τ] begin
    -(1/2) * ((E(τ) * ( V(x) + V(μ) + (E(x) - E(μ))^2 ) + log(2π) + Elog(τ)))
end

@deffactor LogGammaFactor [x, α, β] begin
    (E(α) - 1) * Elog(x) - E(β) * E(x) + E(α) * E(β) - Eloggamma(α)
end

@deffactor LogMvNormalCanonFactor [x, μ, Λ] begin
    δ = E(x) - E(μ)
    EL = E(Λ)
    EΛ = ndims(EL) == 1 ? diagm(EL) : EL
    -(1/2) * (trace(EΛ * (V(x) .+ V(μ) .+ δ * δ')) + length(x) * log(2π) - Elogdet(Λ))
end

# define an expectation method on Distributions
"Calculate the expected value of a Node x."
E(x) = x
E(x::Distribution) = mean(x)
V(x) = zero(x)
V(x::Distribution{Univariate}) = var(x)
V(x::AbstractMvNormal) = cov(x)
V{D <: Distribution}(x::Vector{D}) = diagm(map(V, x))
H(x) = zero(x)
H(x::Distribution) = entropy(x)

# Define functions for nonrandom nodes.
# In each case, a specialized method is already defined for distributions.
Elog(x) = log(x)
Eloggamma(x) = lgamma(x)
Elogdet(x) = logdet(x)
Elogdet{D <: Distribution{Univariate}}(x::Array{D}) = prod(Elog(x))
Elogdet(x::Distribution{Univariate}) = Elog(x)

# now make version of all these functions that work on nodes
# by working elementwise
macro make_mapped_version(fn)
    esc(:($fn{D <: Distribution}(n::Array{D}) = map($fn, n)))
end

to_map = [E, V, H, Elog, Eloggamma, Elogdet]
for fn in to_map
    @make_mapped_version fn
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
        nats = naturals(n.data[1])
        η_type = typeof(nats)
        η = Array{η_type}(get_node_size(f, n)...)
        for i in eachindex(η)
            η[i] = map(zero_like, nats)
        end

        @nloops $N i d -> 1:f.inds.maxvals[d] begin
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

@defnaturals LogMvNormalCanonFactor x MvNormalCanon begin
    Eμ, EΛ = E(μ), E(Λ)
    (EΛ * Eμ, -EΛ/2)
end

@defnaturals LogMvNormalCanonFactor μ MvNormalCanon begin
    Ex, EΛ = E(x), E(Λ)
    (EΛ * Ex, -EΛ/2)
end

@defnaturals LogMvNormalCanonFactor μ Normal begin
    Ex, EΛ = E(x), E(Λ)
    (sum(EΛ * Ex), -sum(EΛ)/2)
end

@defnaturals LogMvNormalCanonFactor Λ Gamma begin
    δ = E(x) - E(μ)
    d = length(x)
    v = var(x) + var(μ) + δ.^2
    (d/2, sum(v)/2)
end

@defnaturals LogMvNormalCanonFactor Λ Wishart begin
    δ = E(x) - E(μ)
    d = length(x)
    v = V(x) .+ V(μ) .+ δ * δ'
    (v/2, 0)
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
        n.data[idx] = D(naturals_to_params(natpars, D)...)
    end
end

function update!(n::Node, m::VBModel, ::Type{Val{:constant}})
end

function update!(m::VBModel)
    for n in m.nodes
        update!(n, m, Val{m.update_strategy[n]})
    end
end
