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

    #=
    if only one parameter passed, check if there is a matching constructor;
        if so, feed directly to D
        if not, assume this is a collection of parameters to be mapped over
    else if all pars have the same size, assume these are arrays of
        params to be mapped over;
    otherwise, assume they are tuples to be fed directly to D
    =#
    if length(pars) == 1
        if method_exists(call, (Type{D}, map(typeof, pars)...))
            RandomNode{D}(name, indices, [D(pars...)])
        else
            RandomNode{D}(name, indices, map(D, pars...))
        end
    elseif length(unique(dims)) == 1
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
        if !isempty(indices)
            @assert ndims(data) == length(indices) "Indices do not match data shape."
        end
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

size(n::Node) = size(n.data)

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
    if length(project_inds(f, name, rangetuple)) > 0
        out = node.data[project_inds(f, name, rangetuple)...]
    else
        out = node.data
    end
    out
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
    EΛ = E(Λ)
    -(1/2) * (trace(EΛ * (V(x) .+ V(μ) .+ δ * δ')) + length(x) * log(2π) - Elogdet(Λ))
end

@deffactor LogMvNormalDiagCanonFactor [x, μ, τ] begin
    δ = E(x) - E(μ)
    Eτ = E(τ)
    -(1/2) * (sum(Eτ .* (V(x) .+ V(μ) .+ δ * δ')) + length(x) * log(2π) - Elogdet(τ))
end

@deffactor LogDirichletFactor [x, α] begin
    dot(E(α) - 1, Elog(x)) - ElogB(α)
end

################### Dealing with HMMs in factors #####################
@deffactor LogMarkovChainFactor [z, π0, A] begin
    dot(E(z)[:, 1], Elog(π0)) + sum(C(z) .* Elog(A))
end

@deffactor LogMarkovMatrixFactor [x, A] begin
    sum((E(A) - 1) .* Elog(x) .- ElogB(A))
end


################### Product-of-Rates Poisson model #####################
# @deffactor LogProdofRatesPoisson [z, λ, N] begin
#     dot(Elog(λ), )
# end

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
Eloggamma(x) = lgamma(x)
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

to_map = [E, V, C, H, Elog, Eloggamma, Elogdet]
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
        η = Array{η_type}(size(n.data)...)
        for i in eachindex(η)
            η[i] = map(zero_like, nats)
        end

        @nloops $N i d -> 1:f.inds.maxvals[d] begin
            nats = @wrapvars $vars $nat_expr (@ntuple $N i)
            nat_tup = project_inds(f, S, (@ntuple $N i))
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

nats_mvn{T <: Number}(μ::Vector{T}, Λ::Matrix{T}, x::MvNormalCanon) =
    (Λ * μ, -Λ/2)
nats_mvn{T <: Number}(μ::Number, Λ::Matrix{T}, x::MvNormalCanon) =
    (μ * sum(Λ, 2), -Λ/2)
nats_mvn{T <: Number}(μ::Vector{T}, τ::Vector{T}, x::MvNormalCanon) =
    (τ .* μ, -diagm(τ)/2)
nats_mvn{T <: Number}(μ::Number, τ::Vector{T}, x::MvNormalCanon) =
    (τ * μ, -diagm(τ)/2)
nats_mvn{T <: Number}(μ::Vector{T}, τ::Number, x::MvNormalCanon) =
    (τ * μ, -τ * eye(length(μ))/2)
nats_mvn(μ::Number, τ::Number, x::MvNormalCanon) =
    (μ * τ * ones(length(x)), -τ * eye(length(x))/2)
nats_mvn{T <: Number}(μ::Vector{T}, Λ::Matrix{T}, x::Normal) =
    (sum(Λ * μ), -sum(Λ)/2)
nats_mvn{T <: Number}(μ::Number, Λ::Matrix{T}, x::Normal) =
    (μ * sum(Λ), -sum(Λ)/2)
nats_mvn{T <: Number}(μ::Vector{T}, τ::Vector{T}, x::Vector{Normal}) =
    [nats_mvn(m, t) for (m, t) in zip(μ, τ)]
nats_mvn{T <: Number}(μ::Vector{T}, τ::Number, x::Vector{Normal}) =
    [nats_mvn(m, τ) for m in μ]
nats_mvn{T <: Number}(μ::Number, τ::Vector{T}, x::Vector{Normal}) =
    [nats_mvn(μ, t) for t in τ]
nats_mvn(μ, τ, x::Normal) =
    reduce(add_nats, nats_mvn(μ, τ, [x]))
nats_mvn(μ::Number, τ::Number) = (μ * τ, -τ/2)
nats_mvn(v::Vector, x::Vector{Gamma}) =
    Tuple{Float64, Float64}[(1/2, vv/2) for vv in v]
nats_mvn(v::Vector, x::Gamma) =
    reduce(add_nats, nats_mvn(v, [x]))

@defnaturals LogMvNormalCanonFactor x MvNormalCanon begin
    nats_mvn(E(μ), E(Λ), x)
end

@defnaturals LogMvNormalCanonFactor μ MvNormalCanon begin
    nats_mvn(E(x), E(Λ), μ)
end

@defnaturals LogMvNormalCanonFactor x Normal begin
    nats_mvn(E(μ), E(Λ), x)
end

@defnaturals LogMvNormalCanonFactor μ Normal begin
    nats_mvn(E(x), E(Λ), μ)
end

@defnaturals LogMvNormalCanonFactor Λ Wishart begin
    δ = E(x) - E(μ)
    v = C(x) .+ C(μ) .+ δ * δ'
    (v/2, 0.)
end

@defnaturals LogMvNormalDiagCanonFactor x Normal begin
    nats_mvn(E(μ), E(τ), x)
end

@defnaturals LogMvNormalDiagCanonFactor x MvNormalCanon begin
    nats_mvn(E(μ), E(τ), x)
end

@defnaturals LogMvNormalDiagCanonFactor μ Normal begin
    nats_mvn(E(x), E(τ), μ)
end

@defnaturals LogMvNormalDiagCanonFactor μ MvNormalCanon begin
    nats_mvn(E(x), E(τ), μ)
end

@defnaturals LogMvNormalDiagCanonFactor τ Gamma begin
    δ = E(x) - E(μ)
    v = V(x) + V(μ) + δ.^2
    nats_mvn(v, τ)
end

@defnaturals LogMarkovChainFactor π0 Dirichlet begin
    (E(z)[:, 1], )
end

@defnaturals LogMarkovChainFactor A MarkovMatrix begin
    (slice(sum(C(z), 3), :, :), )
end

@defnaturals LogMarkovChainFactor z HMM begin
    (zero_like(z.ψ), Elog(π0), Elog(A))
end

@defnaturals LogDirichletFactor x Dirichlet begin
    (E(α) - 1,)
end

@defnaturals LogMarkovMatrixFactor x MarkovMatrix begin
    (E(A) - 1,)
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

typealias explicit_opts Union{Val{:l_bfgs}, Val{:cg}}
function update!{D, S <: explicit_opts}(n::RandomNode{D}, m::VBModel, ::Type{S})
    # make backup of node
    pars_old = copy(n.data)

    # use const here so that the closure below can be optimized
    const fac_list = [f for (f, _) in m.graph[n]]

    const x0 = unroll_pars(n)
    # TODO: get lower and upper bounds from each variable in x from n

    # make an objective function that sets the parameters of n to x
    # and sums the values of all factors containing n
    function objfun(x)
        update_pars!(n, x)
        val = 0
        for f in fac_list
            val += value(f)
        end
        val
    end

    # use ForwardDiff to get the gradient; autodiff in optimize uses
    # ReverseDiff, but ForwardDiff only requires args <: Real, but
    # ReverseDiff needs args <: Number, which Distributions doesn't allow
    gradf! = ForwardDiff.gradient(objfun, mutates=true)

    # define mutating gradient; store gradient in storage array
    objgrad!(x, storage) = gradf!(storage, x)

    # try optimization; if it fails, set parameters back to initial guess
    try
        optimize(objfun, objgrad!, x0, method=S.parameters[1])
    catch
        finalpars = unroll_pars(n)
        println("Optimization failed at pars:\n$finalpars")
        update_pars!(n, x0)
    end
end

function get_par_sizes(d::Distribution)
    [size(p) for p in params(d)]
end

flatten(a::Number) = a
flatten(a::Array) = reshape(a, prod(size(a)))
flatten(a::AbstractPDMat) = flatten(full(a))

function unroll_pars(n::RandomNode)
    vcat(map(unroll_pars, n.data)...)
end

function unroll_pars(d::Distribution)
    vcat([flatten(par) for par in params(d)]...)
end

function reroll_pars{D <: Distribution}(d::D, par_sizes, x)
    ctr = 0
    pars = []
    for (i, dims) in enumerate(par_sizes)
        sz = prod(dims)
        p = sz == 1 ? x[ctr + 1] : reshape(x[ctr + 1: ctr + sz], dims)
        push!(pars, p)
        ctr += sz
    end
    D(pars...)
end

function update_pars!(n::RandomNode, x)
    par_sizes = get_par_sizes(n.data[1])
    npars = mapreduce(prod, +, par_sizes)

    ctr = 0
    for i in eachindex(n.data)
        n.data[i] = reroll_pars(n.data[i], par_sizes, x[ctr + 1: ctr + npars])
        ctr += npars
    end
end

function update!(m::VBModel)
    for n in m.nodes
        update!(n, m, Val{m.update_strategy[n]})
    end
end
