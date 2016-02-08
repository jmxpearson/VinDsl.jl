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

immutable ExprNode <: Node
    name::Symbol
    ex::Expr  # expression defining node
    nodedict::Dict{Symbol, Node}  # nodes in the expression
    inds::FactorInds  # so we can treat this like a factor
    innerinds::Vector{Symbol}
    outerinds::Vector{Symbol}
    dims::Vector{Int}
end

function ExprNode(name::Symbol, ex::Expr, nodelist::Vector{Node})
    fi = get_structure(nodelist...)
    nodedict = Dict([n.name => n for n in nodelist])
    inners = union([n.innerinds for n in nodelist]...)
    outers = union([n.outerinds for n in nodelist]...)
    allinds = union(inners, outers)
    outerinds = fi.indices  # fully outer inds
    innerinds = setdiff(allinds, outerinds)
    dims = fi.maxvals  # size of each fully outer index

    # now get rid of :scalar index (if it's not the only one)
    scalar_inds = findin(outerinds, [:scalar])
    if !isempty(scalar_inds) && length(outerinds) > 1
        scalar_index = scalar_inds[1]  # should only have 1 or 0 entries
        splice!(outerinds, scalar_index)
        splice!(dims, scalar_index)
    end

    ExprNode(name, ex, nodedict, fi, innerinds, outerinds, dims)
end

nodeextract(key, node) = node.nodedict[key]

macro exprnode(name, ex)
    nodelist = collect(get_all_syms(ex))
    qname = Expr(:quote, name)
    qex = Expr(:quote, ex)
    Eex = :(E($ex))
    out_expr = quote
        $name = ExprNode($qname, $qex, Node[$(nodelist...)])

        # need to fully qualify E, else running @exprnode
        # outside the module will not extend, but overwrite
        VinDsl.E(d::ExprDist{Val{$qname}}) = @wrapvars $nodelist (@expandE $Eex) nodeextract d
    end
    esc(out_expr)
end

immutable ExprDist{V <: Val} <: Distribution
    nodedict::Dict{Symbol, Distribution}
end

size(n::Node) = size(n.data)
size(n::ExprNode) = tuple(n.dims...)
ndims(n::Node) = length(size(n))

getindex(n::Node, inds...) = n.data[inds...]
function getindex(n::ExprNode, inds...)
    nd = Dict([(s => project(s, n, inds)) for (s, _) in n.nodedict])
    ExprDist{Val{n.name}}(nd)
end
setindex!(n::Node, val, inds...) = setindex!(n.data, val, inds...)

#################### Factor #######################
"Defines a factor, a term in the variational objective."
abstract Factor{N}


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
            v += @wrapvars $vars $val_expr project f (@ntuple $N i)
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
    par_sizes = get_par_sizes(n[1])
    npars = mapreduce(prod, +, par_sizes)

    ctr = 0
    for i in eachindex(n.data)
        n[i] = reroll_pars(n[i], par_sizes, x[ctr + 1: ctr + npars])
        ctr += npars
    end
end

function update!(m::VBModel)
    for n in m.nodes
        update!(n, m, Val{m.update_strategy[n]})
    end
end
