# types used by the module


#################### Node #######################
abstract Node

#################### Factor #######################
"Defines a factor, a term in the variational objective."
abstract Factor{N}


###################################################
# RandomNode <: Node
###################################################
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
    elseif length(unique(dims)) == 1 && dims[1] != ()
        RandomNode{D}(name, indices, map(D, pars...))
    else
        RandomNode{D}(name, indices, [D(pars...)])
    end
end


###################################################
# ConstantNode <: Node
###################################################
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

function show(io::IO, n::Node)
    T = typeof(n)
    print(io, T.name, "{", T.parameters[1].name.name, "}", " ", n.name)
    inners = filter(x -> x != :scalar, n.innerinds)
    outers = filter(x -> x != :scalar, n.outerinds)
    if !isempty(inners)
        print(io, "(")
        _printinds(io, n.innerinds)
        print(io, ")")
    end
    if !isempty(outers)
        print(io, "[")
        _printinds(io, n.outerinds)
        print(io, "]")
    end
end

function _printinds(io::IO, inds::Vector{Symbol})
    for idx in inds
        lastind = last(inds)
        if idx == lastind
            print(io, idx)
        else
            print(io, idx, ",")
        end
    end
end

###################################################
# FactorInds
###################################################
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

function show(io::IO, f::Factor)
    println(io, typeof(f))
    for v in fieldnames(f)
        if v ∉ [:inds, :namemap]
            print(io, v, ": ")
            println(io, getfield(f, v))
        end
    end
end

###################################################
# ExprNode <: Node
###################################################
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
    explicit_inners = collect(get_all_inds(ex))
    stripped_ex = strip_inds(ex)
    fi = get_structure(explicit_inners, nodelist...)
    nodedict = Dict(n.name => n for n in nodelist)
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

    ExprNode(name, stripped_ex, nodedict, fi, innerinds, outerinds, dims)
end


###################################################
# ExprDist <: Distribution
###################################################
immutable ExprDist{V <: Val} <: Distribution
    nodedict::Dict{Symbol, Distribution}
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
            else
                m.update_strategy[n] = :undefined
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


function update!(m::VBModel)
    for n in m.nodes
        update!(n, m, Val{m.update_strategy[n]})
    end
end

function update!(n::Node, m::VBModel, ::Type{Val{:constant}})
end
