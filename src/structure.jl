#=
functions to help with the construction and management of factor, node, and model structure
=#

macro factor(ftype, nodes...)
    ex = quote
        fi = get_structure($(nodes...))
        symdict = get_name_mapping($ftype, $(nodes...))
        $ftype{length(fi.indices)}($(nodes...), fi, symdict)
    end
    esc(ex)
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
            v += @wrapvars $vars $val_expr project f (@ntuple $N i)
        end
        v
    end
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
get_structure(nodes...) = get_structure(Symbol[], nodes...)

function get_structure(forced_inners::Vector{Symbol}, nodes...)
    # first, get all indices that are not inner for any node
    # these will be the nodes that need explicit summing
    outers = union([n.outerinds for n in nodes]...)
    inners = union(forced_inners, [n.innerinds for n in nodes]...)
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
                push!(idxdict[idx], size(n)[d])
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
function project(name::Symbol, f, rangetuple)
    node = node_from_name(f, name)
    if length(project_inds(name, f, rangetuple)) > 0
        out = node[project_inds(name, f, rangetuple)...]
    else
        alltuple = [Colon() for _ in 1:ndims(node)]
        out = node[alltuple...]
    end
    out
end

function project_inds(name::Symbol, f, rangetuple)
    node = node_from_name(f, name)
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

###################################################
# helper functions on nodes
###################################################
node_from_name(f::Factor, name::Symbol) = getfield(f, name)
node_from_name(f::ExprNode, name::Symbol) = f.nodedict[name]

nodeextract(key, node) = node.nodedict[key]

size(n::Node) = size(n.data)
size(n::ExprNode) = tuple(n.dims...)
ndims(n::Node) = length(size(n))

getindex(n::Node, inds...) = n.data[inds...]

function getindex(n::ExprNode, inds...)
    nd = Dict([(s => project(s, n, inds)) for (s, _) in n.nodedict])
    ExprDist{Val{n.name}}(nd)
end
setindex!(n::Node, val, inds...) = setindex!(n.data, val, inds...)

function get_par_sizes(d::Distribution)
    [size(p) for p in params(d)]
end

function unroll_pars(n::RandomNode)
    vcat(map(unroll_pars, n.data)...)
end

function unroll_pars(d::Distribution)
    vcat([flatten(par) for par in uparams(d)]...)
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
    D.name.primary(constrain(pars, D)...)
end
