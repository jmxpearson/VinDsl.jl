# code to handle the domain-specific language used to define models

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

"""
Define a node by an expression involving other nodes.
"""
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

###################################################
# Macro to define a factor
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


################# macros/functions to generalize E(⋅) to expressions ##########
macro expandE(ex)
    esc(_expandE(ex))
end

# _expandE expands the expectations in an arbitrary expression
# _expand_wrapE expands the expression inside an E(⋅) and wraps the result in E
_expandE(x) = x  # all terminal expressions not otherwise specified
_expandE(x::Symbol) = x
function _expandE(ex::Expr)
    if ex.head == :call && ex.args[1] in [:E]  # E call
        out_expr = _expand_wrapE(ex.args[2], ex.args[1])
    else
        out_expr = ex
    end
    out_expr
end

_expand_wrapE(x, f) = x
_expand_wrapE(x::Symbol, f::Symbol) = :($f($x))
function _expand_wrapE(ex::Expr, f::Symbol)
    if ex.head == :call && ex.args[1] in [:E]  # E call
        out_expr = _expand_wrapE(ex.args[2], ex.args[1])

    # linearity of E over +, -
    elseif ex.head == :call && ex.args[1] in [:+, :-, :.+, :.-]
        out_expr = ex
        rest = ex.args[2:end]
        for (i, arg) in enumerate(rest)
            out_expr.args[i + 1] = _expand_wrapE(arg, f)
        end

    # linearity of E for *
    # STRONG ASSUMPTION: different symbols/nodes are INDEPENDENT
    elseif ex.head == :call && ex.args[1] in [:*]
        op = ex.args[1]  # operator
        rest = ex.args[2:end]  # factors

        # symbol factors
        symsets = [get_all_syms(r) for r in rest]

        # are symbols repeated across arguments?
        pairwise_intersections = map(x -> intersect(x[1], x[2]), combinations(symsets, 2))
        repeats = any(x -> !isempty(x), pairwise_intersections)

        out_expr = ex
        if !repeats
            for (i, arg) in enumerate(rest)
                out_expr.args[i + 1] = _expand_wrapE(arg, f)
            end
        else
            # some variables are repeated, and we can't assume * is
            # commutative, so don't rearrange factors

            repeated_syms = union(pairwise_intersections...)

            # which positions contain repeated symbols?
            syminds = []
            for i in eachindex(rest)
                if !isempty(intersect(symsets[i], repeated_syms))
                    push!(syminds, i)
                end
            end

            # check to see whether we can do any rewriting:
            if (syminds[1] == 1 && syminds[end] == length(rest))
                # first and last expressions contain shared symbols:
                # no rewriting possible

                # wrap in E(⋅) and output
                out_expr = Expr(:call, :E, out_expr)

            else
                # we can rewrite arguments of Expr
                out_expr.args = [op]

                # expand everything up to first repeated symbol
                if syminds[1] > 1
                    for arg in rest[1:(syminds[1] - 1)]
                        push!(out_expr.args, _expand_wrapE(arg, f))
                    end
                end

                # clump everything between repeated symbols together
                base_expr = Expr(:call, op, rest[syminds[1]:syminds[end]]...)
                E_expr = Expr(:call, :E, base_expr)
                push!(out_expr.args, E_expr)

                # expand everything after last repeated symbol
                if syminds[end] < length(rest)
                    for arg in rest[(syminds[end] + 1):end]
                        push!(out_expr.args, _expand_wrapE(arg, f))
                    end
                end
            end
        end

    else
        out_expr = :($f($ex))
    end
    out_expr
end
