#=
code to handle parsing and manipulating of expressions defining nodes,
factors, naturals, and value functions
=#

"""
Recursively wrap variables in an expression so that in the new expression,
they are projected down to their index tuples in a factor with variables
given in vars.
"""
macro wrapvars(vars, ex, wrapper, varargs...)
    esc(_wrapvars(vars, ex, wrapper, varargs...))
end

_wrapvars(vars, ex, wrapper, varargs...) = ex

function _wrapvars(vars::Vector{Symbol}, ex::Expr, wrapper, varargs...)
    # copy AST
    new_ex = copy(ex)

    # recursively wrap variables
    for i in eachindex(ex.args)
        new_ex.args[i] = _wrapvars(vars, ex.args[i], wrapper, varargs...)
    end

    # return new expression
    new_ex
end

function _wrapvars(vars::Vector{Symbol}, s::Symbol, wrapper, varargs...)
    # if s is a variable in the approved list of vars
    if s in vars
        sym = Expr(:quote, s)
        # :(project(f, $sym, $indtup))
        :($wrapper($sym, $(varargs...)))
    else
        s
    end
end


"""
Recursively make a list of all symbols used as arguments in a given expression.
"""
get_all_syms(x) = Set(Symbol[])
get_all_syms(s::Symbol) = Set(Symbol[s])
function get_all_syms(ex::Expr)
    symset = Set{Symbol}()
    _get_all_syms(ex, symset)
    symset
end
function _get_all_syms(ex::Expr, symset)
    if ex.head in [:call, :quote]
        arglist = ex.args[2:end]
    else
        arglist = ex.args
    end
    for arg in arglist
        if isa(arg, Symbol)
            push!(symset, arg)
        elseif isa(arg, Expr)
            _get_all_syms(arg, symset)
        end
    end
end

"""
Recursively get a list of all index symbols used in a given expression.
"""
get_all_inds(x) = Set(Symbol[])
get_all_inds(s::Symbol) = Set(Symbol[])
function get_all_inds(ex::Expr)
    indset = Set{Symbol}()
    _get_all_inds(ex, indset)
    indset
end
function _get_all_inds(ex::Expr, indset)
    if ex.head == :ref
        for arg in ex.args[2:end]
            if isa(arg, Symbol)
                push!(indset, arg)
            end
        end
    else
        for arg in ex.args
            if isa(arg, Expr)
                _get_all_inds(arg, indset)
            end
        end
    end
end

"""
Remove all indexing from an expression.
"""
strip_inds(x) = x
strip_inds(s::Symbol) = s
function strip_inds(ex::Expr)
    _strip_inds(ex)
end
_strip_inds(x) = x
function _strip_inds(ex::Expr)
    if ex.head == :ref
        out = ex.args[1]
    else
        out = copy(ex)
        out.args = map(_strip_inds, ex.args)
    end
    out
end
