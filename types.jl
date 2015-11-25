# let's define some types

abstract VBModel  

"Defines a factor, a term in the variational objective."
abstract Factor

"Defines a node (variable) in the model."
abstract Node

"Node corresponding to a collection of random variables."
type RandomNode{D <: Distribution, N} <: Node
    data::Array{D, N}
    factors::Vector{Factor}
end
RandomNode{D <: Distribution}(d::Type{D}, pars...) = RandomNode(map(d, pars...), Factor[])

type ConstantNode{T <: Number, N} <: Node
    data::Array{T, N}
    factors::Vector{Factor}
end

convert(::Type{Node}, x::Array) = ConstantNode(x, Factor[])

# register a factor with its associated nodes in the graph
function register(f::Factor) 
    for var in fieldnames(f)
        n = getfield(f, var)
        push!(n.factors, f)
    end
end


# define some factors
type EntropyFactor <: Factor
    x::RandomNode
end

type LogNormalFactor <: Factor
    x::Node
    μ::Node
    τ::Node
end


# define an expectation method on Nodes
"Calculate the expected value of a Node x."
E(x::RandomNode) = map(mean, x.data)
E(x::ConstantNode) = x.data

"Calculate the variance of a Node x."
var(x::RandomNode) = map(var, x.data)
var(x::ConstantNode) = zeros(x.data)

"Calculate the entropy of a Node x."
entropy(x::ConstantNode) = zeros(x.data)
entropy(x::RandomNode) = map(entropy, x.data)

"Calculate the expected value of the log of a Node x."
Elog(x::ConstantNode) = map(log, x.data)

"Calculate the contribution of a Factor f to the objective function."
value(f::LogNormalFactor) = -(1/2) * sum((E(f.τ) .* ( var(f.x) + var(f.μ) + 
    (E(f.x) - E(f.μ)).^2 ) + log(2π) + Elog(f.τ)))

value(f::EntropyFactor) = sum(entropy(f.x))