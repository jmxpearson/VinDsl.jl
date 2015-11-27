# let's define some types

"Defines a factor, a term in the variational objective."
abstract Factor

"Defines a node (variable) in the model."
abstract Node

"Defines a Variational Bayes model."
type VBModel  
    nodes::Dict{Symbol, Node}
    factors::Vector{Factor}

    VBModel(nodes, factors) = begin
        for f in factors
            register(f)
        end
        new(nodes, factors)
    end
end

"Node corresponding to a collection of random variables."
type RandomNode{D <: Distribution, N} <: Node
    data::Array{D, N}

    # factormap includes all factors using this node, along 
    # with the field the node is bound to in that factor
    factormap::Dict{Factor, Symbol}
end

RandomNode{D <: Distribution}(d::Type{D}, pars...) = RandomNode(map(d, pars...), Dict{Factor, Symbol}())

type ConstantNode{T <: Number, N} <: Node
    data::Array{T, N}
    factormap::Dict{Factor, Symbol}
end

convert(::Type{Node}, x::Array) = ConstantNode(x, Dict{Factor, Symbol}())

# register a factor with its associated nodes in the graph
function register(f::Factor) 
    for var in fieldnames(f)
        n = getfield(f, var)
        if isa(n, RandomNode)
            push!(n.factormap, f => var)
        end
    end
end

function check_conjugate(n::RandomNode)
    is_conj = Bool[method_exists(naturals, Tuple{typeof(f), Type{Val{s}}, typeof(n)}) for (f, s) in n.factormap]
    all(is_conj)
end


# define some factors
type EntropyFactor <: Factor
    x::RandomNode
end

type LogNormalFactor <: Factor
    x::Node
    μ::Node  # mean
    τ::Node  # precision
end

type LogGammaFactor <: Factor
    x::Node
    α::Node  # shape
    β::Node  # rate
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
Elog(x::RandomNode) = map(Elog, x.data)

doc"Calculate the expected value of $\log \Gamma(x).$"
Eloggamma(x::ConstantNode) = map(lgamma, x.data)

"Calculate the contribution of a Factor f to the objective function."
value(f::LogNormalFactor) = -(1/2) * sum((E(f.τ) .* ( var(f.x) + var(f.μ) + 
    (E(f.x) - E(f.μ)).^2 ) + log(2π) + Elog(f.τ)))

value(f::LogGammaFactor) = sum((E(f.α) - 1) .* E(f.x) - E(f.β) .* E(f.x) + 
    E(f.α) .* E(f.β) - Eloggamma(f.α))

value(f::EntropyFactor) = sum(entropy(f.x))

"Return natural parameters from an exponential family Node x."
naturals(x::RandomNode) = map(naturals, x.data)

"Return natural parameters from a Factor f viewed as a distribution for 
a given symbol. The last parameter is a type check for conjugacy."
naturals(f::LogNormalFactor, ::Type{Val{:x}}, ::RandomNode{Normal}) = begin
    μ, τ = E(f.μ), E(f.τ)
    (μ .* τ, -τ/2)
end
naturals(f::LogNormalFactor, ::Type{Val{:μ}}, ::RandomNode{Normal}) = begin
    x, τ = E(f.x), E(f.τ)
    (x .* τ, -τ/2)
end
naturals(f::LogNormalFactor, ::Type{Val{:τ}}, ::RandomNode{Gamma}) = begin
    v = var(f.x) + var(f.μ) + (E(f.x) - E(f.μ)).^2
    (1/2, v/2)
end
naturals(f::LogGammaFactor, ::Type{Val{:x}}, ::RandomNode{Gamma}) = (E(f.α) - 1, -E(f.β))

"Update a RandomNode n."
update(n::RandomNode, ::Type{Val{:conjugate}}) = begin
    #foo
end

