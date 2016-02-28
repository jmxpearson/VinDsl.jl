# VinDsl.jl: Fast and furious variational inference

# TO DO:
- [x] Set up as proper package
- [x] Get factor macros correctly finding indices
- [x] make sure said indices can be passed as optional inner args to get_structure
- [ ] Implement V/C so that multivariate normal works
    - [x] +
    - [x] ^
    - [x] *
    - [x] sum and product
    - [ ] integration tests
- [ ] Set up Travis
    - does FactCheck correctly fail?
- [ ] basic `pmodel` and `qmodel` macros?
- [ ] update README/docs see [this](http://maurow.bitbucket.org/notes/documenting-a-julia-package.html) blog post
- [ ] release to contributors
- [ ] state space models from Beal thesis
- [ ] Implement parameterized distributions &agrave; la [here](https://github.com/JuliaStats/Distributions.jl/pull/430)

# Roadmap

## Short term:
This is what we need to get the *Neuron's eye view* models running:
- E-calculus:
    - implement V, C, Elog, etc. (need a way to define these rules more easily)
    - rename expandE ⟶ expand?
- Poisson Model:
    - [ ] factors where we take a product over some index, rather than a sum?
        - could have a function that makes an outer node in a factor inner, but this seems like a hack, and breaks the DSL facade
        - could look for `:ref`-headed `Expr`s and get all symbols used as indices; make these inner
    - [ ] gamma priors with linked shape/rate
    - [ ] pieces of Poisson with binarized exponent
    - [ ] BFGS optimization piece
        - right now, hard to do automatic differentiation until [this](https://github.com/JuliaStats/Distributions.jl/pull/430) is resolved ([more background](https://github.com/JuliaStats/Distributions.jl/issues/432))
- Log-Normal Model
    - [ ] **Jacobian of λ ⟶ η**
    - [ ] ExprNode that's a linear combo of pieces
        - [ ] V calculus for this
        - [ ] naturals for this
            - if V calculus not ready, can brute-force by breaking in pieces
    - bottleneck model (where η is a node)
        - [ ] implement Poisson
        - [ ] Lambert W for mean
        - [ ] explicit optimization for noise variance
- both models
    - [ ] iteration and convergence monitoring

## Medium-term:
- maps from each distribution's canonical (and/or natural) parameters to unconstrained parameters (like Stan)
- use these to set up optimization functions for BFGS/CG optimization steps
- when Julia 0.5 has fast anonymous functions and closures, this will be much easier
- update strategies:
    - [ ] SGD
    - [ ] LEG (local expectation gradients)
    - [ ] location-scale trick (like BBVI)


## Long-term
- build a @model macro that's fed a single (multiline) expression of tilde statements
- these statements are reparsed, replacing ~ with @factor and changing, e.g., Normal -> LogNormalFactor
- factors are made, then model is built from them
- `:=` defines ExprNode
- if a node definition contains an expression instead of a symbol in one of its arguments, gensym a new ExprNode and pass this to constructor
- implement Base.show for new types
- better error reporting for mismatched indices, etc.
- profile naturals using .+ vs + (probably bad) vs a custom in-place update for speed
- make sure project and project_inds return slices, not copies
- use CartesianRange instead of generated functions to define value(f::Factor)
- turn project and project_ind into macros that take x -> x[i_1, i_2] , etc. so as to avoid recurring project function call
- implement caching of expected values, etc. in RandomNodes (node can still be immutable if cache is a dict that gets cleared whenever data is updated); functions like E, Elog, V, etc. would need to be decorated to search cache first, then calculate, then update cache
- implement caching in ExprNodes of things like E(n)
- look carefully at Lora.jl api and graph structure of model
