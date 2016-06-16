# code for node updating via optimization

typealias explicit_opts Union{Val{:l_bfgs}, Val{:cg}}
function update!{D, S <: explicit_opts}(n::RandomNode{D}, m::VBModel, ::Type{S})
    # use const here so that the closure below can be optimized
    const fac_list = [f for (f, _) in m.graph[n]]

    const x0 = unroll_pars(n)
    # TODO: get lower and upper bounds from each variable in x from n

    # make an objective function that sets the parameters of n to x
    # and sums the values of all factors containing n
    function objfun(x)
        update_pars!(n, x)
        val = 0.0
        for f in fac_list
            val += value(f)
        end
        -val  # Optim minimizes, so objfun = -ELBO
    end

    # use ForwardDiff to get the gradient; autodiff in optimize uses
    # ReverseDiff, but ForwardDiff only requires args <: Real, but
    # ReverseDiff needs args <: Number, which Distributions doesn't allow
    gradf! = (out, x) -> ForwardDiff.gradient!(out, objfun, x)

    # define mutating gradient; store gradient in storage array
    objgrad!(x, storage) = gradf!(storage, x)

    if S.parameters[1] == :l_bfgs
        method = LBFGS()
    elseif S.parameters[1] == :cg
        method = ConjugateGradient()
    end

    # try optimization; if it fails, set parameters back to initial guess
    try
        optimize(objfun, objgrad!, x0, method=method)
    catch
        finalpars = unroll_pars(n)
        println("Optimization failed at pars:\n$finalpars")
        update_pars!(n, x0)
    end
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
