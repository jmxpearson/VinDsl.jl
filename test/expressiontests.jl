
facts("E calculus") do
    context("Get symbols in expression") do
        @fact get_all_syms(5) --> Set{Symbol}([])
        @fact get_all_syms(:x) --> Set([:x])
        @fact get_all_syms(:(x + 1)) --> Set([:x])
        @fact get_all_syms(:(x + y)) --> Set([:x, :y])
        @fact get_all_syms(:(x + (y * x))) --> Set([:x, :y])
        @fact get_all_syms(:(2x + (y * x))) --> Set([:x, :y])
    end

    context("Get indices in expression") do
        @fact get_all_inds(5) --> Set{Symbol}([])
        @fact get_all_inds(:x) --> Set{Symbol}([])
        @fact get_all_inds(:(x[i] + 1)) --> Set([:i])
        @fact get_all_inds(:(x[i] + y[j, k])) --> Set([:i, :j, :k])
        @fact get_all_inds(:(log(x[i]) + y[j, k])) --> Set([:i, :j, :k])
    end

    context("Strip indexing from expression") do
        @fact strip_inds(5) --> 5
        @fact strip_inds(:x) --> :x
        @fact strip_inds(:(x[i] + 1)) --> :(x + 1)
        @fact strip_inds(:(x[i] + y[j, k])) --> :(x + y)
        @fact strip_inds(:(log(x[i]) + y[j, k])) --> :(log(x) + y)
    end

    context("Basic identities") do
        @fact _simplify(1) --> 1
        @fact _simplify(ones(5)) --> ones(5)
        @fact _simplify(:x) --> :(x)
        @fact _simplify(:(E(x))) --> :(E(x))
        @fact _simplify(:(x + y)) --> :(x + y)

        @fact _simplify_and_wrap(1, :E) --> 1
        @fact _simplify_and_wrap(ones(5), :E) --> ones(5)
        @fact _simplify_and_wrap(:x, :E) --> :(E(x))
        @fact _simplify_and_wrap(:(E(x)), :E) --> :(E(x))
        @fact _simplify_and_wrap(:(x + y), :E) --> :(E(x) + E(y))
    end

    context("+ and -") do
        @fact _simplify(:(E(x + y))) --> :(E(x) + E(y))
        @fact _simplify(:(E(x + y + z))) --> :(E(x) + E(y) + E(z))
        @fact _simplify(:(E(x - y))) --> :(E(x) - E(y))
        @fact _simplify(:(E(x - y + z))) --> :(E(x) - E(y) + E(z))
        @fact _simplify(:(E(x .+ y))) --> :(E(x) .+ E(y))
    end

    context("*") do
        @fact _simplify(:(E(2x))) --> :(2 * E(x))
        @fact _simplify(:(E(2x * 3))) --> :((2 * E(x)) * 3)
        @fact _simplify(:(E(2x * y))) --> :((2 * E(x)) * E(y))
        @fact _simplify(:(E(2 * x * y))) --> :(2 * E(x) * E(y))
        @fact _simplify(:(E(2 * x * y * x))) --> :(2 * E(x * y * x))
        @fact _simplify(:(E(2 * x * y * x * z))) --> :(2 * E(x * y * x) * E(z))
        @fact _simplify(:(E((x * y) * x))) --> :(E((x * y) * x))
        @fact _simplify(:(E((x * y) * x * (w * z)))) --> :(E((x * y) * x) * (E(w) * E(z)))
    end

    context("macro expansion") do
        x ~ Normal(rand(), rand())
        y ~ Normal(rand(), rand())

        xy = @simplify E(x.data[1] + y.data[1])
        @fact xy --> E(x.data[1]) + E(y.data[1])

        xy = @simplify E(x.data[1] * y.data[1] + 5)
        @fact xy --> E(x.data[1]) * E(y.data[1]) + 5
    end
end
