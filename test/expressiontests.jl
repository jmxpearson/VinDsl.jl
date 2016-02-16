
facts("E calculus") do
    context("Get symbols in expression") do
        @fact get_all_syms(5) --> Set{Symbol}([])
        @fact get_all_syms(:x) --> Set([:x])
        @fact get_all_syms(:(x + 1)) --> Set([:x])
        @fact get_all_syms(:(x + y)) --> Set([:x, :y])
        @fact get_all_syms(:(x + (y * x))) --> Set([:x, :y])
        @fact get_all_syms(:(2x + (y * x))) --> Set([:x, :y])
    end

    context("Basic identities") do
        @fact _expandE(1) --> 1
        @fact _expandE(ones(5)) --> ones(5)
        @fact _expandE(:x) --> :(x)
        @fact _expandE(:(E(x))) --> :(E(x))
        @fact _expandE(:(x + y)) --> :(x + y)

        @fact _expand_wrapE(1, :E) --> 1
        @fact _expand_wrapE(ones(5), :E) --> ones(5)
        @fact _expand_wrapE(:x, :E) --> :(E(x))
        @fact _expand_wrapE(:(E(x)), :E) --> :(E(x))
        @fact _expand_wrapE(:(x + y), :E) --> :(E(x) + E(y))
    end

    context("+ and -") do
        @fact _expandE(:(E(x + y))) --> :(E(x) + E(y))
        @fact _expandE(:(E(x + y + z))) --> :(E(x) + E(y) + E(z))
        @fact _expandE(:(E(x - y))) --> :(E(x) - E(y))
        @fact _expandE(:(E(x - y + z))) --> :(E(x) - E(y) + E(z))
        @fact _expandE(:(E(x .+ y))) --> :(E(x) .+ E(y))
    end

    context("*") do
        @fact _expandE(:(E(2x))) --> :(2 * E(x))
        @fact _expandE(:(E(2x * 3))) --> :((2 * E(x)) * 3)
        @fact _expandE(:(E(2x * y))) --> :((2 * E(x)) * E(y))
        @fact _expandE(:(E(2 * x * y))) --> :(2 * E(x) * E(y))
        @fact _expandE(:(E(2 * x * y * x))) --> :(2 * E(x * y * x))
        @fact _expandE(:(E(2 * x * y * x * z))) --> :(2 * E(x * y * x) * E(z))
        @fact _expandE(:(E((x * y) * x))) --> :(E((x * y) * x))
        @fact _expandE(:(E((x * y) * x * (w * z)))) --> :(E((x * y) * x) * (E(w) * E(z)))
    end

    context("macro expansion") do
        x ~ Normal(rand(), rand())
        y ~ Normal(rand(), rand())

        xy = @expandE E(x.data[1] + y.data[1])
        @fact xy --> E(x.data[1]) + E(y.data[1])

        xy = @expandE E(x.data[1] * y.data[1] + 5)
        @fact xy --> E(x.data[1]) * E(y.data[1]) + 5
    end
end