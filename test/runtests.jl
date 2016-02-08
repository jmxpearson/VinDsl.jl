push!(LOAD_PATH, "$(homedir())/code/VinDsl.jl/src")  # needed if VB not a full module
tests = [
    "tests",
    "hmmtests"
]

print_with_color(:blue, "Running tests:\n")

srand(12345)

for t in tests
    test_fn = "$t.jl"
    print_with_color(:green, "* $test_fn\n")
    include(test_fn)
end
