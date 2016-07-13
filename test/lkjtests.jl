using FactCheck
using VinDsl
srand(56778)

facts("Checking LKJ distribution") do
    eta = .4
    d = 5
    yy = VinDsl.lkj_to_beta_pars(eta, d)
    @fact yy --> [2.4,2.4,2.4,2.4,1.9,1.9,1.9,1.4,1.4,0.9]
    yy = draw_lkj(eta, d)
    @fact diag(yy) --> ones(d)
    hh = lkj_entropy(eta, d)
    @fact hh --> -1.1430726249017593
end
