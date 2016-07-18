using FactCheck
using VinDsl
srand(56778)

facts("Checking LKJ distribution") do
    lkjeta = 5
    p = 5
    println(dim(LKJcorr(lkjeta, p)))
    #d = LKJcorr(eta, p)
    #yy = VinDsl._lkj_to_beta_pars(eta, p)
    #@fact yy --> [6.5,6.5,6.5,6.5,6,6,6,5.5,5.5,5]
    yy = LKJcorr(lkjeta, p)
    @fact diag(yy) --> ones(p)
    #vv = LKJcorr()
    hh = entropy(yy)
    @fact hh --> -1.1430726249017593
end
