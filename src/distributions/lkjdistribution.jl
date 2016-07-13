# Define LKJ prior distribution

#immutable LKJcorr{T <: Real} <: Vector
#    η::T
#    d::Int
#end

#ndims(x::LKJcorr) = x.d

"""
Transform LKJ distribution with parameter eta for matrix of dimension d
to vector of beta distribution parameters:
p_{i >= 1, j>i; 1...i-1} ~ Beta(b_i, b_i)
b_i = eta + (d - 1 - i)/2
"""
function lkj_to_beta_pars(η, d::Int)
    idxmat = (d - 1)/2. * ones(d)
    for i in 2:d
        idxmat = hcat(idxmat, (d - i)/2. * ones(d))
    end
    bmat = η .+ idxmat
    #diag(bmat) = zeros(d)
    flatten(LowerTriangular(bmat[2:d, 1:d-1]))
end


"""
Given a vector of canonical partial correlations (taken from the
upper triangle by rows), return a vector of correlations in the
same format.
Makes use of the relation between partial correlations
r_{ij;L} = \sqrt{(1 - r_{ik;L}^2)(1 - r_{jk;L}^2)} r_{ij;kL} + r_{ik;L} r_{jk;L}
"""
function cpc_to_corr(x::Vector, d::Int)
    #d = Int(.5 + .5 * sqrt(1 + 8 * length(x)))
    #d = .5sqrt(1 + 8length(x)) - .5
    #d = Int(d)
    L = hcat(vcat(zeros(d-1)', LowerTriangular(x)), zeros(d))
    #L = LowerTriangular(x)
    LL = zeros(similar(L))
    LL[:, 1] = L[:, 1]
    for j in 2:d
        for i in j+1:d
            rho = L[i, j]
            for k in j-1:-1:1
                rho = rho * sqrt((1 - L[i, k]^2) * (1 - L[j, k]^2)) + L[i, k] * L[j, k]
            end
            LL[i, j] = rho
        end
    end
    LowerTriangular(LL)
end


"""
Random draw from the LKJ distribution with parameter eta and dimension d.
"""
function draw_lkj(η, d::Int)
    β = lkj_to_beta_pars(η, d)
    cpcs = zeros(length(β))
    for i in 1:length(β)
        cpcs[i] = 2 * rand(Distributions.Beta(β[i], β[i])) - 1
    end
    LL = cpc_to_corr(cpcs, d)
    LL + transpose(LL) + diagm(ones(d))
end


"""
E[log |A|] where
A ~ LKJ(eta) of dimension d
"""
function logdetjac_lkj(η, d::Int)
    betas = lkj_to_beta_pars(eta, d)
    sum(digamma(betas) - digamma(2 * betas))
end


"""
Compute entropy for LKJ distribution by Dirichelet entropy
"""
function lkj_entropy(η, d::Int)
    betas = lkj_to_beta_pars(η, d)
    alpha = hcat(betas, betas)   # copmute Dirichelet entropy
    alpha0 = sum(alpha, 2)
    H = sum(lgamma(alpha), 2) - lgamma(sum(alpha, 2))
    H += (alpha0 - size(alpha)[2]) .* digamma(alpha0)
    H -= sum((alpha - 1) .* digamma(alpha), 2)
    sum(H)
end
