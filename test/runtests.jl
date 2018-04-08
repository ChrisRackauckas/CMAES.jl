@everywhere rastrigin(x) = 10length(x) + sum(x.^2 .- 10 .* cos.(2π .* x))

N, maxfevals = 100, 10000
x0, σ0, lo, hi = 0.3ones(N), 0.2, fill(-5.12, N), fill(5.12, N)

using CMAES
xmin, fmin = CMAES.minimize(rastrigin, x0, σ0, lo, hi; maxfevals = maxfevals)

using CMA
xmin, fmin = CMA.minimize(rastrigin, x0, σ0, lo, hi; maxfevals = maxfevals)

using BlackBoxOptim
res = bboptimize(rastrigin; SearchRange = (-5.12, 5.12), NumDimensions = N, Method = :dxnes, MaxFuncEvals = maxfevals, ini_x = x0, ini_sigma = σ0)
xmin, fmin = best_candidate(res), best_fitness(res)
