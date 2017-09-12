using CMAES

@everywhere rastrigin(x) = 10length(x) + sum(x.^2 - 10cos(2π * x))

N = 1000
x0, σ0, lo, hi = 0.3ones(N), 0.2, fill(-5.12, N), fill(5.12, N)
xmin, fmin = CMAES.minimize(rastrigin, x0, σ0, lo, hi; maxfevals = 10000)

# using CMA
# xmin, fmin = CMA.minimize(rastrigin, x0, σ0, lo, hi; maxfevals = 10000) #, pool = [2:2:20;])
#
# using BlackBoxOptim
# res = bboptimize(rastrigin; SearchRange = (-5.12, 5.12), NumDimensions = N, Method = :dxnes, MaxFuncEvals = 10000, ini_x = x0, ini_sigma = σ0)
# xmin, fmin = best_candidate(res), best_fitness(res)
