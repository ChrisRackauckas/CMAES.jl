using Plots, CMAES, CMA, BlackBoxOptim; plotly()

function get_progress(f, eval_seq)
    progress = Float64[]
    for maxevals in eval_seq
        try run(`bash -c 'rm *.jld *.dat'`) end
        push!(progress, f(maxevals))
    end
    return progress
end

function get_progress(f, eval_seq, nprog)
    global prog, q50
    prog = hcat([get_progress(f, eval_seq) for i in 1:nprog]...)
    q10 = [quantile(prog[i, :], 0.1) for i in 1:size(prog, 1)]
    q50 = [quantile(prog[i, :], 0.5) for i in 1:size(prog, 1)]
    q90 = [quantile(prog[i, :], 0.9) for i in 1:size(prog, 1)]
    return q10, q50, q90
end

function plot_progress(f, eval_seq, nprog; label = "")
    q10, q50, q90 = get_progress(f, eval_seq, nprog)
    plot!(q50; ribbon = (q10, q90), label = label)
    return nothing
end

@everywhere rastrigin(x) = 10length(x) + sum(x.^2 .- 10 .* cos.(2π .* x))

N, maxfevals, nprog, eval_seq = 10, 1000, 10, 100:100:10000
x0, σ0, lo, hi = 0.3ones(N), 0.2, fill(-5.12, N), fill(5.12, N)

plot();

plot_progress(eval_seq, nprog; label = "CMAES") do maxfevals
    xmin, fmin = CMAES.minimize(rastrigin, x0, σ0, lo, hi; maxfevals = maxfevals)
    return fmin
end

plot_progress(eval_seq, nprog; label = "CMA") do maxfevals
    xmin, fmin = CMA.minimize(rastrigin, x0, σ0, lo, hi; maxfevals = maxfevals)
    return fmin
end

plot_progress(eval_seq, nprog; label = "BBO") do maxfevals
    res = bboptimize(rastrigin; SearchRange = (-5.12, 5.12), NumDimensions = N, Method = :dxnes, MaxFuncEvals = maxfevals, ini_x = x0, ini_sigma = σ0)
    xmin, fmin = best_candidate(res), best_fitness(res)
    return fmin
end

savefig("CMAES Benchmark.html")