__precompile__(true)

module CMAES

using JLD, Utils

mutable struct CMAESOpt
    # fixed hyper-parameters
    f::Function
    N::Int
    σ0::Float64
    lo::Vector{Float64}
    hi::Vector{Float64}
    penalty::Bool
    # Strategy parameter setting: Selection
    λ::Int
    μ::Int
    w::Vector{Float64}
    μeff::Float64
    # Strategy parameter setting: Adaptation
    σ::Float64
    cc::Float64
    cσ::Float64
    c1::Float64
    cμ::Float64
    dσ::Float64
    # Initialize dynamic (internal) strategy parameters and constants
    x̄::Vector{Float64}
    pc::Vector{Float64}
    pσ::Vector{Float64}
    D::Vector{Float64}
    B::Matrix{Float64}
    BD::Matrix{Float64}
    C::Matrix{Float64}
    χₙ::Float64
    arx::Matrix{Float64}
    ary::Matrix{Float64}
    arz::Matrix{Float64}
    arfitness::Vector{Float64}
    arindex::Vector{Int}
    xmin::Vector{Float64}
    fmin::Float64
    fmins::Vector{Float64}
    # report
    last_report_time::Float64
    file::String
end

function CMAESOpt(f, x0, σ0, lo, hi; λ = 0, penalty = false)
    N, x̄, xmin, fmin, σ = length(x0), x0, x0, f(x0), σ0
    #########
    # Strategy parameter setting: Selection
    λ = λ == 0 ? Int(4 + 3log(N)) : λ
    μ = λ ÷ 2                    # number of parents/points for recombination
    w = log(μ + 1/2) .- log.(1:μ) # μXone array for weighted recombination
    normalize!(w, 1)             # normalize recombination w array
    μeff = 1 / sum(abs2, w)     # variance-effectiveness of sum w_i x_i
    #########
    # Strategy parameter setting: Adaptation
    cc = 4 / (N + 4) # time constant for cumulation for C
    cσ = (μeff + 2) / (N + μeff + 3)  # t-const for cumulation for σ control
    c1 = 2 / ((N + 1.3)^2 + μeff)    # learning rate for rank-one update of C
    cμ = min(1 - c1, 2(μeff - 2 + 1 / μeff) / ((N + 2)^2 + μeff) )   # and for rank-μ update
    dσ = 1 + 2 * max(0, sqrt((μeff - 1) / (N + 1)) - 1) + cσ # damping for σ, usually close to 1
    #########
    # Initialize dynamic (internal) strategy parameters and constants
    pc = zeros(N); pσ = zeros(N)   # evolution paths for C and σ
    D = fill(σ, N)                # diagonal matrix D defines the scaling
    B = eye(N, N)                 # B defines the coordinate system
    BD = B .* reshape(D, 1, N)    # B*D for speed up only
    C = diagm(D.^2)                    # covariance matrix == BD*(BD)'
    χₙ = sqrt(N) * (1 -1 / 4N + 1 / 21N^2)  # expectation of  ||N(0,I)|| == norm(randn(N,1))
    # init a few things
    arx, ary, arz = zeros(N, λ), zeros(N, λ), zeros(N, λ)
    arfitness = zeros(λ);  arindex = zeros(λ)
    @printf("%i-%i CMA-ES\n", λ, μ)
    return CMAESOpt(f, N, σ0, lo, hi, penalty,
                    λ, μ, w, μeff,
                    σ, cc, cσ, c1, cμ, dσ,
                    x̄, pc, pσ, D, B, BD, C, χₙ,
                    arx, ary, arz, arfitness, arindex,
                    xmin, fmin, [],
                    time(), get(ENV, "CMAES_LOGFILE", "") * "_CMAES.jld")
end

@replace function update_candidates!(opt::CMAESOpt, pool)
    # Generate and evaluate λ offspring
    randn!(arz) # resample
    ary .= BD * arz
    arx .= x̄ .+ σ .* ary
    arfitness .= pmap(WorkerPool(pool), f, [arx[:, k] for k in 1:λ])
    if penalty arfitness .+= vec(boundpenalty(arx, lo, hi)) end
    # Sort by fitness and compute weighted mean into x̄
    sortperm!(arindex, arfitness)
    arfitness = arfitness[arindex]  # minimization
    if arfitness[1] < fmin
        xmin, fmin = arx[:, arindex[1]], arfitness[1]
    end
    push!(fmins, arfitness[1])
end

@replace function update_parameters!(opt::CMAESOpt, iter)
    # Calculate new x̄, this is selection and recombination
    x̄old = copy(x̄) # for speed up of Eq. (2) and (3)
    x̄ = arx[:, arindex[1:μ]] * w
    z̄ = arz[:, arindex[1:μ]] * w # ==D^-1*B'*(x̄-x̄old)/σ
    # Cumulation: Update evolution paths
    BLAS.gemv!('N', sqrt(cσ * (2 - cσ) * μeff), B, z̄, 1 - cσ, pσ)
    #  i.e. pσ = (1 - cσ) * pσ + sqrt(cσ * (2 - cσ) * μeff) * (B * z̄) # Eq. (4)
    hsig = norm(pσ) / sqrt(1 - (1 - cσ)^2iter) / χₙ < 1.4 + 2 / (N + 1)
    BLAS.scale!(pc, 1 - cc); BLAS.axpy!(hsig * sqrt(cc * (2 - cc) * μeff) / σ, x̄ - x̄old, pc)
    # i.e. pc = (1 - cc) * pc + (hsig * sqrt(cc * (2 - cc) * μeff) / σ) * (x̄ - x̄old)
    # Adapt covariance matrix C
    scale!(C, (1 - c1 - cμ + (1 - hsig) * c1 * cc * (2 - cc))) # discard old C
    BLAS.syr!('U', c1, pc, C) # rank 1 update C += c1 * pc * pc'
    artmp = ary[:, arindex[1:μ]]  # μ difference vectors
    artmp = (artmp .* reshape(w, 1, μ)) * artmp.'
    BLAS.axpy!(cμ, artmp, C)
    # Adapt step size σ
    σ *= exp((norm(pσ) / χₙ - 1) * cσ / dσ)  #Eq. (5)
    # Update B and D from C
    # if counteval - eigeneval > λ / (c1 + cμ) / N / 10  # to achieve O(N^2)
    if mod(iter, 1 / (c1 + cμ) / N / 10) < 1
        (D, B) = eig(Symmetric(C, :U)) # eigen decomposition, B==normalized eigenvectors
        D .= sqrt.(D)                   # D contains standard deviations now
        BD .= B .* reshape(D, 1, N)     # O(n^2)
    end
end

@replace function terminate(opt::CMAESOpt)
    #Stop conditions:
    # TolHistFun = 1e-12:  the range of the best function values during the last
    # 10 + ⌈30 D / λ⌉ iterations is smaller than TolHistFun
    lastiter = 10 + round(Int, 30N/λ)
    PTP::Float64 = parse(get(ENV, "CMAES_PTP", "1e-12"))
    TERMINATE_ON_FLAT_FITNESS::Bool = parse(get(ENV, "CMAES_TERMINATE_ON_FLAT_FITNESS", "false"))
    if arfitness[1] == arfitness[ceil(Int, 0.7 * λ)]
        σ *= exp(0.2 + cσ / dσ)
        println("warning: flat fitness, consider reformulating the objective")
    end
    length(fmins) > lastiter && TERMINATE_ON_FLAT_FITNESS && (arfitness[1] == arfitness[ceil(Int, 0.7 * λ)]) ||
    length(fmins) > lastiter && ptp(fmins[end-lastiter:end]) < PTP ||
    # EqualFunVals:  in more than 1/3rd of the last D iterations the objective
    # function value of the best and the k-th best solution are identical
    # arfitness[1] == arfitness[ceil(Int, 0.7 * λ)] ||
    # TolX:  all components of pc and all square roots of diagonal components of C,
    # multiplied by σ / σ0 , are smaller than TolX.
    max(sqrt(maximum(abs, diag(C))), maximum(abs, pc)) * σ / σ0 < 1e-12 ||
    # TolUpSigma = 1e-12: σ / σ0 >= 1e20 * maximum(D)
    σ / σ0 > 1e20 * maximum(D) ||
    # Stagnation: TODO
    # ConditionCov:  the condition number of C exceeds 1e14
    maximum(D) > 1e7 * minimum(D)
    # NoEffectAxis: TODO
    # NoEffectCoor: TODO
end

@replace function restart(opt::CMAESOpt)
    @printf("restarting...\n")
    optnew = CMAESOpt(f, sample(lo, hi), σ0, lo, hi; :λ => 2λ)
    optnew.xmin, optnew.fmin = xmin, fmin
    return optnew
end

@replace function trace_state(opt::CMAESOpt, iter, fcount)
    elapsed_time = time() - last_report_time
    # write to file
    !startswith(file, "_") && JLD.jldopen(file, "w") do fid
        for s in fieldnames(opt)
            s != :f && write(fid, string(s), getfield(opt, s))
        end
    end
    # Display some information every iteration
    @printf("time: %s iter: %d  elapsed-time: %.2f fcount: %d  fval: %2.2e  fmin: %2.2e  axis-ratio: %2.2e \n",
            now(), iter, elapsed_time, fcount, arfitness[1], fmin, maximum(D) / minimum(D) )
    last_report_time = time()
    return nothing
end

function cmaes(f::Function, x0, σ0, lo, hi; pool = workers(), maxfevals = 0, o...)
    maxfevals = (maxfevals == 0) ? 1e3N^2 : maxfevals
    opt = CMAESOpt(f, x0, σ0, lo, hi; o...)
    if isfile(opt.file)
        d = JLD.load(opt.file)
        for s in keys(d)
            hasfield(opt, Symbol(s)) && setfield!(opt, Symbol(s), d[s])
        end
    end
    fcount = iter = 0
    while fcount < maxfevals
        iter += 1; fcount += opt.λ
        update_candidates!(opt, pool)
        update_parameters!(opt, iter)
        trace_state(opt, iter, fcount)
        terminate(opt) && break
        # if terminate(opt) opt, iter = restart(opt), 0 end
    end
    return opt.xmin, opt.fmin
end

minibatch(x, b) = [x[i:min(end, i+b-1)] for i in 1:b:max(1, length(x)-b+1)]

function optimize(f, x0, σ0, lo, hi; pool = workers(), restarts = 1, λ = 0, o...)
    λ = λ == 0 ? Int(4 + 3log(length(x0))) : λ
    pop_pools = minibatch(pool, λ) # population pools
    restarts = max(restarts, length(pool) ÷ λ)
    head_pool = first.(pop_pools)
    fun = i -> begin
        x0 = (i == 1 || rand() < 0.5) ? x0 : sample(lo, hi)
        idx = findfirst(head_pool, myid())
        cmaes(f, x0, σ0, lo, hi; pool = pop_pools[idx], λ = λ, o...)
    end
    res = pmap(WorkerPool(head_pool), fun, 1:restarts)
    x, y = hcat(res...)
    fmin, index = findmin(y)
    xmin = x[:, index]
    return xmin, fmin
end

minimize = optimize

function maximize(f, args...; kwargs...)
    xmin, fmin = optimize(x -> -f(x), args...; kwargs...)
    return xmin, -fmin
end

sample(lo, hi) = lo .+ rand(size(lo)) .* (hi .- lo)

function boundpenalty(x, lo, hi)
    σ = (hi .- lo) ./ 100
    sum(exp.(max.(x .- hi, zero(x)) ./ σ) .- 1, 1) +
    sum(exp.(max.(lo .- x, zero(x)) ./ σ) .- 1, 1)
end

end

# module CMAES
#
# using JLD
#
# function cmaes(f::Function, x0, σ; pool = workers(), λ = 0, maxfevals = 0, stopΔfitness = 1e-12)
#     x̄, xmin, fmin = x0, x0, f(x0)
#     #########
#     # Strategy parameter setting: Selection
#     N = length(x0)
#     λ = λ == 0 ? Int(4 + 3log(N)) : λ
#     maxfevals = maxfevals == 0 ? 1e3N^2 : maxfevals
#     μ = λ ÷ 2                    # number of parents/points for recombination
#     w = log(μ + 1/2) .- log(1:μ) # μXone array for weighted recombination
#     normalize!(w, 1)             # normalize recombination w array
#     μeff = 1 / sum(abs2, w)     # variance-effectiveness of sum w_i x_i
#     #########
#     # Strategy parameter setting: Adaptation
#     cc = 4 / (N + 4) # time constant for cumulation for C
#     cσ = (μeff + 2) / (N + μeff + 3)  # t-const for cumulation for σ control
#     c1 = 2 / ((N + 1.3)^2 + μeff)    # learning rate for rank-one update of C
#     cμ = min(1 - c1, 2(μeff - 2 + 1 / μeff) / ((N + 2)^2 + μeff) )   # and for rank-μ update
#     dσ = 1 + 2 * max(0, sqrt((μeff - 1) / (N + 1)) - 1) + cσ # damping for σ, usually close to 1
#     #########
#     # Initialize dynamic (internal) strategy parameters and constants
#     pc = zeros(N); pσ = zeros(N)   # evolution paths for C and σ
#     D = fill(σ, N)                # diagonal matrix D defines the scaling
#     B = eye(N, N)                 # B defines the coordinate system
#     BD = B .* reshape(D, 1, N)    # B*D for speed up only
#     C = diagm(D.^2)                    # covariance matrix == BD*(BD)'
#     χₙ = sqrt(N) * (1 -1 / 4N + 1 / 21N^2)  # expectation of  ||N(0,I)|| == norm(randn(N,1))
#     # init a few things
#     arx, ary, arz = zeros(N, λ), zeros(N, λ), zeros(N, λ)
#     arfitness = zeros(λ);  arindex = zeros(λ)
#     @printf("%i-%i CMA-ES\n", λ, μ);
#     # -------------------- Generation Loop --------------------------------
#     eigeneval = counteval = iter = 0  # the next 40 lines contain the 20 lines of interesting code
#     while counteval < maxfevals
#         iter += 1
#         # Generate and evaluate λ offspring
#         arz = randn(size(arz)) # resample
#         ary = BD * arz
#         arx = x̄ .+ σ .* ary
#         counteval += λ
#         arfitness .= pmap(WorkerPool(pool), k -> f(arx[:, k]), 1:size(arx, 2))
#         # Sort by fitness and compute weighted mean into x̄
#         arindex = sortperm(arfitness)
#         arfitness = arfitness[arindex]  # minimization
#         # Calculate new x̄, this is selection and recombination
#         x̄old = x̄ # for speed up of Eq. (2) and (3)
#         x̄ = arx[:, arindex[1:μ]] * w
#         z̄ = arz[:, arindex[1:μ]] * w # ==D^-1*B'*(x̄-x̄old)/σ
#         # Cumulation: Update evolution paths
#         pσ = (1 - cσ) * pσ + sqrt(cσ * (2 - cσ) * μeff) * (B * z̄)          # Eq. (4)
#         hsig = norm(pσ) / sqrt(1 - (1 - cσ)^2iter) / χₙ < 1.4 + 2 / (N + 1)
#         pc = (1 - cc) * pc + hsig * (sqrt(cc * (2 - cc) * μeff) / σ) * (x̄ - x̄old)
#         # Adapt covariance matrix C
#         scale!(C, (1 - c1 - cμ + (1 - hsig) * c1 * cc * (2 - cc))) # discard old C
#         BLAS.syr!('U', c1, pc, C) # rank 1 update C += c1 * pc * pc'
#         C1 = copy(C); C2 = copy(C)
#         artmp = ary[:, arindex[1:μ]]  # μ difference vectors
#         artmp = (artmp .* reshape(w, 1, μ)) * artmp.'
#         BLAS.axpy!(cμ, artmp, C1)
#         # Adapt step size σ
#         σ *= exp((norm(pσ) / χₙ - 1) * cσ / dσ)  #Eq. (5)
#         # Update B and D from C
#         # if counteval - eigeneval > λ/(c1+cμ)/N/10  # to achieve O(N^2)
#         if counteval - eigeneval > λ / (c1 + cμ) / N / 10
#             eigeneval = counteval
#             (D, B) = eig(Symmetric(C, :U)) # eigen decomposition, B==normalized eigenvectors
#             D .= sqrt.(D)                   # D contains standard deviations now
#             BD .= B .* reshape(D, 1, N)     # O(n^2)
#         end
#         # write to file
#         if arfitness[1] < fmin xmin, fmin = arx[:, arindex[1]], arfitness[1] end
#         JLD.write(joinpath(tempdir(), "CMAES.jld"), "x", xmin, "y", fmin)
#         #Stop conditions:
#         # 1. break if fitness is good enough or condition exceeds 1e14, better termination methods are advisable
#         maximum(D) > 1e7 * minimum(D) && (x̄ = restart(lo, hi))
#         # 2. break if fitness is flat
#         arfitness[1] == arfitness[round(Int, 0.7 * λ)] && (x̄ = restart(lo, hi))
#         # Display some information every iteration
#         @printf("iter: %d    fcount: %d    fval: %2.2e    axis-ratio: %2.2e \n",
#                 iter, counteval, arfitness[1], maximum(D) / minimum(D) )
#     end # while, end generation loop
#     return xmin, fmin
# end
#
# minibatch(x, b) = [x[i:min(end, i+b-1)] for i in 1:b:length(x)]
#
# function optimize(f, x0, σ0, lo, hi; pool = workers(), λ = 0, restarts = 1, o...)
#     λ = λ == 0 ? Int(4 + 3log(length(x0))) : λ
#     pop_pools = minibatch(pool, λ) # population pools
#     head_pool = pool[1:λ:end]
#     fun = i -> begin
#         x0 = (i == 1 || rand() < 0.5) ? x0 : lo .+ rand(size(lo)) .* (hi .- lo)
#         idx = findfirst(head_pool, myid())
#         cmaes(f, x0, σ0; pool = pop_pools[idx], λ = λ, o...)
#     end
#     res = map(fun, 1:restarts) #
#     # res = pmap(WorkerPool(head_pool), fun, 1:restarts)
#     x, y = hcat(res...)
#     fmin, index = findmin(y)
#     xmin = x[:, index]
#     return xmin, fmin
# end
#
# minimize = optimize
#
# function maximize(f, args...; kwargs...)
#     xmin, fmin = optimize(x -> -f(x), args...; kwargs...)
#     return xmin, -fmin
# end
#
# end
