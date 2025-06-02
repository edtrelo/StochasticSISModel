include("utilities.jl")
using Plots
using QuadGK
using Distributions
using Random
using Optim
using FiniteDiff 
using Bootstrap
using Statistics
Random.seed!(123)

# Caso 1: Usando datos sintéticos

# Parámetros para la simulaciñón
M = 10000
dt = 1/M
N = 763
β = 0.0013925
γ = 0.476
σ = 0.0005
p = (β, γ, σ, N)
I₀ = 10

# Queremos ver cómo mejora la estimación según usemos más datos
nDias = [7, 14, 28, 50]
dW = rand(Normal(0, 1), trunc(Int, nDias[end]/dt))
X = solveSIS(p, nDias[end], I₀, dW; dt = dt)
ts = collect(0:dt:nDias[end])
# Trayectoria completa
tray = [X ts]

obs = tray[(tray[1:end, 2].%1 .== 0), 1]
obs_ts = tray[(tray[1:end, 2].%1 .== 0), 2]



N = 200
I₀ = 0.0912*N*10^(-3)
obs = [0.1397, 0.1966, 0.2530, 0.3476, 0.4575, 0.6142, 0.7986, 0.8176, 0.6257, 0.8376, 0.3979]*10^(-3)
obs = obs.*N
obs_ts = collect(1:length(obs))
γ = 31.39

γ = 1/2
sss = SIS_sigma_by_qv(obs, obs_ts)
b = 0
for i in 1:(length(obs)-1)
    b += (obs[i+1] - obs[i] + γ*obs[i])/(obs[i]*(N-obs[i]))
end
b = b/(length(obs)-1)

S = 1000
M = 10
h = 1/M 

x0 = [b, γ, sss]


α = 10^(-4) 
xnew = ∇log_likelihood(x0, DZ, M, S, h, obs, N, obs_ts)

# H*∇log_likelihood(x0, DZ, M, S, h, obs, N, obs_ts)
u = optimize(f, [α])
u.minimizer
# inv(FiniteDiff.finite_difference_hessian(F, x0))
# FiniteDiff.finite_difference_jacobian(∇F, x0)

lower = [0, 365/28, 0]
upper = [Inf, 365/1, Inf]

DZ = rand(Normal(0, 1), (S, M+1))
oldlike = F(x0)
#sol = optimize(x -> -F(x), lower, upper, x0, Fminbox(BFGS()); inplace = false)
sol = optimize(x -> -F(x), x0, BFGS())
x0 = sol.minimizer


# x0 = [b, γ, sss]
# oldlike = F(x0)
# for i in 1:1000
#     #xnew = ∇log_likelihood(x0, DZ, M, S, h, obs, N, obs_ts
#     try 
#         xnew = ∇F(x0)
#         H = FiniteDiff.finite_difference_hessian(x-> -F(x), x0)
#     catch 
#         α = α/10 

#     x0 += α*inv(H)*xnew
# end

F(x0)>oldlike
x0

T = 50
p = plot(obs_ts, obs)
ts = collect(0:0.001:T)
sim = zeros(100, length(ts))
for i in 1:100
    sim[i, 1:end] = solveSIS(vcat(x0, N), T, I₀)
end
plot!(ts, mean(sim, dims = 1)[1, 1:end])



CIL = zeros(3, length(ts))


for i in 1:length(ts)
    datai = sim[1:end, i]
    CIL[2, i] = mean(datai)
    CIL[1, i] = quantile(datai, 0.05)
    CIL[3, i] = quantile(datai, 0.95)
    # bs = bootstrap(mean, datai, BasicSampling(100))
    # cil, = Bootstrap.confint(bs, BasicConfInt(0.95))
    # CIL[1:end, i] .= cil
end

CIL

m = CIL[2, 1:end]
low = CIL[1, 1:end]
upper = CIL[3, 1:end]
plot(obs_ts, obs, seriestype = :scatter)
plot!(ts, m, ribbon = (m.-low, upper.-m))


plot!(ts, m)
plot!(ts, low)
plot!(ts, upper)