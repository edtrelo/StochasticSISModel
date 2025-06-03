include("utilities.jl")
using Plots
using QuadGK
using Distributions
using Random
using Optim
using FiniteDiff 
using Bootstrap
using Statistics
using StatsPlots
using HypothesisTests
using LaTeXStrings
Random.seed!(1)
# Observaciones
N = 200
I₀ = 0.0912*N*10^(-3)
obs = [0.1397, 0.1966, 0.2530, 0.3476, 0.4575, 0.6142, 0.7986, 0.8176, 0.6257, 0.8376, 0.3979]*10^(-3)
obs = obs.*N
obs_ts = collect(1:length(obs))
# 3 a 6 semana
γ₀ = 52/4.5
# estimación de σ₀
σ₀ = SIS_sigma_by_qv(vcat(I₀, obs), vcat(0, obs_ts))
# obtención de β₀
β₀ = (obs[1] - I₀ + γ₀*I₀)/(I₀*(N-I₀))
for i in 1:(length(obs)-1)
    β₀ += (obs[i+1] - obs[i] + γ₀*obs[i])/(obs[i]*(N-obs[i]))
end
β₀ = β₀/(length(obs))
# Valores fijos de la simulación de la verosimiliad
S = 2500
M = 10
h = 1/M 
# Valor inicial para el algoritmo de optimización
θ₀ = [β₀, γ₀, σ₀]
# Cubo para la constrained optimization
lower = [0, 52/6, 0]
upper = [Inf, 52/3, Inf]
# errores fijos para toooooda la optimización
errores_sample = [rand(Normal(0, 1), (S, M+1)) for _ in 1:(length(obs)-1)]
DZ = errores_sample
# Optimización 
# F es la función de log-verosimilitud
sol = optimize(x -> -F(x), lower, upper, θ₀, Fminbox(BFGS()); inplace = false)
#sol = optimize(x -> -F(x), x0, BFGS())
# Estimador máximo verosímil simulado
θmle = sol.minimizer
# Simulación de n trayectorias
# Horizonte temporal final
T = obs_ts[end]
# tamaño de paso para las simulaciones
dt = 0.001
ts = collect(0:dt:T)
# Aquí se guardarán las simulaciones
n_simulaciones = 500
sim = zeros(n_simulaciones, length(ts))
# Simulación de trayectorias con los parámetros de MV
for i in 1:n_simulaciones
    sim[i, 1:end] = solveSIS(vcat(θmle, N), T, I₀; dt = dt)
end
# Ahora 
intervals = zeros(3, length(ts))
# Obtención de la media y los quantiles 0.05 y 0.95
for i in 1:length(ts)
    datai = sim[1:end, i]
    intervals[2, i] = mean(datai)
    intervals[1, i] = quantile(datai, 0.05)
    intervals[3, i] = quantile(datai, 0.95)
end
# Gráficas 🦖
m = intervals[2, 1:end] # mean para Xₜ
low = intervals[1, 1:end] # cuantil 0.05 para Xₜ
upper = intervals[3, 1:end] # cuantil 0.95 
plot(ts, m, ribbon = (m.-low, upper.-m), label = "Media de la trayectoria", lw = 2,
     title = "Simulación de trayectorias con "*L"\theta_{MVS}", 
     xlabel = "Tiempo (en años)", ylabel = "Número de infectados (en millones)")
plot!(ts, low, color = :gray, label = "Cuantiles p = 0.05, 0.95")
plot!(ts, upper, color = :gray, label = false)
plot_sims = plot!(obs_ts, obs, seriestype = :scatter, label = "Observaciones", dpi = 200)
savefig(plot_sims, "D:/Edgar Trejo/Universidad/Proyecto/data/images/sims_theta_MVS.png")
# Validación del modelo :)
# Obtención de valores estimados
βmle, γmle, σmle = θmle
# Definición de h(x) y su inversa
# func_h(x) = (1/(N*σ₀))*log(x/(N-x))
inv_h(x) = (N*exp(σmle*N*x))/(1+exp(σmle*N*x))
# Funciones de deriva y con los parámetros estimados
func_b(x) = βmle*x*(N-x)-γmle*x
func_s(x) = σmle*x*(N-x)
func_ds(x) = σmle*(N-2x)
# Definición de la función μ en la transformación de Lamperti
func_μ(x) = func_b(inv_h(x))/func_s(inv_h(x)) - (1/2)*func_ds(inv_h(x))
# Cálculo de los residuales
n = length(obs)
residuals = zeros(n-1)
for i in 2:n
    residuals[i-1] = obs[i] - obs[i-1] - func_μ(obs[i-1])
end
# Análsis de los residuales ¿Cumplen con la normalidad?
# qq-plot: los puntos deben estar en la identidad 
plot_qq = plot(qqnorm(residuals), xlabel = "Datos", ylabel = "Normales", 
title = "Comparación de residuales", dpi = 200)
savefig(plot_qq, "D:/Edgar Trejo/Universidad/Proyecto/data/images/qqplot_info_incompleta.png")
# prueba estadística
prueba = ShapiroWilkTest(residuals)

