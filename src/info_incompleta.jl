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
using DataFrames
#using DelimitedFiles
using CSV
using Tables
Random.seed!(1)
# Observaciones
N = 1
I₀ = 0.0912*10^(-3)
obs = [0.1397, 0.1966, 0.2530, 0.3476, 0.4575, 0.6142, 0.7986, 0.8176, 0.6257, 0.8376, 0.3979]*10^(-3)
#obs = obs.
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
# número de estimaciones que vamos a realizar
n_est = 500
estimadores = zeros(3, n_est)
DZ = zeros()
descartados = 0
# Ya no es necesario correr esto
# Las muestras de los estimadores ya fueron obtenidos y se encuentra en
# ~ data/samples/sample_estimadores_info_incompleta.csv

# for i in 1:n_est
#     # errores fijos para toooooda la optimización
#     try
#         errores_sample = [rand(Normal(0, 1), (S, M+1)) for _ in 1:(length(obs)-1)]
#         DZ = errores_sample
#         # Optimización 
#         # F es la función de log-verosimilitud
#         sol = optimize(x -> -F(x), lower, upper, θ₀, Fminbox(BFGS()); inplace = false)
#         #sol = optimize(x -> -F(x), x0, BFGS())
#         # Estimador máximo verosímil simulado
#         θmle = sol.minimizer
#         estimadores[1:end, i] = θmle
#     catch
#         descartados += 1
#     end
# end
# descartados # 22

estimadores_buenos = estimadores[1:end, estimadores[1, 1:end].!=0]
# CSV.write("sample_estimadores_info_incompleta.csv",  Tables.table(transpose(estimadores_buenos)), 
#     header = ["beta","gamma","sigma"])
estimadores_buenos = CSV.read("D:/Edgar Trejo/Universidad/Proyecto/data/samples/sample_estimadores_info_incompleta.csv",
DataFrame)

sample_beta = estimadores_buenos[!, "beta"]
sample_gamma = estimadores_buenos[!, "gamma"]
sample_sigma = estimadores_buenos[!, "sigma"]

hb = histogram(sample_beta, normalize = :pdf, xlabel = L"\hat{\beta}_{MVS}", label = false)
hg = histogram(log2.(sample_gamma), normalize = :pdf, xlabel = L"\log_2(\hat{\gamma}_{MVS})", label = false)
hs = histogram(sample_sigma, normalize = :pdf, xlabel = L"\hat{\sigma}_{MVS}", label = false)
hists = plot(hb, hg, hs, layout = (3,1), dpi = 200)
savefig(hists, "D:/Edgar Trejo/Universidad/Proyecto/data/images/samples_MVS.png")

# intervalos de confianza
ibeta = (quantile(sample_beta, 0.05/2), quantile(sample_beta, 1-0.05/2))
igamma = (quantile(sample_gamma, 0.05/2), quantile(sample_gamma, 1-0.05/2))
isigma = (quantile(sample_sigma, 0.05/2), quantile(sample_sigma, 1-0.05/2))

βmvs, γmvs, σmvs = mode(sample_beta), mode(sample_gamma), mode(sample_sigma)
mean(estimadores_buenos, dims = 2)

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
    sim[i, 1:end] = solveSIS([βmvs, γmvs, σmvs, N], T, I₀; dt = dt)
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
     title = "Simulación de trayectorias con las modas marginales\ndel estimador "*L"\theta_{MVS}", 
     xlabel = "Tiempo (en años)", ylabel = "Proporción de infectados")
plot!(ts, low, color = :gray, label = "Cuantiles p = 0.05, 0.95")
plot!(ts, upper, color = :gray, label = false)
plot_sims = plot!(obs_ts, obs, seriestype = :scatter, label = "Observaciones", dpi = 200)
savefig(plot_sims, "D:/Edgar Trejo/Universidad/Proyecto/data/images/sims_theta_MVS.png")
# Validación del modelo :)
# Obtención de valores estimados
#βmle, γmle, σmle = θmle
# Definición de h(x) y su inversa
# func_h(x) = (1/(N*σ₀))*log(x/(N-x))
inv_h(x) = (N*exp(σmvs*N*x))/(1+exp(σmvs*N*x))
# Funciones de deriva y con los parámetros estimados
func_b(x) = βmvs*x*(N-x)-γmvs*x
func_s(x) = σmvs*x*(N-x)
func_ds(x) = σmvs*(N-2x)
# Definición de la función μ en la transformación de Lamperti
func_μ(x) = func_b(inv_h(x))/func_s(inv_h(x)) - (1/2)*func_ds(inv_h(x))
# Cálculo de los residuales
n = length(obs)+1
residuals = zeros(n-1)
obs_plus = vcat(I₀, obs)
for i in 2:n
    residuals[i-1] = obs_plus[i] - obs_plus[i-1] - func_μ(obs_plus[i-1])
end
# Análsis de los residuales ¿Cumplen con la normalidad?
# qq-plot: los puntos deben estar en la identidad 
plot_qq = plot(qqnorm(residuals), xlabel = "Datos", ylabel = "Normales", 
title = "Comparación de residuales", dpi = 200)
savefig(plot_qq, "D:/Edgar Trejo/Universidad/Proyecto/data/images/qqplot_info_incompleta.png")
# prueba estadística
prueba = ShapiroWilkTest(residuals)

interp_means = linear_interpolation(ts, m)
sim_means = interp_means.(obs_ts)

mae(obs, sim_means)
mse(obs, sim_means)

# caso pre-covid
N = 1
I₀ = 0.0912*10^(-3)
obs = [0.1397, 0.1966, 0.2530, 0.3476, 0.4575, 0.6142, 0.7986, 0.8176]*10^(-3)
#obs = obs.
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
# número de estimaciones que vamos a realizar
n_est = 100
estimadores_precovid = zeros(3, n_est)
DZ =zeros()
descartados = 0

for i in 1:n_est
    # errores fijos para toooooda la optimización
    try
        errores_sample = [rand(Normal(0, 1), (S, M+1)) for _ in 1:(length(obs)-1)]
        DZ = errores_sample
        # Optimización 
        # F es la función de log-verosimilitud
        sol = optimize(x -> -F(x), lower, upper, θ₀, Fminbox(BFGS()); inplace = false)
        #sol = optimize(x -> -F(x), x0, BFGS())
        # Estimador máximo verosímil simulado
        θmle = sol.minimizer
        estimadores_precovid[1:end, i] = θmle
    catch
        descartados += 1
    end
end

descartados # 22

estimadores_buenos = estimadores_precovid[1:end, estimadores_precovid[1, 1:end].!=0]
# CSV.write("sample_estimadores_info_incompleta_precovid.csv",  Tables.table(transpose(estimadores_buenos)), 
#      header = ["beta","gamma","sigma"])

sample_beta = estimadores_buenos[1, 1:end]
sample_gamma = estimadores_buenos[2, 1:end]
sample_sigma = estimadores_buenos[3, 1:end]

hb = histogram(sample_beta, normalize = :pdf, xlabel = L"\hat{\beta}_{MVS}", label = false)
hg = histogram(sample_gamma, normalize = :pdf, xlabel = L"\hat{\gamma}_{MVS}", label = false)
hs = histogram(sample_sigma, normalize = :pdf, xlabel = L"\hat{\sigma}_{MVS}", label = false)
hists = plot(hb, hg, hs, layout = (3,1), dpi = 200)
savefig(hists, "D:/Edgar Trejo/Universidad/Proyecto/data/images/samples_MVS_precovid.png")

# intervalos de confianza
ibeta = (quantile(sample_beta, 0.05/2), quantile(sample_beta, 1-0.05/2))
igamma = (quantile(sample_gamma, 0.05/2), quantile(sample_gamma, 1-0.05/2))
isigma = (quantile(sample_sigma, 0.05/2), quantile(sample_sigma, 1-0.05/2))

βmvs, γmvs, σmvs = mode(sample_beta), mode(sample_gamma), mode(sample_sigma)
mean(estimadores_buenos, dims = 2)

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
    sim[i, 1:end] = solveSIS([βmvs, γmvs, σmvs, N], T, I₀; dt = dt)
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
     title = "Simulación de trayectorias con modas marginales \ndel estimador "*L"\theta_{MVS}"*" con datos pre-pandemia", 
     xlabel = "Tiempo (en años)", ylabel = "Proporción de infectados")
plot!(ts, low, color = :gray, label = "Cuantiles p = 0.05, 0.95")
plot!(ts, upper, color = :gray, label = false)
plot_sims = plot!(obs_ts, obs, seriestype = :scatter, label = "Observaciones", dpi = 200)
savefig(plot_sims, "D:/Edgar Trejo/Universidad/Proyecto/data/images/sims_theta_MVS_precovid.png")
# Validación del modelo :)
# Obtención de valores estimados
#βmle, γmle, σmle = θmle
# Definición de h(x) y su inversa
# func_h(x) = (1/(N*σ₀))*log(x/(N-x))
inv_h(x) = (N*exp(σmvs*N*x))/(1+exp(σmvs*N*x))
# Funciones de deriva y con los parámetros estimados
func_b(x) = βmvs*x*(N-x)-γmvs*x
func_s(x) = σmvs*x*(N-x)
func_ds(x) = σmvs*(N-2x)
# Definición de la función μ en la transformación de Lamperti
func_μ(x) = func_b(inv_h(x))/func_s(inv_h(x)) - (1/2)*func_ds(inv_h(x))
# Cálculo de los residuales
n = length(obs)+1
residuals = zeros(n-1)
obs_plus = vcat(I₀, obs)
for i in 2:n
    residuals[i-1] = obs_plus[i] - obs_plus[i-1] - func_μ(obs_plus[i-1])
end
# Análsis de los residuales ¿Cumplen con la normalidad?
# qq-plot: los puntos deben estar en la identidad 
plot_qq = plot(qqnorm(residuals), xlabel = "Datos", ylabel = "Normales", 
title = "Comparación de residuales", dpi = 200)
savefig(plot_qq, "D:/Edgar Trejo/Universidad/Proyecto/data/images/qqplot_info_incompleta_precovid.png")
# prueba estadística
prueba = ShapiroWilkTest(residuals) # p-value = 0.5277

interp_means = linear_interpolation(ts, m)
sim_means = interp_means.(obs_ts)

mae(obs, sim_means)
mse(obs, sim_means)
