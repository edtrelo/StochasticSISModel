include("utilities.jl")
using Plots
using Random
using Statistics
using LinearSolve
using LaTeXStrings
using KernelDensity
using Distributions
using Interpolations
using Optim
using DataFrames
using CSV
using StatsPlots
Random.seed!(1)

# datos discretos

N = 1
obs = [0.0912, 0.1397, 0.1966, 0.2530, 0.3476, 0.4575, 0.6142, 0.7986, 0.8176, 0.6257, 0.8376, 0.3979]*10^(-3)
#obs = obs.
obs_ts = collect(0:(length(obs)-1))

tray = [obs obs_ts]
s = SIS_sigma_by_qv(tray[1:end, 1], obs_ts)
function likelihhod_info_completa(θ)
    beta, gamma = θ
    ts = tray[1:end, 2]
    Xi = linear_interpolation(ts, tray[1:end, 1])

    int_3 = quadgk(t->1/(N-Xi(t)), 0, ts[end])[1]
    int_4 = quadgk(t->1/(N-Xi(t))^2, 0, ts[end])[1]

    int_1 = itointegral(y-> 1/(y*(N-y)), tray[1:end, 1])
    int_2 = itointegral(y-> 1/(y*(N-y)^2), tray[1:end, 1])

    T = ts[end]
    l = beta/(s^2) *int_1 - gamma/(s^2)*int_2 
    l += - ((beta^2)*T)/(2*(s^2)) + (beta*gamma)/(s^2)*int_3 - (gamma^2)/(2*(s^2))*int_4
    return l
end
γ₀ = 52/4.5
# obtención de β₀
β₀ = 0
for i in 1:(length(obs)-1)
    β₀ += (obs[i+1] - obs[i] + γ₀*obs[i])/(obs[i]*(N-obs[i]))
end
β₀ = β₀/(length(obs)-1)
# Valor inicial para el algoritmo de optimización
θ₀ = [β₀, γ₀]
# Cubo para la constrained optimization
lower = [0, 52/6]
upper = [Inf, 52/3]
# Optimización 
# F es la función de log-verosimilitud
sol = optimize(x -> -likelihhod_info_completa(x), lower, upper, θ₀, Fminbox(BFGS()); inplace = false)
θmle = sol.minimizer
# simulaciones vs observaciones
T = obs_ts[end]+0.5
dt = 0.001
ts = collect(0:dt:T)
# Aquí se guardarán las simulaciones
n_simulaciones = 500
sim = zeros(n_simulaciones, length(ts))
# Simulación de trayectorias con los parámetros de MV
for i in 1:n_simulaciones
    sim[i, 1:end] = solveSIS(vcat(θmle, [s,N]), T, obs[1]; dt = dt)
end
# Ahora 
intervals = zeros(3, length(ts))
# Obtención de la media y los quantiles 0.05 y 0.95
for i in 1:length(ts)
    datai = sim[1:end, i]
    intervals[2, i] = mean(datai)
    intervals[1, i] = quantile(datai, 0.05/2)
    intervals[3, i] = quantile(datai, 1-0.05/2)
end
# Gráficas 🦖
m = intervals[2, 1:end] # mean para Xₜ
low = intervals[1, 1:end] # cuantil 0.05 para Xₜ
upper = intervals[3, 1:end] # cuantil 0.95 

data_simulaciones_4_2_2 = DataFrame(time = ts, mean = m, low_limit = low, upper_limit = upper)
CSV.write("D:/Edgar Trejo/Universidad/Proyecto/StochasticSISModel/data/samples/simulaciones_4_2_1.csv", 
          data_simulaciones_4_2_2, writeindex=false)

theme(:default)
plot(ts, m, ribbon = (m.-low, upper.-m), label = "Media de la trayectoria", lw = 2,
     title = "\n"*L"\textbf{a}."*" Simulación contra datos observados", 
     titlefont = 10, 
     xlabel = "t", ylabel = "Proporción de infectados")
plot!(ts, low, color = :gray, label = "Cuantiles "*L"\alpha = 0.05")
plot!(ts, upper, color = :gray, label = false)
plot_sims = plot!(obs_ts, obs, xticks = xticks = (obs_ts, string.(2011:2022)),
                  seriestype = :scatter, label = "Observaciones", tickfont = 7, guidefont = 9,
                  bottom_margin = 5Plots.mm, left_margin = 5Plots.mm, dpi=200)

# Validación del modelo :)
# Obtención de valores estimados
βmle, γmle = θmle
# Definición de h(x) y su inversa
# func_h(x) = (1/(N*σ₀))*log(x/(N-x))
inv_h(x) = (N*exp(s*N*x))/(1+exp(s*N*x))
# Funciones de deriva y con los parámetros estimados
func_b(x) = βmle*x*(N-x)-γmle*x
func_s(x) = s*x*(N-x)
func_ds(x) = s*(N-2x)
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
title = "\n"*L"\textbf{b}."*" Qqplot de residuales", titlefont = 10, lw = 2, tickfont = 8, guidefont = 9,
bottom_margin = 5Plots.mm, right_margin = 5Plots.mm, topmargin = 5Plots.mm, dpi = 200)

result_plot = plot(plot_sims, plot_qq, layout = @layout([a{0.66w} b{0.34w}]), size = (800, 300), dpi = 200,
plot_title = "Estimación con la metodología de información completa", plot_titlefont = 12,
)

savefig(result_plot, "D:/Edgar Trejo/Universidad/Proyecto/StochasticSISModel/data/images/plot_4_2_1.png")
# prueba estadística
prueba = ShapiroWilkTest(residuals)
# rss
interp_means = linear_interpolation(ts, m)
sim_means = interp_means.(obs_ts[2:end])

mae(obs[2:end], sim_means)
mse(obs[2:end], sim_means)



