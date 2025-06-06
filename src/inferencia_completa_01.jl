include("utilities.jl")
using Plots
using Random
using Statistics
using LinearSolve
using LaTeXStrings
using KernelDensity
using Distributions
using Interpolations
Random.seed!(1)

T = 500
M = 10000
dt = T/M
N = 763
β = 0.00035
γ = 1/32 # 0.03125
σ = 0.0001
p_original = (β, γ, σ, N)
I₀ = 10
does_disease_dies(p_original)

function EstimadorMVSIS(X)
    ts = X[1:end, 2]
    Xi = linear_interpolation(ts, X[1:end, 1])

    b = quadgk(t->1/(N-Xi(t)), 0, ts[end])[1]
    d = quadgk(t->1/(N-Xi(t))^2, 0, ts[end])[1]

    x = itointegral(y-> 1/(y*(N-y)), X[1:end, 1])
    y = itointegral(y-> 1/(y*(N-y)^2), X[1:end, 1])

    A = [ts[end] -b; b -d]
    v = [x, y]
    prob = LinearProblem(A, v)
    b, g = solve(prob)
    b, g, SIS_sigma_by_qv(X[1:end, 1], ts)
end

X = solveSIS(p_original, T, I₀; dt = dt)
ts = collect(0:dt:T)
tray = [X ts]

times = collect(20:20:T)
estimators = zeros(3, length(times))
for (i, t) in enumerate(times)
    subtray =tray[(tray[1:end, 2]).<t, 1:end]
    estimators[1:end, i] .= EstimadorMVSIS(subtray)
end
# Obtención de los estimadores de MV y el de σ


plots = Any[]
xlabels = ["", "", "T"]
ylabels = [L"\hat{\beta}_{MV}", L"\hat{\gamma}_{MV}", L"\hat{\sigma}"]
for i in 1:3
    y = estimators[i, 1:end]
    exponente = floor(log10(maximum(y)))
    p =  plot(times, y ./ (10^(exponente)), legend=false, lw=2, 
    xlabel = xlabels[i], ylabel = ylabels[i], marker = :circle, yformatter = identity)  
    i_exp = Int(exponente)
    p = annotate!(17, 
        maximum(y) / (10^(exponente)) + ((maximum(y))-minimum(y))/(10^(exponente)) * 0.15, 
        (L"10^{%$i_exp}", 7, :top, :center)) 
    p = hline!([p_original[i] ./ (10^(exponente)) ])
    push!(plots, p)
end

plot_est = plot(plots..., layout=(3,1), top_margin = 3.5*Plots.mm, dpi = 200)
savefig(plot_est, "D:/Edgar Trejo/Universidad/Proyecto/data/images/est_info_completa.png")

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
T = obs_ts[end]
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
    intervals[1, i] = quantile(datai, 0.05)
    intervals[3, i] = quantile(datai, 0.95)
end
# Gráficas 🦖
m = intervals[2, 1:end] # mean para Xₜ
low = intervals[1, 1:end] # cuantil 0.05 para Xₜ
upper = intervals[3, 1:end] # cuantil 0.95 
plot(ts, m, ribbon = (m.-low, upper.-m), label = "Media de la trayectoria", lw = 2,
     title = "Simulación con parámetros estimados por\ntratamiento de información completa", 
     xlabel = "Tiempo (en años)", ylabel = "Proporción de infectados")
plot!(ts, low, color = :gray, label = "Cuantiles p = 0.05, 0.95")
plot!(ts, upper, color = :gray, label = false)
plot_sims = plot!(obs_ts, obs, seriestype = :scatter, label = "Observaciones", dpi = 200)
savefig(plot_sims, "D:/Edgar Trejo/Universidad/Proyecto/data/images/simulacion_info_completa.png")

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
title = "Comparación de residuales", dpi = 200)
savefig(plot_qq, "D:/Edgar Trejo/Universidad/Proyecto/data/images/qqplot_info_completa.png")
# prueba estadística
prueba = ShapiroWilkTest(residuals)
# rss
interp_means = linear_interpolation(ts, m)
sim_means = interp_means.(obs_ts[2:end])

mae(obs[2:end], sim_means)
mse(obs[2:end], sim_means)