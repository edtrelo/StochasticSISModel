using DifferentialEquations
using Plots
using Statistics
using QuadGK
using Distributions
using Random
using Interpolations
using Optim
Random.seed!(1)

# Utilidades ------------------------------------------------------------------- #

# Definición del modelo SIS estocástico 

# Funciones de deriva y difusión
function _driftSIS(u, p, t)
    β, γ, σ, N = p
    u*(β*(N-u) - γ)
end
function _diffusionSIS(u, p, t)
    β, γ, σ, N = p
    σ*u*(N-u)
end
# Derivada de la función de difusión
function _diffusionSISderiv(u, p, t)
    β, γ, σ, N = p
    -σ*(2*u - N)
end
# Test de extinción o prevalencia
function does_disease_dies(p)
    β, γ, σ, N = p
    R₀ = (β*N)/γ - ((σ*N)^2)/(2*γ)
    if (R₀ <= 1) & (σ^2 ≤ β/N)
        return true
    elseif σ^2 >= max(β/N, (β^2)/(2*γ))
        return true
    elseif R₀>1
        return false
    end
end

"""
Implementación de los métodos Euler-Maruyama y Milstein.

EM: Euler-Maruyama
Mil: Milstein
"""
function solveSIS(θ, T, x₀; t₀ = 0, dt = 0.001, method = "EM")
    extra_term = 0.0
    if method == "Mil"
        extra_term = 1.0
    end
    ts = collect(t₀:dt:T)
    Y = zeros(length(ts))
    Y[1] = x₀
    for i in 2:length(ts)
        dW = rand(Normal(0, sqrt(dt)))
        Y[i] = Y[i-1]
        Y[i] += _driftSIS(Y[i-1], θ, ts[i-1])*dt
        Y[i] += _diffusionSIS(Y[i-1], θ, ts[i-1])*dW
        Y[i] += extra_term*0.5*_diffusionSIS(Y[i-1], θ, ts[i-1])*_diffusionSISderiv(Y[i-1], θ, ts[i-1])*(dW^2 - dt)
    end
    return Y
end

# Estimadores respecto a la densidad estacionaria --------------------- #
function analytic_mean(p)
    # Teorema 
    β, γ, σ, N = p
    R₀ = (β*N)/γ - ((σ*N)^2)/(2*γ)
    (2*β*γ*(R₀ - 1))/(2*β*(β - (σ^2 * N)) + (σ^2)*(β*N - γ))
end


function _m(x, p, x₀)
    # Distribución estacionaria analítica
    β, γ, σ, N = p 
    numerador = ( (x₀*(N-x))/(x*(N-x₀)) )^((2*(γ - N*β))/(σ*N)^2)
    numerador *= exp((2*γ*(x₀-x))/((σ^2)*(x-N)*(N*x₀-N^2)))
    denominador = σ^2*x^2*(N-x)^2
    numerador/denominador
end

function define_stationary_dist(θ, i₀)
    cte = quadgk(y -> _m(y, θ, i₀), 0, N)[1]
    x -> _m(x, θ, i₀)/cte
end

function numeric_mean(θ, i₀)
    f = define_stationary_dist(θ, i₀)
    quadgk(y -> y*f(y), 0, N)[1]
end

# ---------------------------------------------------------------------- #

# Estimaciones dada una realización de un modelo
"""
Estima el valor de la integral

        ∫f(Xₛ)dXₛ 

por medio de la definición

        ∫f(Xₛ)dXₛ ≈ ∑ᵢ₌₀ f(X(tᵢ₋₁))*(X(tᵢ) - X(tᵢ₋₁))    

"""
function itointegral(f, X)
    sum(f.(X[1:end-1]) .* diff(X))
end

"""Estima σ del modelo SIS por medio de la variación cuadrática."""
function SIS_sigma_by_qv(X, ts)
    Xi = linear_interpolation(ts, X)
    sqrt( sum( diff(X).^2 ) / quadgk(t -> Xi(t)^2 * (N-Xi(t))^2, 0, 20)[1])
end
# --------------------------------------------------------------------------- #
dt = 0.001
# Parámetros del modelo
N = 763
β = 0.0002342
γ = 1/7
σ = 0.003
p = (β, γ, σ, N)
does_disease_dies(p)

# Simulaciones
i₀ = 3
# Definición del problema como EDE
solEM = solveSIS(p, 20.0, i₀)
p_sol = plot(solEM, label = "Método de Euler-Maruyama")
solMil = solveSIS(p, 20.0, i₀;  method = "Mil")
p_sol = plot!(solMil, label = "Método de Milstein");
# Graficamos un histograma de la distribución estacionaria
# en caso de que sí exista.
last_n_elements = Int(round(length(solMil)*0.25))
if !does_disease_dies(p)
    # Histograma
    hist_data = vcat(solEM[end - last_n_elements: end],
                     solMil[end - last_n_elements: end])
    hist_plot = histogram(
                        hist_data, 
                        color = :lightgrey, 
                        normalize=true,
                        lw = 0,
                        label = "Distribución muestral"
                )
    # Media muestral
    hist_plot = vline!([mean(hist_data)], 
                        label = "Media muestral",
                        color = :black, 
                        lw = 1.5,
                        linestyle = :dash)
    # Media analítica
    hist_plot = vline!([analytic_mean(p)], 
                        label = "Media analítica",
                        color = :green,
                        lw = 1.5,
                        linestyle = :dash) 
    # Media numérica 
    #hist_plot = vline!([numeric_mean(p, i₀)], 
                        # label = "Media numérica",
                        # color = :orange,
                        # lw = 1.5,
                        # linestyle = :dash) 
    # Distribución estacionaria analítica 
    SD = define_stationary_dist(p, i₀)
    hist_plot = plot!(
                    SD.(0:N), 
                    label = "Distribución estacionaria analítica",
                    lw = 2,
                    color = :purple)
    plot(p_sol, hist_plot, layout = (2, 1))
else 
    plot(p_sol)
end


# Ejemplo 1: Enfermedad con extinción
p = (0.0002342, 1/7, 0.003, 763)
does_disease_dies(p)

I₀ = rand(1:N-1, 10)
ts = collect(0:dt:30.0)
sol = solveSIS(p, 30.0, I₀[1])
plot_sol = plot(ts, sol, label = I₀[1]);
for i in 2:10
    sol = solveSIS(p, 30.0, I₀[i])
    plot_sol = plot!(ts, sol, label = I₀[i])
end
plot_sol = plot(plot_sol, legendtitle = "Valor inicial", 
     xlabel = "t", ylabel = "Infectados")
savefig(plot_sol, "D:/Edgar Trejo/Universidad/POMP/data/ejemplo1.png")

# Ejemplo 2:
N = 763
β = 0.0005
γ = 1/7
σ = 0.0003
p = (β, γ, σ, N)
does_disease_dies(p)
#R₀ = (β*N)/γ - ((σ*N)^2)/(2*γ)

I₀ = 30
ts = collect(0:dt:50.0)
sol = solveSIS(p, 50.0, I₀; method = "Mil")
plot_tray = plot(ts, sol, label = false)
last_n_elements = Int(round(length(sol)*0.20))
hist_data = sol[end-last_n_elements: end]

for i in 2:10
    sol = solveSIS(p, 50.0, I₀; method = "Mil")
    plot_tray = plot!(ts, sol, label = false)
    hist_data = vcat(hist_data, sol[end-last_n_elements: end])
end
plot_sol = plot(plot_tray, ylabel = "Infectados", xlabel="t")

dist_est_plot = histogram(hist_data, normalize = true, label = "Distribución muestral",
                          color = :lightgrey, lw = 0, xlabel = "Infectados",
                          legend = :topleft)
# Distribución estacionaria analítica 
SD = define_stationary_dist(p, I₀)
dist_est_plot = plot!(
                SD.(0:N), 
                label = "Distribución estacionaria analítica",
                lw = 2,
                color = :purple)
# Media muestral
dist_est_plot = vline!([mean(hist_data)], 
                        label = "Media muestral",
                        color = :black, 
                        lw = 1.5,
                        linestyle = :dash)
# Media analítica
dist_est_plot = vline!([analytic_mean(p)], 
                        label = "Media analítica",
                        color = :green,
                        lw = 1.5,
                        linestyle = :dash) 
# Media numérica 
#dist_est_plot = vline!([numeric_mean(p, i₀)], 
                        # label = "Media numérica",
                        # color = :orange,
                        # lw = 1.5,
                        # linestyle = :dash)

ej2_plt = plot(plot_tray, dist_est_plot, layout = (2,1))

savefig(ej2_plt, "D:/Edgar Trejo/Universidad/POMP/data/ejemplo2.png")


# Estimación de parámetros ------------------------------------------ #
# Información completa
N = 763
β = 0.0005
γ = 1/7
σ = 0.0003
p = (β, γ, σ, N)
dt = 0.001
does_disease_dies(p)
#R₀ = (β*N)/γ - ((σ*N)^2)/(2*γ)

I₀ = 30
X = solveSIS(p, 30.0, I₀; method = "Mil")
length(X)
ts = collect(0:dt:30.0)
tray = plot(ts, X, xlabel = "t", ylabel = "Infectados")
#savefig(tray, "D:/Edgar Trejo/Universidad/POMP/data/tray.png")
# Estimación de σ por medio de variación cuadrática
ss = SIS_sigma_by_qv(X, ts)
Xi = linear_interpolation(ts, X)
function loglike(q)
    beta = q[1]
    p = (beta, γ, ss, N)
    i1 = itointegral(x -> _driftSIS(x, p, 0)/_diffusionSIS(x, p, 0)^2, X)
    i2 = quadgk(s -> _driftSIS(Xi(s), p, 0)^2/_diffusionSIS(Xi(s), p, 0)^2, 0, 30)[1]
    i1 - 1/2 * i2
end

# It-It-1 = bI-1(N-I-1)-γ i
((Xi(1)-Xi(0))+(1/7)*Xi(0))/(Xi(0)*(N-Xi(0)))

sol_maximizer = optimize(x -> -loglike(x), [0.0007193])
pm = Optim.minimizer(sol_maximizer)
pm

