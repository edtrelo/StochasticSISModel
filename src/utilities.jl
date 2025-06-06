using QuadGK
using Distributions
using Interpolations
include("simulated_likelihood_utilities.jl")

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

# --- Resultados teóricos del modelo SIS -------------------------------------------- #
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

# La media analítica obtendia por Grey, et al. 2011
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

# ----------------------------------------------------------------------------------- #

# --- Simulación -------------------------------------------------------------------- #
"""
Implementación de los métodos Euler-Maruyama y Milstein.

EM: Euler-Maruyama
Mil: Milstein
"""
function _updatestate!(Y, ts, θ, dt, dW, extra_term)
    for i in 2:length(ts)
        Y[i] = Y[i-1]
        Y[i] += _driftSIS(Y[i-1], θ, ts[i-1])*dt
        Y[i] += _diffusionSIS(Y[i-1], θ, ts[i-1])*dW[i-1]*sqrt(dt)
        Y[i] += extra_term*0.5*_diffusionSIS(Y[i-1], θ, ts[i-1])*_diffusionSISderiv(Y[i-1], θ, ts[i-1])*((dW[i-1]*sqrt(dt))^2 - dt)
    end
end

function setvalues(θ, T, x₀, t₀, dt, method)
    extra_term = 0.0
    if method == "Mil"
        extra_term = 1.0
    end
    ts = collect(t₀:dt:T)
    Y = zeros(length(ts))
    Y[1] = x₀
    return extra_term, ts, Y
end

function solveSIS(θ, T, x₀; t₀ = 0, dt = 0.001, method = "EM")
    extra_term, ts, Y = setvalues(θ, T, x₀, t₀, dt, method)
    dW = rand(Normal(0, 1), length(ts)-1)
    _updatestate!(Y, ts, θ, dt, dW, extra_term)
    return Y
end

function solveSIS(θ, T, x₀, dW; t₀ = 0, dt = 0.001, method = "EM")
    extra_term, ts, Y = setvalues(θ, T, x₀, t₀, dt, method)
    _updatestate!(Y, ts, θ, dt, dW, extra_term)
    return Y
end

# --- Utilidades para inferencia ------------------------------------------------------- #
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
    sqrt( sum( diff(X).^2 ) / quadgk(t -> Xi(t)^2 * (N-Xi(t))^2, ts[1], ts[end])[1])
end

"""Calcula la mean squared error"""
function mse(obs, predicted)
    return sum((obs.-predicted).^2)/length(obs)
end

"""Mean absolute error"""
function mae(obs, predicted)
    return sum(abs.(obs.-predicted))/length(obs)
end