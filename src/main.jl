using DifferentialEquations
using Plots
using Statistics
using QuadGK
# Parámetros del modelo
N = 763
β = 0.002342
γ = 0.476
σ = 1e-3
p = (β, γ, N, σ)

function does_disease_dies(p)
    β, γ, N, σ = p
    R₀ = (β*N)/γ - ((σ*N)^2)/(2*γ)
    if (R₀ <= 1) & (σ^2 ≤ β/N)
        return true
    elseif σ^2 >= max(β/N, (β^2)/(2*γ))
        return true
    else
        return false
    end
end

does_disease_dies(p)


 
σ^2 ≤ (β/N)
# Definición del modelo
function drift(u, p, t)
    β, γ, N, σ = p
    u*(β*(N-u) - γ)
end

function diffusion(u, p, t)
    β, γ, N, σ = p
    σ*u*(N-u)
end

i₀ = 3
dt = 1/1000
tspan = (0.0, 20.0)
prob = SDEProblem(drift, diffusion, i₀, tspan, p)
sol = solve(prob, EM(), dt = dt)
plot(sol.u)
n = length(sol.u)

U = sol.u
 
using Interpolations
X = linear_interpolation(sol.t, sol.u)


function loglike(p)
    i1 = quadgk(s -> drift(X(s), p, 0)/diffusion(X(s), p, 0)^2, 0, 20)[1]
    i2 = quadgk(s -> drift(X(s), p, 0)^2/diffusion(X(s), p, 0)^2, 0, 20)[1]
    i1 - 1/2 * i2
end

loglike(p)

using Optim

solm = maximize(loglike, [1.0, 1.0, 763, 1.0])

R₀ = (β*N)/γ - ((σ*N)^2)/(2*γ)

m = (2*β*γ*(R₀ - 1))/(2*β*(β - (σ^2 * N)) + (σ^2)*(β*N - γ))
mean(sol.u[n-100:n])
histogram(sol.u[n-1000:n])


integrand(x) = drift(x, p, 0)/diffusion(x, p, 0)^2
I(x) = quadgk(integrand, i₀, x, rtol = 1e-3)[1]

function s(x)
    exp(-2*I(x))
end

plot(s.(1:100))

function M(x)
    1/ (diffusion(x, p, 0)^2 * I(x))
end

plot(M.(1:N))

import utilities.jl