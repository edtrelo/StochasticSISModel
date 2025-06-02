include("utilities.jl")
using Plots
using Random
using Statistics
using LinearSolve
using LaTeXStrings
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

plot(plots..., layout=(3,1), size = (500, 400), top_margin = 3.5*Plots.mm)

estimators = zeros(3, 100)

ts = collect(0:dt:T)
for i in 1:100
    X = solveSIS(p_original, T, I₀; dt = dt)
    tray = [X ts]
    estimators[1:end, i] .= EstimadorMVSIS(tray)
end

histogram(X[1:end, 1])

X = transpose(estimators[1:2, 1:end])
Z = [pdf(mvnorm,[i,j]) for i in 0:100, j in 0:100]
plot(0:100,0:100,Z,st=:surface)

mvnorm = fit(MvNormal, X')
Z = [pdf(mvnorm,[i,j]) for i in minimum(X[1:end, 1]):0.000001:maximum(X[1:end, 1]), 
j in minimum(X[1:end, 2]):0.000001:maximum(X[1:end, 2])]
plot(minimum(X[1:end, 2]):0.000001:maximum(X[1:end, 2]),minimum(X[1:end, 1]):0.000001:maximum(X[1:end, 1]),Z,st=:surface)


f(x,y) = pdf(mvnorm, [x,y])
contourf(minimum(X[1:end, 1]):0.000001:maximum(X[1:end, 1]), minimum(X[1:end, 2]):0.000001:maximum(X[1:end, 2]), f, color=:viridis)


collect(minimum(X[1:end, 2]):0.0001:maximum(X[1:end, 2]))