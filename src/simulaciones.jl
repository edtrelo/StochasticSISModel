include("utilities.jl")
using Plots
using Random
Random.seed!(1)

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

savefig(plot_sol, "D:/Edgar Trejo/Universidad/Proyecto/data/images/ejemplo1.png")