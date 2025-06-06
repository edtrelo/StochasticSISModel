include("utilities.jl")
using Plots
Random.seed!(1)

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