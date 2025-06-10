library(pomp)
library(foreach)
library(dplyr)

N <- 1
I0 <- 0.0912*10^(-3)
obs <- c(0.1397, 0.1966, 0.2530, 0.3476, 0.4575, 0.6142, 
         0.7986, 0.8176, 0.6257, 0.8376, 0.3979)
obs <- obs*10^(-3)


rproc_snippet <- Csnippet("
  double dW = rnorm(0, sqrt(dt));
  double dX = (beta * X * (N - X) - gamma * X) * dt + sigma*X*(N-X)*dW;
  X += dX;
  
  beta += rnorm(0, sigma_beta);
  sigma += rnorm(0, sigma_sigma);
")

data <- data.frame(times = 1:length(obs), Y = obs)

sis.model <- pomp(
  data = data,
  times = "times",
  t0=0,
  rinit=Csnippet("X = I0; 
                 beta = beta0; 
                 sigma=sigma0;"),
  rprocess = euler(rproc_snippet, delta.t = 1/100),
  dmeasure = Csnippet("lik = dnorm(Y, X, 1e-5, give_log);"),
  paramnames=c("gamma", "sigma_beta", "sigma_sigma", "I0", "beta0", "sigma0"),
  globals = "const double N = 1;",
  statenames = c("X", "beta", "sigma"))


b <- c(17.56702665721366, 11.7575)
g <- c(17.333333332417734, 11.5555)
s <- c(0.17076322461407206, 0.334710)


sis.model |> pfilter(Np=200,
                     params=c(beta0 = b[2], 
                              gamma = g[2], 
                              sigma0 = s[2], 
                              I0 = I0,
                              sigma_beta = 0.1,
                              sigma_sigma = 0.001),
                     statenames="X", filter.mean = TRUE) -> pf
plot(pf)
data.frame(pf)
filter_mean(pf)

pf_sims <- replicate(
  n = 500,
  expr = pfilter(
    sis.model,
    Np = 200,
    params = c(
      beta0 = b[2],
      gamma = g[2],
      sigma0 = s[2],
      I0 = I0,
      sigma_beta = 0.1,
      sigma_sigma = 0.001
    ),
    filter.mean = TRUE
  )@filter.mean,
  simplify = "array"
)

times <- 1:dim(pf_sims)[2]
state_names <- dimnames(pf_sims)[[1]]

pf_df <- do.call(rbind, lapply(1:500, function(i) {
  as.data.frame(t(pf_sims[,,i])) |>
    mutate(time = times, sim = i)
})) |>
  relocate(sim, time)

pf_summary <- pf_df |>
  group_by(time) |>
  summarise(
    X_mean = mean(X),
    X_lo = quantile(X, 0.025),
    X_hi = quantile(X, 0.975),
    
    beta_mean = mean(beta),
    beta_lo = quantile(beta, 0.025),
    beta_hi = quantile(beta, 0.975),
    
    sigma_mean = mean(sigma),
    sigma_lo = quantile(sigma, 0.025),
    sigma_hi = quantile(sigma, 0.975)
  )

library(ggplot2)

ggplot(pf_summary, aes(x = time)) +
  geom_ribbon(aes(ymin = X_lo, ymax = X_hi), fill = "lightblue", alpha = 0.3) +
  geom_line(aes(y = X_mean), color = "blue", size = 1) +
  labs(title = "X (infectados) - Media y banda 95% de filtros de partículas", 
       y = "X", x = "Tiempo") +
  theme_minimal()

ggplot(pf_summary, aes(x = time)) +
  geom_ribbon(aes(ymin = beta_lo, ymax = beta_hi), fill = "orange", alpha = 0.3) +
  geom_line(aes(y = beta_mean), color = "darkorange", size = 1) +
  labs(title = "β (transmisión) - Filtros de partículas", 
       y = "β", x = "Tiempo") +
  theme_minimal()

ggplot(pf_summary, aes(x = time)) +
  geom_ribbon(aes(ymin = sigma_lo, ymax = sigma_hi), fill = "gray80", alpha = 0.3) +
  geom_line(aes(y = sigma_mean), color = "black", size = 1) +
  labs(title = "σ (difusión) - Filtros de partículas", 
       y = "σ", x = "Tiempo") +
  theme_minimal()

obs_data <- data.frame(time = 1:length(obs), Y = obs)

# Gráfica de X con banda de incertidumbre y puntos observados
library(ggplot2)

ggplot(pf_summary, aes(x = time)) +
  geom_ribbon(aes(ymin = X_lo, ymax = X_hi), fill = "lightblue", alpha = 0.4) +
  geom_line(aes(y = X_mean), color = "blue", size = 1.2) +
  geom_point(data = obs_data, aes(y = Y), color = "red", size = 2) +
  labs(
    title = "Estado latente X (infectados) vs observaciones",
    x = "Tiempo",
    y = "Proporción infectados"
  ) +
  theme_minimal() +
  theme(text = element_text(size = 14))



pmcmc1 <- foreach(i = 1:2, .combine = c) %dopar%
  pmcmc(sis.model, Nmcmc = 4000, Np = 100, 
        params = c(beta= b[i], gamma = g[i], sigma = s[i], I0 = I0),
        dprior = function(beta, gamma, sigma,I0,...,log){
          p <- dnorm(beta, mean = b, sd = 1)
          p <- p + dnorm(gamma, mean = g, sd = 0.1) 
          p <- p + dbeta(sigma, 2.142, 10.388)
          if (log) p else exp(p)
        },
        proposal = mvn_diag_rw(c(beta=0.1,
                                 gamma=0.01,
                                 sigma = 0.01,
                                 I0 = 0.001))
        
        hist(as.data.frame(traces(pmcmc1))$beta)
        plot(pmcmc1)
        