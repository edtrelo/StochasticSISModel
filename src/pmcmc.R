library(pomp)
library(foreach)
library(coda)

N <- 1
I0 <- 0.0912*10^(-3)
obs <- c(0.1397, 0.1966, 0.2530, 0.3476, 0.4575, 0.6142, 
         0.7986, 0.8176, 0.6257, 0.8376, 0.3979)
obs <- obs*10^(-3)

vpstep <- function (X, beta, gamma, sigma, delta.t, ...){
  dW <- rnorm(n=1, mean=0, sd=sqrt(delta.t))
  c(X = X + (beta*X*(N-X)-gamma*X)*delta.t + sigma*X*(N-X)*dW)
}

rproc_snippet <- Csnippet("
  double dW = rnorm(0, sqrt(dt));
  double dX = (beta * X * (N - X) - gamma * X) * dt + sigma * X * (N - X) * dW;
  X += dX;
")

data <- data.frame(times = 1:length(obs), Y = obs)

sis.model <- pomp(
  data = data,
  times = "times",
  t0=0,
  rinit=Csnippet("X = I0;"),
  rprocess = euler(rproc_snippet, delta.t = 1/100),
  dmeasure = Csnippet("lik = dnorm(Y, X, 8e-4, give_log);"),
  paramnames=c("beta", "gamma", "sigma"),
  globals = "double N = 1; double I0=9.12e-05;",
  statenames = c("X"))


b <- c(17.56702665721366, 11.7575)
g <- c(17.333333332417734, 11.5555)
s <- c(0.17076322461407206, 0.334710)


sis.model |> pfilter(Np=200,
        params=c(beta = b[2], gamma = g[2], sigma = s[2], I0 = I0),
        statenames="X", filter.mean = TRUE) -> pf
plot(pf)
data.frame(pf)
filter_mean(pf)


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

pmcmc1 <- foreach(i = 1:2, .combine = c) %dopar%
        pmcmc(
          sis.model, 
          Nmcmc = 4000, 
          Np = 500,
          params = c(beta = b[i], gamma = g[i], sigma = s[i]),
          
          dprior = function(beta, gamma, sigma, I0, ..., log) {
            p <- dnorm(beta, mean = b[i], sd = 6, log = TRUE) +
                 dunif(gamma, min = 52/6, max = 52/3, log = TRUE) +
                 dbeta(sigma, shape1 = 2.142, shape2 = 10.388, log = TRUE)
            if (log) p else exp(p)
          },
          proposal = mvn_rw_adaptive(
            rw.sd = c(beta = 0.05^2, gamma = 0.01^2, sigma = 0.001^2),
            scale.start=100,shape.start=100
          )
        )
      
plot(pmcmc1)

library(coda)

# Supongamos que tienes dos cadenas:
chain1 <- as.mcmc(pmcmc1[[1]]@traces)
chain2 <- as.mcmc(pmcmc1[[2]]@traces)

# Combina para análisis conjunto:
mcmc.list <- mcmc.list(chain1, chain2)
mcmc.burned <- window(mcmc.list, start = 1000)
mcmc.burned

summary(mcmc.burned)
traceplot(mcmc.burned)
gelman.diag(mcmc.burned)

densplot(mcmc.burned)

acceptanceRate(pmcmc1[[1]])

library(bayesplot)
posterior <- as.matrix(mcmc.burned)
mcmc_pairs(posterior, pars = c("beta", "gamma", "sigma"))
