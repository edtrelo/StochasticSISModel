# --- funciones para calcular la función de log-verosimilitud 
# y su gradiente (para encontrar el máximo).

# La función φ puede expresarse de la siguiente forma
# ϕ = f1 * exp(-(f2^2)/f3), por ello es útil ya tener estas expresiones.

function f1(σ, N, z, h)
    return 1 / (sqrt(2*π*h) * σ*(N-z)*z)
end

function f2(Ynext, z, h, β, N, γ)
    return Ynext - z*(1 + h*β*(N-z) - γ*h)
end

function f3(σ, h, N, z)
    return 2*h*(σ*z*(N-z))^2
end

function ∂φ∂β(β, γ, σ, z, h, N, Ynext)
    den = f3(σ, h, N, z)
    num = f2(Ynext, z, h, β, N, γ)
    res = (-2*h*f1(σ, N, z, h))/(den)
    res *= exp(-(num^2)/den)
    res *= z*(z-N)
    res
end

function ∂φ∂γ(β, γ, σ, z, h, N, Ynext)
    den = f3(σ, h, N, z)
    num = f2(Ynext, z, h, β, N, γ)
    f1(σ, N, z, h)*exp(-(num^2)/den)*(-2*num*(-h*z))*(1/den)
end

function ∂φ∂σ(β, γ, σ, z, h, N, Ynext)
    den = f3(σ, h, N, z)
    num = f2(Ynext, z, h, β, N, γ)
    ff =  f1(σ, N, z, h)
    term_uno = exp(-(num^2)/den) * σ * ff *(-1/(σ^2))
    term_dos = ff * exp(-(num^2)/den) * (-(num^2)/(2*h*(z*(N-z))^2))*(-1/(σ^3))
    term_uno+term_dos
end

function ∂φ∂z(β, γ, σ, z, h, N, Ynext)
    den = f3(σ, h, N, z)
    num = f2(Ynext, z, h, β, N, γ)
    term_uno = (1/(sqrt(2*π*h)*σ))*exp(-(num^2)/den) * ((2*z-N)/(z*(N-z))^2)
    term_dos = f1(σ, N, z, h)*exp(-(num^2)/den)
    term_dos *= -((2*h*(2*h*β*z - h*β*N - 1)*den - 4*((σ*h)^2) * (-(N-z)*(z^2) + z*((N-z)^2))))
    term_dos *= (1/(4*(h^2)*((σ*(N-z)*z)^2)^2))
    term_uno + term_dos
end

function ∂z∂β(M, solution, h, β, γ, σ, N, dz)
    dY = 0
    for i in 1:(M-1)
        x = solution[i]
        dY += ((N-x)*x + (β*(N-2x) - γ)*dY)*h + σ*dY*(N-2x)*dz[i+1]*sqrt(h)
    end
    return dY
end

function ∂z∂γ(M, solution, h, β, γ, σ, N, dz)
    dY = 0
    for i in 1:(M-1)
        x = solution[i]
        dY+= (-x + (β*(N-2x) - γ)*dY)*h + σ*dY*(N-2x)*dz[i+1]*sqrt(h)
    end
    return dY
end 

function ∂z∂σ(M, solution, h, β, γ, σ, N, dz)
    dY = 0
    for i in 1:(M-1)
        x = solution[i]
        dY+= ( (β*N - γ - 2*β*x)*dY)*h + ((N-x)*x +  σ*dY*(N-2x))*dz[i+1]*sqrt(h) 
    end
    return dY
end 

function ∂q∂β(M, S, h, β, γ, σ, N, DZ, Ycurrent, Ynext)
    # Solutions: SxM matriz
    res = 1
    for s in 1:S 
        dz = DZ[s, 1:end]
        solution = solveSIS([β, γ, σ, N], 1, Ycurrent, dz; dt = h)
        z = solution[end - 1]
        res += ∂φ∂β(β, γ, σ, z, h, N, Ynext) + ∂φ∂z(β, γ, σ, z, h, N, Ynext)*∂z∂β(M, solution, h, β, γ, σ, N, dz)
    end
    (1/S)*res
end

function ∂q∂γ(M, S, h, β, γ, σ, N, DZ, Ycurrent, Ynext)
    # Solutions: SxM matriz
    res = 1
    for i in 1:S 
        dz = DZ[i, 1:end]
        solution = solveSIS([β, γ, σ, N], 1, Ycurrent, dz; dt = h)
        z = solution[end - 1]
        res += ∂φ∂γ(β, γ, σ, z, h, N, Ynext) + ∂φ∂z(β, γ, σ, z, h, N, Ynext)*∂z∂γ(M, solution, h, β, γ, σ, N, dz)
    end
    (1/S)*res
end

function ∂q∂σ(M, S, h, β, γ, σ, N, DZ, Ycurrent, Ynext)
    # Solutions: SxM matriz
    res = 1
    for i in 1:S 
        dz = DZ[i, 1:end]
        solution = solveSIS([β, γ, σ, N], 1, Ycurrent, dz; dt = h)
        z = solution[end - 1]
        res += ∂φ∂σ(β, γ, σ, z, h, N, Ynext) + ∂φ∂z(β, γ, σ, z, h, N, Ynext)*∂z∂σ(M, solution, h, β, γ, σ, N, dz)
    end
    (1/S)*res
end

# Función que aproxima la densidad de transisión.
function q(θ, M, S, Ynext, tnext, Ycurrent, tcurrent, DZ, N)
    h = (tnext - tcurrent)/M
    β, γ, σ = θ
    suma = 0
    for s in 1:S 
        Z = solveSIS([β, γ, σ, N], (tnext - tcurrent), Ycurrent, DZ[s, 1:end]; dt = h)
        zs = Z[end-1]
        D = Normal(zs + (β*zs*(N-zs) - γ*zs)*h, σ*zs*(N-zs)*sqrt(h))
        suma += pdf(D, Ynext)
    end 
    return suma/S
end

function ∇F_β(β, γ, σ, DZ, M, S, h, obs, N, obs_ts)
    value = 0
    for i in 1:(length(obs)-1)
        Ycurrent = obs[i]
        Ynext = obs[i+1]
        tcurrent = obs_ts[i]
        tnext = obs_ts[i+1]
        value+= ∂q∂β(M, S, h, β, γ, σ, N, DZ, Ycurrent, Ynext)/q([β, γ, σ], M, S, Ynext, tnext, Ycurrent, tcurrent, DZ, N)
    end
    return value
end

function ∇F_γ(β, γ, σ, DZ, M, S, h, obs, N, obs_ts)
    value = 0
    for i in 1:(length(obs)-1)
        Ycurrent = obs[i]
        Ynext = obs[i+1]
        tcurrent = obs_ts[i]
        tnext = obs_ts[i+1]
        value+= ∂q∂γ(M, S, h, β, γ, σ, N, DZ, Ycurrent, Ynext)/q([β, γ, σ], M, S, Ynext, tnext, Ycurrent, tcurrent, DZ, N)
    end
    return value
end

function ∇F_σ(β, γ, σ, DZ, M, S, h, obs, N, obs_ts)
    value = 0
    for i in 1:(length(obs)-1)
        Ycurrent = obs[i]
        Ynext = obs[i+1]
        tcurrent = obs_ts[i]
        tnext = obs_ts[i+1]
        value+= ∂q∂σ(M, S, h, β, γ, σ, N, DZ, Ycurrent, Ynext)/q([β, γ, σ], M, S, Ynext, tnext, Ycurrent, tcurrent, DZ, N)
    end
    return value
end

# Gradiente de la log-verosimilitud
function ∇log_likelihood(x_0, DZ, M, S, h, obs, N, obs_ts)
    β, γ, σ = x_0 
    [∇F_β(β, γ, σ, DZ, M, S, h, obs, N, obs_ts), 
    ∇F_γ(β, γ, σ, DZ, M, S, h, obs, N, obs_ts), 
    ∇F_σ(β, γ, σ, DZ, M, S, h, obs, N, obs_ts)]
end

function ∇log_likelihood(x_0, DZ, M, S, h, obs, N, obs_ts)
    β, γ, σ = x_0 
    [∇F_β(β, γ, σ, DZ, M, S, h, obs, N, obs_ts), 
    ∇F_γ(β, γ, σ, DZ, M, S, h, obs, N, obs_ts), 
    ∇F_σ(β, γ, σ, DZ, M, S, h, obs, N, obs_ts)]
end

function ∇F!(F, x_0)
    β, γ, σ = x_0 
    F[1] = -∇F_β(β, γ, σ, DZ, M, S, h, obs, N, obs_ts)
    F[2] = -∇F_γ(β, γ, σ, DZ, M, S, h, obs, N, obs_ts) 
    F[3] = -∇F_σ(β, γ, σ, DZ, M, S, h, obs, N, obs_ts)
end

function ∇F(x_0)
    β, γ, σ = x_0 
    [∇F_β(β, γ, σ, DZ, M, S, h, obs, N, obs_ts),
    ∇F_γ(β, γ, σ, DZ, M, S, h, obs, N, obs_ts),
    ∇F_σ(β, γ, σ, DZ, M, S, h, obs, N, obs_ts)]
end

function F(x_0)
    suma = 0
    if sum(x_0 .<= 0) > 0
        return -Inf
    end
    for i in 2:(length(obs)-1)
        suma += log(q(x_0, M, S, obs[i], obs_ts[i], obs[i-1], obs_ts[i-1], DZ, N))
    end
    return suma
end

