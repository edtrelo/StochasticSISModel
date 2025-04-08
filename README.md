## Modelo SIS estocástico

En este repositorio se pretende almacenar el código usado para realizar tareas de *simulación* e *inferencia* en un modelo epidemiológico SIS 
estocástico, como parte del curso **Proyecto I** del semestre 2025-02.

El modelo SIS con el que se trabaja tiene la forma

$$dI_t = (\beta I_t(N-I_t)-\gamma I_t)dt + \sigma I_t(N-I_t)dW_t$$

donde $I_t$ es el número de infectados al tiemopo $t$, $\beta$ es la tasa de infección, $\gamma$ es la tasa de recuperación$, 
$N$ es el tamaño de la población, $\sigma$ es la intensidad del movimiento Browniano $W_t$.
