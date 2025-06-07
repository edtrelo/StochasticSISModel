## Modelo SIS estocástico

En este repositorio sirve para presentar el código usado para realizar *simulación* e *inferencia* en un modelo epidemiológico SIS 
estocástico, como parte del curso **Proyecto I** 2025-02, de la Facultad de Ciencias de la UNAM.

El modelo SIS con el que se trabaja tiene la forma de la siguiente ecuación diferencial estástica

$$dI_t = (\beta I_t(N-I_t)-\gamma I_t)dt + \sigma I_t(N-I_t)dW_t$$

donde $I_t$ es el número de infectados al tiempo $t$, $\beta$ es la tasa de infección, $\gamma$ es la tasa de recuperación, 
$N$ es el tamaño de la población y $\sigma$ es la intensidad del movimiento Browniano $W_t$ que introduce ruido al modelo.
