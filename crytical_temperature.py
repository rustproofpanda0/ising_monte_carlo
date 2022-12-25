import numpy as np

from ising_mc import IsingMC


### ----  initial parameters  ---- ###

Jv = 1.2
Jh = 1
size = 32

Hmag = np.arange(0.05, 1.01, 0.05)
temp = np.arange(0.1, 10, 0.1)

### ----------------------------- ###

try:

    M = []

    for H in Hmag:
        magnetization = []
        for T in temp:

            system = IsingMC(size=size, T=T, Jh=Jh, Jv=Jv, Hmag=H)
            system.run(32**4)

            magnetization.append(system.state.mean())

            print(f'Hmag = {H}    T = {T}')

        M.append(magnetization)
    
finally:

    filename = f"magnetization_Jv_{Jv}.npy"
    np.save(filename, np.array(M))



