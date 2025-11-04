import numpy as np
import matplotlib.pyplot as plt

def eos(P,K,gamma):
    if P <= 0:
        return 0.0, 0.0

    rho_b = (P/K)**(1/gamma) # baryon density
    epsilon = rho_b + P/(gamma - 1) # energy density
    return rho_b, epsilon

def tov_eqs(r, m, P, K, gamma):
    if P <= 0:
        return np.array([0.0, 0.0, 0.0])
    
    rho_b, epsilon = eos(P, K, gamma)

    dm_dr = 4 * np.pi * r**2 * epsilon
    dP_dr = - (rho_b + epsilon) * (m + 4 * np.pi * r**3 * P) / (r * (r - 2 * m))
    dphi_dr = (m + 4 * np.pi * r**3 * P) / (r * (r - 2 * m))

    return np.array([dm_dr, dP_dr, dphi_dr])

def rk4_step(r, y, dr, K, gamma):
    m, P, phi = y  # unpack y
    
    k1 = tov_eqs(r, m, P, K, gamma)
    k2 = tov_eqs(r + dr/2, m + dr*k1[0]/2, P + dr*k1[1]/2, K, gamma)
    k3 = tov_eqs(r + dr/2, m + dr*k2[0]/2, P + dr*k2[1]/2, K, gamma)
    k4 = tov_eqs(r + dr,   m + dr*k3[0],   P + dr*k3[1],   K, gamma)
    
    # RK4 weighted sum
    dm = (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6
    dP = (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6
    dphi = (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6
    
    # update all three components
    return np.array([m + dr*dm, P + dr*dP, phi + dr*dphi])

def solve_tov(rho_c, K=100, gamma=2, dr=1e-3):
    P = K * rho_c**gamma # central pressure
    r = 1e-6 # starting radius
    eps_c, _ = eos(P, K, gamma) # central energy density
    m = (4.0/3.0) * np.pi * r**3 * eps_c   # initial mass
    phi = 0.0 # initial metric potential

    y = np.array([m, P, phi])

    Radius = [r]
    Mass = [m]
    Pressure =[P]
    Phi = [phi]
    while y[1] > 0:
        y = rk4_step(r, y, dr, K, gamma)
        r += dr

        m, P, phi = y
        Radius.append(r)
        Mass.append(m)
        Pressure.append(P)
        Phi.append(phi)

    R = r
    M = m    
    R_iso = R / (1 + M/(2*R))**2
    phi_R = 0.5*np.log(1 - 2*M/R)
    phi_Shift = phi_R - Phi
    return M, R, R_iso, np.array(Radius), np.array(Mass), np.array(Pressure), np.array(Phi), np.array(phi_Shift)

rho_c = 1.28e-3
M, R, R_iso, Radius, Mass, Pressure, Phi, phi_shift = solve_tov(rho_c)
print(f"Gravitational Mass: {M:.4f}, Radius: {R:.4f}, Isotropic Radius: {R_iso:.4f}")

plt.figure(figsize=(8,6))
plt.plot(Radius, Mass)
plt.xlabel("Radius")
plt.ylabel("Mass")
plt.title(r"${M(R)}$ from TOV Solver")
plt.grid()

plt.figure(figsize=(8,6))
plt.plot(Radius, Pressure)
plt.xlabel("Radius")
plt.ylabel("Pressure")
plt.title(r"${P(R)}$ from TOV Solver")
plt.grid()

plt.figure(figsize=(8,6))
plt.plot(Radius, phi_shift)
plt.xlabel("Radius")
plt.ylabel(r"$\Phi(R) - \Phi(0)$")
plt.title(r"${\Phi(R) - \Phi(0)}$ from TOV Solver")
plt.grid()

plt.show()