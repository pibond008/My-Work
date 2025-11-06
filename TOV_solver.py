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
    
    dm = (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6
    dP = (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6
    dphi = (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6

    return np.array([m + dr*dm, P + dr*dP, phi + dr*dphi])

def solve_tov_full(rho_c, K=100, gamma=2, dr=1e-3, r_max=200):
    P = K*rho_c**gamma # central pressure
    r = 1e-6 # start at a small radius to avoid singularity
    _, eps_c = eos(P, K, gamma) # central energy density
    m = (4*np.pi/3)*r**3*eps_c # initial mass
    phi = 0.0 # initial metric potential
    y = np.array([m, P, phi]) # initial state vector

    r_int = [r]
    m_int = [m]
    P_int = [P]
    phi_int = [phi]

    while y[1] > 0 and r < r_max:
        y = rk4_step(r, y, dr, K, gamma)
        r += dr
        m, P, phi = y
        r_int.append(r)
        m_int.append(m)
        P_int.append(P)
        phi_int.append(phi)

    R = r_int[-1]
    M = m_int[-1]
    phi_R = 0.5*np.log(1 - 2*M/R)
    phi_shift = phi_R - phi_int[-1]
    phi_int = np.array(phi_int) + phi_shift

    r_ext = np.linspace(R, r_max, 600)
    phi_ext = 0.5*np.log(1 - 2*M/r_ext)
    R_iso = R/(1+M/(2*R))**2

    return M, R, R_iso, np.array(r_int), np.array(m_int), np.array(P_int), phi_int, r_ext, phi_ext

rho_c = 1.28e-3
M, R, R_iso, r_int, m_int, P_int, phi_int, r_ext, phi_ext = solve_tov_full(rho_c)

print(f"Gravitational Mass: {M:.4f}, Radius: {R:.4f}, Isotropic Radius: {R_iso:.4f}")


plt.plot(r_int, phi_int, label="Interior")
plt.plot(r_ext, phi_ext, '--', label="Exterior")
plt.axvline(R, linestyle=':', color='k')
plt.legend()
plt.show()