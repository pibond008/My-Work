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
    dP_dr = - (epsilon + P) * (m + 4 * np.pi * r**3 * P) / (r * (r - 2 * m))
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

def convert_to_isotropic(r_arr, m_arr):
    r_bar = np.zeros_like(r_arr)
    r_bar[0] = r_arr[0]

    for i in range(len(r_arr)-1):
        r = r_arr[i]
        m = m_arr[i]
        dr = r_arr[i+1] - r_arr[i]
        fac = np.sqrt(1 - 2*m/r)
        dbar = (r_bar[i] / r) * fac * dr
        r_bar[i+1] = r_bar[i] + dbar

    psi = np.sqrt(r_arr / r_bar) # Conformal factor
    return r_bar, psi

K = 100
gamma = 2
rho_c = 1.28e-3
M, R, R_iso, r_int, m_int, P_int, phi_int, r_ext, phi_ext = solve_tov_full(rho_c)

print(f"Gravitational Mass: {M:.4f}, Radius: {R:.4f}, Isotropic Radius: {R_iso:.4f}")

rho_b_arr = (np.array(P_int)/K)**(1/gamma) # Baryon density
epsilon_arr = rho_b_arr + np.array(P_int)/(gamma - 1) # Energy density
enthalpy_arr = np.where(rho_b_arr > 0, (epsilon_arr + np.array(P_int))/np.array(rho_b_arr), 0.0) # Enthalpy
M_0 = 4 * np.pi * np.array(r_int)**2 * rho_b_arr / np.sqrt(1 - (2*np.array(m_int) / np.array(r_int))) # Baryonic Mass

# plt.plot(r_int, rho_b_arr, label=r"Baryon density $\rho_b(r)$")
# plt.xlabel(r"Radius $r$")
# plt.ylabel(r"$\rho_b(r)$")
# plt.legend()
# plt.show()

# plt.plot(r_int[0:-1], enthalpy_arr[0:-1], label=r"Enthalpy $h(r)$")
# plt.xlabel(r"Radius $r$")
# plt.ylabel(r"Enthalpy ($h$)")
# plt.legend()
# plt.show()

# plt.plot(r_int, M_0, label=r"Baryonic Mass $M_0(r)$")
# plt.xlabel(r"Radius $r$")
# plt.ylabel(r"$M_0(r)$")
# plt.legend()
# plt.show()

# plt.plot(r_int, m_int, label="Mass Profile")
# plt.xlabel("Radius r")
# plt.ylabel("Enclosed Mass m(r)")
# plt.legend()
# plt.show()

# plt.plot(r_int, P_int, label="Pressure Profile")
# plt.xlabel("Radius r")
# plt.ylabel("Pressure P(r)")
# plt.legend()
# plt.show()

# plt.plot(r_int, phi_int, label="Interior")
# plt.plot(r_ext, phi_ext, '--', label="Exterior")
# plt.axvline(R, linestyle=':', color='k')
# plt.xlabel("Radius r")
# plt.ylabel("Metric Potential φ(r)")
# plt.legend()
# plt.show()

########## Isotropic Coordinate Conversion ##########

### Interior

r_bar_int, psi_int_iso = convert_to_isotropic(r_int, m_int)
alpha_int = np.exp(phi_int) # Lapse function in isotropic coords (Interior)


### Exterior

r_bar_ext = r_ext / (1 + M/(2*r_ext))**2
psi_ext_iso = np.sqrt(r_ext / r_bar_ext)    # Conformal factor
alpha_ext = np.exp(phi_ext) # Lapse function in isotropic coords (Exterior)

### Combine Interior and Exterior

# Conformal factor
plt.plot(r_bar_int, psi_int_iso, label="Interior")
plt.plot(r_bar_ext, psi_ext_iso, '--', label="Exterior")
plt.xlabel(r"Isotropic radius $\bar{r}$")
plt.ylabel(r"Conformal factor $\psi(\bar{r})$")
plt.legend()
plt.savefig("iso_psi.png", dpi=300, bbox_inches='tight')
plt.show()

# Lapse
plt.plot(r_bar_int, alpha_int, label="Interior")
plt.plot(r_bar_ext, alpha_ext, '--', label="Exterior")
plt.xlabel(r"Isotropic radius $\bar{r}$")
plt.ylabel(r"Lapse $\alpha(\bar{r})$")
plt.legend()
plt.savefig("iso_alpha.png", dpi=300, bbox_inches='tight')
plt.show()

# Mapping stretch r → r̄
plt.plot(r_int, r_bar_int, label=r"$\bar{r}(r)$")
plt.xlabel(r"Areal radius $r$")
plt.ylabel(r"Isotropic radius $\bar{r}$")
plt.legend()
plt.savefig("iso_r_stretch.png", dpi=300, bbox_inches='tight')
plt.show()


print("psi_int(R) =", psi_int_iso[-1])
print("psi_ext(R) =", psi_ext_iso[0])
print("alpha_int(R) =", alpha_int[-1])
print("alpha_ext(R) =", alpha_ext[0])
