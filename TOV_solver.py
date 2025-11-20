import os, sys, re
import numpy as np
import matplotlib.pyplot as plt
from kuibit.simdir import SimDir 
from kuibit.grid_data import UniformGrid 

##### ETK extraction Functions #####

def get_info(thorn,quantity, folder,t0,coordinate="x"):
        print("Looking for files in the folder: {}".format(folder))
        os.chdir(folder)
        filex = f"{thorn}-{quantity}.{coordinate}.asc"
        print("Opening file: {}.....".format(filex))
        datax = np.loadtxt(filex, comments='#')
        print("Reading file....")
        
        # Number of iterations
        it = np.unique(datax[:, 0])
        it_n = len(it)
        print("Number of iterations:", it_n)
        
        # Time values
        t = np.unique(datax[:, 8])
        # X values
        if coordinate == "x": 
           x_p = np.unique(datax[:, 9])
        if coordinate == "y": 
           x_p = np.unique(datax[:, 10])
        if coordinate == "z": 
           x_p = np.unique(datax[:, 11])
         
        # Refinement levels	
        rl = np.unique(datax[:, 2])
        print('N points in x_p:')
        print(len(x_p))
        rl_n = len(rl)
        print("Total number of refinement levels:", rl_n)

        if t0<t[-1] and t0!=0:
            t=t[t>t0]
            t_n = len(t)
            print("Number of different time values:", t_n)

        # Points
            x_p_n = len(x_p)
            print("Total number of points:", x_p_n)


            points_per_rl = []
            rl_max_point = []
            for i in range(rl_n):
                x_in_rl = np.unique(datax[datax[:, 2] == rl[i], 9])
                points_in_rl = len(x_in_rl)
                print("Number of points in refinement level", i, ":", points_in_rl)
                rl_max_point.append(np.max(x_in_rl))
                points_per_rl.append(points_in_rl)
       # rl_max_point.append(0.0)
        
        return t,x_p,rl,rl_n,datax

def get_1d_slice(tk1, xk1, datax, itd, coordinate):

    #print(f"Getting 1d-{coordinate} slice at t = {tk1[itd]}")

    t_index = datax[:,8] == tk1[itd] # get all values at fixed time t_i = tk1[itd]

       # get data  as t,coordinate,f(t,coordinate) 
    if coordinate == "x": 
       f_x_ti = np.vstack(  (datax[t_index,8],  datax[t_index,9]  , datax[t_index,12]  ))
    if coordinate == "y": 
       f_x_ti = np.vstack(  (datax[t_index,8],  datax[t_index,10]  , datax[t_index,12]  ))
    if coordinate == "z": 
       f_x_ti = np.vstack(  (datax[t_index,8],  datax[t_index,11]  , datax[t_index,12]  ))

       # split into t_i, x_j, f(x_j,t_i) 
    tj = (f_x_ti[0]).tolist()    # t_i should be all the same
    xj = (f_x_ti[1]).tolist()    # array of {x,y,z} values
    f_xi_tj = (f_x_ti[2]).tolist()

    # Convert lists back to numpy arrays for sorting
    xj = np.array(xj)
    f_xi_tj = np.array(f_xi_tj)
    # Sort the arrays based on xj
    sorted_indices = np.argsort(xj)
    # Reorder both xj and f_xi_tj based on sorted indices
    xj_sorted = xj[sorted_indices]
    f_xi_tj_sorted = f_xi_tj[sorted_indices]

    return xj_sorted, f_xi_tj_sorted

##### Solver Functions #####
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

def solve_tov_full(rho_c, K=100, gamma=2, dr=0.09, r_max=200):
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
rho_c_arr = [7.57e-04, 8.41e-04, 9.35e-04, 1.04e-03, 1.15e-03, 1.28e-03, 1.41e-03, 1.55e-03, 1.71e-03, 1.88e-03, 2.07e-03]

#print("Central Densities:", rho_c_arr)

#### data from the ETK nad solver for comparison

radius_etk = [1.0511e+01, 1.0343e+01, 1.0165e+01, 9.9769e+00, 9.7912e+00, 9.5855e+00, 9.3933e+00, 9.1998e+00, 8.9941e+00, 8.7917e+00, 8.5829e+00]
grav_mass_etk = [1.1070, 1.1696e+00, 1.2313e+00, 1.2915e+00, 1.3457e+00, 1.4002e+00, 1.4457e+00, 1.4864e+00, 1.5241e+00, 1.5558e+00, 1.5828e+00]
baryonic_mass_etk = [1.1677, 1.2385e+00, 1.3091e+00, 1.3785e+00, 1.4419e+00, 1.5062e+00, 1.5606e+00, 1.6098e+00, 1.6559e+00, 1.6950e+00, 1.7289e+00]

M_solver = []
R_solver = []
M_baryonic_solver = []

for ik in range(len(rho_c_arr)):
    rho_c = rho_c_arr[ik]
    M, R, R_iso, r_int, m_int, P_int, phi_int, r_ext, phi_ext = solve_tov_full(rho_c)

    M_solver.append(M)
    R_solver.append(R)

    r = np.array(r_int)
    m = np.array(m_int)
    P = np.array(P_int)
    rho_b = np.where(P > 0, (P / K)**(1/gamma), 0.0)

    mask = (1 - 2*m/r) > 0
    r = r[mask]
    m = m[mask]
    rho_b = rho_b[mask]

    integrand = 4 * np.pi * r**2 * rho_b / np.sqrt(1 - 2*m/r)
    # Total baryonic mass = integral of integrand over radius
    M_baryonic = np.trapz(integrand, r)
    M_baryonic_solver.append(M_baryonic)
    #print(f"rho_c: {rho_c:.4e}, Gravitational Mass: {M*2*1e30:.4e} (Kg), Radius: {R*1.47664:.4f} (Km), Isotropic Radius: {R_iso*1.47664:.4f} (Km)")

radius_etk = np.array(radius_etk)
R_solver = np.array(R_solver)
radius_etk *= 1.47664  # Convert to km
R_solver *= 1.47664  # Convert to km

plt.figure(figsize=(8,5))

plt.subplot(1,3,1)
plt.plot(R_solver, M_solver, label="Solver")
plt.scatter(radius_etk, grav_mass_etk, color='k', label="ETK data")
plt.xlabel("Radius R (km)")
plt.ylabel("Gravitational Mass")
plt.legend()

plt.subplot(1,3,2)
plt.plot(R_solver, M_baryonic_solver, label="Solver")
plt.scatter(radius_etk, baryonic_mass_etk, color='k', label="ETK data")
plt.xlabel("Radius R (km)")
plt.ylabel("Baryonic Mass")
plt.legend()    

plt.subplot(1,3,3)
plt.plot(R_solver, rho_c_arr, label="Solver")
plt.scatter(radius_etk, rho_c_arr, color='k', label="ETK data")
plt.xlabel("Radius R (km)")
plt.ylabel("Density")
plt.legend()

plt.show()

#plt.savefig("RvsM")


t_rns,x_p_rns,rl_rns,rl_n_rns,datax_rns = get_info("hydrobase","rho","/home/harsh/simulations/hydro_rns/output-0006/tov_ET",0.0,"x")
xj_sorted_rns, f_xi_tj_sorted_rns = get_1d_slice(t_rns, x_p_rns, datax_rns, itd = 0, coordinate="x")

M, R, R_iso, r_int, m_int, P_int, phi_int, r_ext, phi_ext = solve_tov_full(rho_c_arr[5])
r_bar_int, psi_int_iso = convert_to_isotropic(r_int, m_int)
rho_b_arr = (np.array(P_int)/K)**(1/gamma) # Baryon density

plt.plot(xj_sorted_rns, f_xi_tj_sorted_rns, label=r"Baryon density $\rho_b(r)$ from ETK data")
plt.plot(r_bar_int, rho_b_arr, '--', label=r"Baryon density $\rho_b(r)$ from the solver")
plt.xlabel(r"Radius $r$ (Km)")
plt.ylabel(r"$\rho_b(r)$")
plt.legend()
plt.show()

sys.exit()


# M = M*2*1e30
# R = R*1.47664  # Convert to km
# r_int = r_int * 1.47664  # Convert to km
# m_int = m_int * 2*1e30  # Convert to Kg
# R_iso = R_iso * 1.47664  # Convert to km

print(f"Gravitational Mass: {M*2*1e30:.4e} (Kg), Radius: {R*1.47664:.4f} (Km), Isotropic Radius: {R_iso*1.47664:.4f} (Km)")

rho_b_arr = (np.array(P_int)/K)**(1/gamma) # Baryon density
epsilon_arr = rho_b_arr + np.array(P_int)/(gamma - 1) # Energy density
enthalpy_arr = np.where(rho_b_arr > 0, (epsilon_arr + np.array(P_int))/np.array(rho_b_arr), 0.0) # Enthalpy
#M_0 = 4 * np.pi * np.array(r_int)**2 * rho_b_arr / np.sqrt(1 - (2*np.array(m_int) / np.array(r_int))) # Baryonic Mass

# r, m, P as numpy arrays from your solver
r = np.array(r_int)
m = np.array(m_int)
P = np.array(P_int)

# Baryon density (avoid negative pressures)
rho_b = np.where(P > 0, (P / K)**(1/gamma), 0.0)

# Avoid unphysical region where 1 - 2*m/r <= 0
mask = (1 - 2*m/r) > 0
r = r[mask]
m = m[mask]
rho_b = rho_b[mask]

# Integrand
integrand = 4 * np.pi * r**2 * rho_b / np.sqrt(1 - 2*m/r)

# Total baryonic mass = integral of integrand over radius
M_baryonic = np.trapz(integrand, r)

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

# plt.plot(r_int*1.47664, m_int*2*1e30, label="Mass Profile")
# plt.xlabel("Radius r(Km)")
# plt.ylabel("Enclosed Mass m(r) (Kg)")
# plt.legend()
# plt.savefig("mass_profile.png", dpi=300, bbox_inches='tight')
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
print(len(r_bar_int), len(psi_int_iso), len(alpha_int))

### Exterior

r_bar_ext = r_ext / (1 + M/(2*r_ext))**2
psi_ext_iso = np.sqrt(r_ext / r_bar_ext)    # Conformal factor
alpha_ext = np.exp(phi_ext) # Lapse function in isotropic coords (Exterior)

### Combine Interior and Exterior

# # Conformal factor
# plt.plot(r_bar_int, psi_int_iso, label="Interior")
# plt.plot(r_bar_ext, psi_ext_iso, '--', label="Exterior")
# plt.xlabel(r"Isotropic radius $\bar{r}$")
# plt.ylabel(r"Conformal factor $\psi(\bar{r})$")
# plt.legend()
# plt.savefig("/home/harsh/m_thesis/Programs/Harsh_thesis/output_plots/iso_psi.png", dpi=300, bbox_inches='tight')
# plt.show()

# # Lapse
# plt.plot(r_bar_int, alpha_int, label="Interior")
# plt.plot(r_bar_ext, alpha_ext, '--', label="Exterior")
# plt.xlabel(r"Isotropic radius $\bar{r}$")
# plt.ylabel(r"Lapse $\alpha(\bar{r})$")
# plt.legend()
# plt.savefig("/home/harsh/m_thesis/Programs/Harsh_thesis/output_plots/iso_alpha.png", dpi=300, bbox_inches='tight')
# plt.show()

# # Mapping stretch r → r̄
# plt.plot(r_int, r_bar_int, label=r"$\bar{r}(r)$")
# plt.xlabel(r"Areal radius $r$")
# plt.ylabel(r"Isotropic radius $\bar{r}$")
# plt.legend()
# plt.savefig("/home/harsh/m_thesis/Programs/Harsh_thesis/output_plots/iso_r_stretch.png", dpi=300, bbox_inches='tight')
# plt.show()


# M = M*2*1e30
# R = R*1.47664  # Convert to km
# r_int = r_int * 1.47664  # Convert to km
# m_int = m_int * 2*1e30  # Convert to Kg
# R_iso = R_iso * 1.47664  # Convert to km

#### matching values at the surface [check up]
print("psi_int(R) =", psi_int_iso[-1])
print("psi_ext(R) =", psi_ext_iso[0])
print("alpha_int(R) =", alpha_int[-1])
print("alpha_ext(R) =", alpha_ext[0])
print(r"\_bar(R) =", r_bar_int[-1]*1.47664, "Km")
print(r"r_{int} = ", r_int[-1]*1.47664, "Km")
print(r"m_{int} = ", m_int[-1])


################################################
 # ETK output data for comparison
################################################

# constants, in SI
G = 6.673e-11       # m^3/(kg s^2)
c = 299792458       # m/s
M_sol = 1.98892e30  # kg
# convertion factors
M_to_ms = 1./(1000*M_sol*G/(c*c*c))
M_to_density = c**5 / (G**3 * M_sol**2) # kg/m^3

ixd = 0  # index of the x point for time series
itd = 0  # index of the time point for 1D slice

t_h,x_p_h,rl_h,rl_n_h,datax_h = get_info("hydrobase","rho","/home/harsh/simulations/tov_ET_high/output-0000/tov_ET",0.0,"x")
xj_sorted_h, f_xi_tj_sorted_h = get_1d_slice(t_h, x_p_h, datax_h, itd, coordinate="x")

print(f"solver legth = {len(r_int)}")
print(f"data legth = {len(xj_sorted_h)}")

plt.figure(figsize=(8,5))

plt.plot(r_bar_int, rho_b_arr, label=r"Baryon density $\rho_b(r)$ from the solver")
plt.plot(xj_sorted_h, f_xi_tj_sorted_h, color='k', label=r"Baryon density $\rho_b(r)$ from ETK data")
plt.xlabel(r"Radius $r$")
plt.ylabel(r"$\rho_b(r)$")
plt.legend()
plt.savefig("/home/harsh/m_thesis/Programs/Harsh_thesis/output_plots/ETK_solver_rho_comparison.png", dpi=300, bbox_inches='tight')
plt.show()