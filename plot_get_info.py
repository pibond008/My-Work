import os, sys, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from kuibit.simdir import SimDir 
from kuibit.grid_data import UniformGrid  

################################################
 # Define constants and conversion factors
################################################
# constants, in SI
G = 6.673e-11       # m^3/(kg s^2)
c = 299792458       # m/s
M_sol = 1.98892e30  # kg
# convertion factors
M_to_ms = 1./(1000*M_sol*G/(c*c*c))
M_to_density = c**5 / (G**3 * M_sol**2) # kg/m^3

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

def fx_timeseries(t,x_p,datax,ixd=0, coordinate="x"):     #index value of x as input
    #create output lists
    print(os.getcwd())
    t_n = len(t)
    time_values = []
    f_xt_values = []
    #print(f"Calculating timeseries for {coordinate} = {x_p[ixd]}")
    print(f"Starting at  t = {t[0]}")
 # create filter for time steps
    for j in range(t_n): 
        t_index = datax[:,8] == t[j]
# get data  as t,coordinate,f(t,coordinate) 
        if coordinate == "x": 
          f_x_ti = np.vstack(  (datax[t_index,8],  datax[t_index,9]  , datax[t_index,12]  ))
        if coordinate == "y": 
          f_x_ti = np.vstack(  (datax[t_index,8],  datax[t_index,10]  , datax[t_index,12]  ))
        if coordinate == "z": 
          f_x_ti = np.vstack(  (datax[t_index,8],  datax[t_index,11]  , datax[t_index,12]  ))
#now x=f_x_ti[0][:] and f(x)=f_x_ti[1][:]
 #create filter for space points
        if ixd==0:
           x_index = f_x_ti[1][:] == 0.0
        else:
           x_index = f_x_ti[1][:] == x_p[ixd]

 # save t, x and f(x,t) in a list (use lists to improve efficiency when extending)
        tj = (f_x_ti[0][x_index]).tolist()
        f_xi_tj = (f_x_ti[2][x_index]).tolist()

 #append values
        time_values.extend(tj)
        f_xt_values.extend(f_xi_tj)
        if(j==np.round(1/8*t_n) or j==np.round(1/4*t_n)) or j==np.round(3/8*t_n) or  j==np.round(1/2*t_n) or j==np.round(5/8*t_n) or j==np.round(3/4*t_n) or j==np.round(7/8*t_n):
                print("Progress: {} %".format(j/t_n *100))
    print("Done...!")
    return time_values,f_xt_values

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

ixd = 4  # index of the x point for time series
itd = 0  # index of the time point for 1D slice
#res = "h" # resolution: low, mid, high

# if res == "l":
#     x_index = ixd*2
# elif res == "m":
#     x_index = ixd
# elif res == "h":
#     x_index = ixd*0.5

# print(f"x_index for resolution {res} is {x_index:.2f}")

t_l,x_p_l,rl_l,rl_n_l,datax_l = get_info("hydrobase","rho","/home/harsh/simulations/tov_ET_low/output-0000/tov_ET",0.0,"x")
t_m,x_p_m,rl_m,rl_n_m,datax_m = get_info("hydrobase","rho","/home/harsh/simulations/tov_ET_mid/output-0000/tov_ET",0.0,"x")
t_h,x_p_h,rl_h,rl_n_h,datax_h = get_info("hydrobase","rho","/home/harsh/simulations/tov_ET_high/output-0000/tov_ET",0.0,"x")
# out_dir = "/home/harsh/Masters Thesis/Programs/blender_data"
# os.makedirs(out_dir, exist_ok=True)

# for itd in range(len(t)):
#     xj_sorted, f_xi_tj_sorted = get_1d_slice(t, x_p, datax, itd, coordinate="x")
#     np.savetxt(
#         f"{out_dir}/frame_{itd:03d}.csv",
#         np.column_stack((xj_sorted, f_xi_tj_sorted)),
#         delimiter=",",
#         header="x,rho",
#         comments=""
#     )

# time_values_l,f_xt_values_l = fx_timeseries(t_l,x_p_l,datax_l,ixd,coordinate="x")
# xj_sorted_l, f_xi_tj_sorted_l = get_1d_slice(t_l, x_p_l, datax_l, itd, coordinate="x")

# time_values_m,f_xt_values_m = fx_timeseries(t_m,x_p_m,datax_m,ixd,coordinate="x")
# xj_sorted_m, f_xi_tj_sorted_m = get_1d_slice(t_m, x_p_m, datax_m, itd, coordinate="x")

time_values_h,f_xt_values_h = fx_timeseries(t_h,x_p_h,datax_h,ixd,coordinate="x")
xj_sorted_h, f_xi_tj_sorted_h = get_1d_slice(t_h, x_p_h, datax_h, itd = 0, coordinate="x")
print(f"time length = {len(xj_sorted_h)}")

# Plotting time series for different resolutions

plt.figure(figsize=(8,5))  # optional, makes figure larger

# plt.plot(time_values_l, f_xt_values_l, color='r', label=f"Low (x={ixd*2})")
# plt.plot(time_values_m, f_xt_values_m, color='b', label=f"Mid (x={ixd})")
plt.plot(xj_sorted_h[0:51], f_xi_tj_sorted_h[0:51], color='k', label=f"High (x={ixd*0.5})")

plt.xlabel("Time")
plt.ylabel("Rho")
plt.title("rho vs time at x index")
plt.legend()
plt.grid(True)  # optional, makes reading easier
plt.savefig("/home/harsh/m_thesis/Programs/output_plots/rho_x.png", dpi=300)
plt.show()

#plt.close()  # closes the figure

sys.exit()
# plt.plot(xj_sorted, f_xi_tj_sorted, label=f"1D slice at ixd={ixd}, itd={itd}")
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.legend()
# plt.savefig(f"/home/harsh/Masters Thesis/Programs/plots/1d_slice_{ixd}_{itd}.png", dpi=300)
# plt.close()  # closes the figure

############### 1D SLICE PLOTTING SCRIPT #####################
xj_sorted, rho_0 = get_1d_slice(t_h, x_p_h, datax_h, itd=0, coordinate="x")
rho_0 = rho_0
rho_threshold = 1e-9

# time = []
surface = []
for itd in range(len(t_h)):
    xj_sorted, f_xi_tj_sorted = get_1d_slice(t_h, x_p_h, datax_h, itd, coordinate="x")
    for i in range(len(f_xi_tj_sorted)):
        if f_xi_tj_sorted[i] < rho_threshold:
            surf = xj_sorted[i]
            surface.append(surf)
            break
    # plt.plot(xj_sorted, f_xi_tj_sorted/rho_0, label=f"t = {t[itd]*M_to_ms:.2f} ms")
    # plt.axvline(surf, color='g', linestyle='-', label=f'surface {surf} at t = {itd}')
    # plt.xlabel("x")
    # plt.ylabel("Density (rho)")
    # plt.ylim(0,4000)
    # plt.title(f"Density profile at t = {t[itd]}")
    # plt.legend()
    # plt.savefig(f"/home/harsh/Masters Thesis/Programs/plots/Movie/slice_{itd:03d}.png", dpi=300)
    # plt.close()


    ################## 2D PLOTTING SCRIPT #####################

last_or_first = 0 # 0 or -1
x_value = surface [last_or_first]
sims_dir = "/home/harsh/simulations/"  # Adjust this path as needed
# Directory and plotting setup
direc = "/home/harsh/simulations/tov_ET_high/output-0000/tov_ET/"
thorns = ["hydrobase"]
quantities = ["rho"]
ax_lims = [[(-10,10),(-0.01, 0.005)],
            [(-10,10),None],
            [(-30,30),None]
            ]  # Adjust if needed for phi


mass_phi_code_units = 0.1

# Create 2x2 subplots
#fig, axs = plt.subplots(2,2,figsize=(22, 22))  # 2 rows, 2 columns
fig,ax = plt.subplots(figsize=(15,15))
#####################################################
# -------- SECOND COLUMN: 2D Contour Plots -------- #
#####################################################

testsim = SimDir(direc)
rho_lim = 10
rho_grid = UniformGrid([100, 100], x0=[0,0], x1=[rho_lim, rho_lim])
grids = [rho_grid]

thorns = ["hydrobase"]
quantities = ["rho"]


for idx, (thorn, quantity,grid) in enumerate(zip(thorns, quantities,grids)):
    #ax = axs[idx, 1]  # Right column
    ax_current = ax
    field = getattr(testsim.gf.xz.fields, quantity)
    it_nbr = field.available_iterations
    time_steps = field.available_times
    #print(f"[{quantity}] Available iterations: {it_nbr}")
    its   = it_nbr[last_or_first]
    dtime = time_steps[last_or_first]
    rho0_center = field.read_on_grid(its, grid)
    newgrid = np.array(rho0_center.coordinates_meshgrid()) * 1.477

    # Use log scale for rho, linear for phi
    if quantity == "rho":
        rho_cgs_xyz = np.log10(rho0_center.data_xyz) #* cgs_to_ETK_density/1e15
        
        cf = ax.contourf(*newgrid, rho_cgs_xyz, levels=100,cmap="inferno")
        
        # Draw surface
        arc1 = Arc((0, 0), width=2*x_value*1.477, height=2*x_value*1.477,
          theta1=0, theta2=90, color='white', linestyle='--', linewidth=2)
        ax.add_patch(arc1)
        text = ax.text(rho_lim*0.8*1.477 , rho_lim*0.9*1.477, f"t = {dtime/M_to_ms:.2f} ms", fontsize=12, color='white')
 
        cbar = plt.colorbar(cf, ax=ax)   
        cbar.set_label('log10[rho]', fontsize=14)

    #ax.set_title(f"{ylabels_2d[quantity]}", fontsize=18)
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Z [km]")



# Only show circle in +x,+y
    # arc1 = Arc((0, 0), width=2*x_value*1.477, height=2*x_value*1.477,
    #         theta1=0, theta2=4*90, color='white', linestyle='--', linewidth=2)
    # axs[idx, 1].add_patch(arc1)


# Final adjustments
saveplotdir = "/home/harsh/m_thesis/Programs/output_plots/"
plt.tight_layout()

fig.savefig(saveplotdir+"TOV_ID_1d2d.pdf", dpi=300)

plt.show()
plt.close(fig)
