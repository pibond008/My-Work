# My Work – Numerical Methods & Orbital Simulations

## Overview
This repository contains **numerical simulations of a three-body orbital system** using various integration schemes. The simulations study **orbital dynamics, energy conservation, angular momentum, and eccentricity evolution** for multiple bodies under Newtonian gravity.

Implemented in **Python** with **NumPy** and **Matplotlib**.

## Physics Context
We simulate a system of three bodies:  
1. Central massive body  
2. Two smaller orbiting bodies  

Key quantities computed:  
- Positions and velocities  
- Energy (kinetic + potential)  
- Angular momentum  
- Orbital elements: eccentricity `e` and semi-major axis `a`  

The code also explores **varying initial separations (`delta`)** and close encounters between bodies.

## Repository Structure
```
my-work/
├── Chaos-N_body_simulations/   # Optional exercises
├── chaos.py          # Main simulation code                       
└── readme.md                          # This file
```

## Usage
1. **Install dependencies**:
```bash
pip install numpy matplotlib
```

2. **Run the main simulation**:
```bash
python orbital_simulations.py
```

3. **Output**:  
- Trajectories of bodies for different integrators  
- Energy and angular momentum plots  
- Eccentricity and semi-major axis evolution  
- Adaptive time-step plots  
- Number of close encounters

## Integrators Implemented
- Euler  
- Euler-Cromer  
- Leap Frog  
- Verlet  
- Runge-Kutta 4 (RK4)  

Supports **adaptive time stepping** and allows varying:  
- Number of bodies (`n`)  
- Initial distances (`a`, `delta`)  
- Orbital eccentricity (`e`)  
- Simulation duration (`tmax`)  
- Time step (`del_t`)  

## Example Figures
- Trajectories  
- Energy & angular momentum conservation  
- Eccentricity & semi-major axis evolution  
- Adaptive time-step changes  
- Close encounters highlighted  

## Author
**Harsh Solanki** – Physics & Computational Modeling Enthusiast  
[GitHub: harsh-c0ds](https://github.com/harsh-c0ds)
