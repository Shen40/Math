import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
from scipy.special import j1

#total force vs distance between charges 
#log scale 
# --- Parameters ---
# Decreased length and time to visualize the interaction clearly and quickly
L = 200
T = 20
nx = 4001      
dx = 2 * L / (nx - 1) 
dt = 0.5 * dx  
nt = int(T / dt)
C2 = (dt / dx)**2

mu = 0.01
a1 = 2
a2 = 2
m = 1.0               
sigma_src = 0.2   

x = np.linspace(-L, L, nx)
t_arr = np.linspace(0, T, nt)

# Setting up dynamic Arrays 
# Particle 1
z1_arr = np.zeros(nt)  
zd1_arr = np.zeros(nt) 
p1_arr = np.zeros(nt)  
forces1 = np.zeros(nt) 

# Particle 2
z2_arr = np.zeros(nt)  
zd2_arr = np.zeros(nt) 
p2_arr = np.zeros(nt)  
forces2 = np.zeros(nt) 

# Set starting location and conditions
z1_arr[0] = -2.0
zd1_arr[0] = 0.0
p1_arr[0] = 0.0

z2_arr[0] = 2.0
zd2_arr[0] = 0.0
p2_arr[0] = 0.0

# Helpers
def dirac_delta_approx(x_val, x0_pos, sig):
    return (1.0 / (sig * np.sqrt(np.pi))) * np.exp(-((x_val - x0_pos) / sig)**2)

def source(x_val, z_val, zd_val, a):
    gamma_inv_sq = np.clip(1.0 - zd_val**2, 0.0, None)
    return a * np.sqrt(gamma_inv_sq) * dirac_delta_approx(x_val, z_val, sigma_src)

# --- Force Calculation ---
def calculate_force(n, t_current, z_cur, zd_cur, z_hist, zd_hist, u_field, a):
    # 1. calculate f_0 
    # No radiation here so no V terms
    gamma_inv_sq = np.clip(1.0 - zd_cur**2, 1e-15, None) 
    f_0 = -(0.5 * a**2) * (zd_cur / gamma_inv_sq)
    
    # 2. History/Tail integral I1^2
    if n == 0:
        I1_2 = 0.0
    else:
        t_past = t_arr[:n+1]
        x1_past = z_hist[:n+1]
        zd_past = zd_hist[:n+1]
        
        sigma = mu * np.sqrt(np.maximum(0.0, (t_current - t_past)**2 - (x1_past - z_cur)**2))
        bessel_term = np.where(sigma < 1e-10, 0.5, j1(sigma) / sigma)
        
        gamma_inv_past = np.sqrt(np.maximum(0.0, 1.0 - zd_past**2))
        integrand = (z_cur - x1_past) * gamma_inv_past * bessel_term
        
        I1_2 = (a * (mu**2) / 2.0) * np.trapezoid(integrand, t_past)

    # 3. Total Field Gradient (Self-gradient is ~0 due to symmetric smearing)
    du_dx = np.gradient(u_field, dx)
    du_dx_z = np.interp(z_cur, x, du_dx)
    
    # 4. Total Force Assembly
    gamma_inv = np.sqrt(np.maximum(0.0, 1.0 - zd_cur**2))
    f_total = f_0 + (I1_2 + du_dx_z) * (a * gamma_inv)
    
    return f_total

# --- Field Initialization ---
u_prev, u_curr, u_next = np.zeros(nx), np.zeros(nx), np.zeros(nx)

# Initial Static Field data 
U0_1 = (a1 / (2 * mu)) * np.exp(-mu * np.abs(x - z1_arr[0])) 
U0_2 = (a2 / (2 * mu)) * np.exp(-mu * np.abs(x - z2_arr[0])) 

u_curr[:] = U0_1[:] + U0_2[:]

# Prepare histories for plotting/animation
frame_step = max(1, nt // 200) 
history_u = [u_curr.copy()]
times_anim = [0.0]

# --- THE MAIN LOOP ---
print("Simulating... Started from defined trajectories, updating dynamically.")
for n in range(nt - 1): 
    
    # 1. Start with current Z and combine sources
    S1 = source(x, z1_arr[n], zd1_arr[n], a1)
    S2 = source(x, z2_arr[n], zd2_arr[n], a2)
    S_total = S1 + S2
    
    # 2. Calculate field (u_next) based on combined sources
    if n == 0:
        u_next[1:-1] = (u_curr[1:-1] + 
                        0.5 * C2 * (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2]) -
                        0.5 * (dt**2) * (mu**2) * u_curr[1:-1] +
                        0.5 * (dt**2) * S_total[1:-1])
    else:
        # Standard Finite Difference steps
        u_next[1:-1] = (2 * u_curr[1:-1] - u_prev[1:-1] +
                        C2 * (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2]) -
                        (dt**2) * (mu**2) * u_curr[1:-1] +
                        (dt**2) * S_total[1:-1])
        
    # Boundary Conditions 
    u_next[0] = 0; u_next[-1] = 0

    # 3. Calculate Force for both particles independently
    forces1[n] = calculate_force(n, t_arr[n+1], z1_arr[n], zd1_arr[n], z1_arr, zd1_arr, u_next, a1)
    forces2[n] = calculate_force(n, t_arr[n+1], z2_arr[n], zd2_arr[n], z2_arr, zd2_arr, u_next, a2)

    # 4. Dynamically update Z for the next step (n+1)
    # Particle 1
    p1_arr[n+1] = p1_arr[n] + dt * forces1[n]
    p_over_m1 = p1_arr[n+1] / m
    zd1_arr[n+1] = p_over_m1 / np.sqrt(np.maximum(0.0, 1.0 + p_over_m1**2))
    z1_arr[n+1] = z1_arr[n] + dt * zd1_arr[n+1]
    
    # Particle 2
    p2_arr[n+1] = p2_arr[n] + dt * forces2[n]
    p_over_m2 = p2_arr[n+1] / m
    zd2_arr[n+1] = p_over_m2 / np.sqrt(np.maximum(0.0, 1.0 + p_over_m2**2))
    z2_arr[n+1] = z2_arr[n] + dt * zd2_arr[n+1]

    # 5. Repeat: Rotate fields
    u_prev[:], u_curr[:] = u_curr[:], u_next[:]

    # Store animation frame
    if n % frame_step == 0:
        history_u.append(u_curr.copy())
        times_anim.append(t_arr[n+1])

# Calculate very last force point for consistency in graphs
forces1[-1] = calculate_force(nt-1, t_arr[-1], z1_arr[-1], zd1_arr[-1], z1_arr, zd1_arr, u_curr, a1)
forces2[-1] = calculate_force(nt-1, t_arr[-1], z2_arr[-1], zd2_arr[-1], z2_arr, zd2_arr, u_curr, a2)
print("Simulation complete.")

# --- Static Plots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the particle's dynamic trajectory comparison
ax1.plot(z1_arr, t_arr, color='green', lw=2, label='Particle 1')
ax1.plot(z2_arr, t_arr, color='orange', linestyle=':', linewidth=2, label='Particle 2')
ax1.set_xlabel('Position $x^1$')
ax1.set_ylabel('Time ($x^0$)')
ax1.set_title('Trajectories')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot Required/Calculated Force
ax2.plot(np.abs(z1_arr-z2_arr), forces1, color='green', lw=2, label='Distance v.s Force')
#ax2.plot(t_arr, forces2, color='orange', lw=2, label='Force on P2')
ax2.set_xlabel('Distance')
ax2.set_ylabel('Force $f(x^0, z, \\dot{z})$')
ax2.set_title('Calculated Force on Particles')
# ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# --- Animation Block ---
fig_ani, ax_ani = plt.subplots(figsize=(10, 6))
ax_ani.set_xlim(-10, 10)
ax_ani.set_ylim(-10, 10) 
ax_ani.set_xlabel('$x^1$')
ax_ani.set_ylabel('$u(x^0, x^1)$')
ax_ani.set_title('1D Klein-Gordon Interacting Charges')
ax_ani.grid(True, linestyle='--', alpha=0.6)

line, = ax_ani.plot([], [], lw=2, color='royalblue', label='Total Scalar Field $u$')
marker1, = ax_ani.plot([], [], 'go', markersize=6, label='Particle 1')
marker2, = ax_ani.plot([], [], 'ro', markersize=6, label='Particle 2')
time_text = ax_ani.text(0.02, 0.95, '', transform=ax_ani.transAxes, fontsize=12)
ax_ani.legend(loc='upper right')

def init():
    line.set_data([], [])
    marker1.set_data([], [])
    marker2.set_data([], [])
    time_text.set_text('')
    return line, marker1, marker2, time_text

def animate(i):
    u_current = history_u[i]
    t_current = times_anim[i]
    
    idx_t_nearest = np.argmin(np.abs(t_arr - t_current))
    z1_pos = z1_arr[idx_t_nearest]
    z2_pos = z2_arr[idx_t_nearest]
    
    line.set_data(x, u_current)
    
    idx_x_marker1 = np.argmin(np.abs(x - z1_pos))
    idx_x_marker2 = np.argmin(np.abs(x - z2_pos))
    marker1.set_data([z1_pos], [u_current[idx_x_marker1]])
    marker2.set_data([z2_pos], [u_current[idx_x_marker2]])
    
    time_text.set_text(f'Time: {t_current:.2f}')
    return line, marker1, marker2, time_text

ani = FuncAnimation(fig_ani, animate, frames=len(history_u), init_func=init, blit=True, interval=60)
plt.tight_layout()
plt.show()