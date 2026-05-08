import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
from scipy.special import j1


# --- Parameters ---
#try decrease the length and time for better graph
L = 2000
T = 1000
nx = 4001 #bigger        
dx = 2 * L / (nx - 1) 
dt = 0.5 * dx  
nt = int(T / dt)
C2 = (dt / dx)**2

mu = 1.5 
a = 2
m = 1.0               
sigma_src = 1   

v0_amp = 5
v0_sigma = 2
v0_center = 10.0      

x = np.linspace(-L, L, nx)
t_arr = np.linspace(0, T, nt)

# --- Dynamic Arrays ---
z_arr = np.zeros(nt)  
zd_arr = np.zeros(nt) 
p_arr = np.zeros(nt)  
forces = np.zeros(nt) 

# Seed the initial state using the prescribed trajectory
# z(0) = 0, z_dot(0) = 0
z_arr[0] = 0
zd_arr[0] = 0
# Set initial momentum p = mvgamma based on that initial trajectory
# Using np.maximum to ensure (1 - zd**2) stays positive just in case
gamma_inv_0 = np.sqrt(np.maximum(0.0, 1.0 - zd_arr[0]**2))
p_arr[0] = m * zd_arr[0] / np.clip(gamma_inv_0, 1e-15, None) 

# --- Helper Functions ---
def dirac_delta_approx(x_val, x0_pos, sig):
    return (1.0 / (sig * np.sqrt(np.pi))) * np.exp(-((x_val - x0_pos) / sig)**2)

def source(x_val, z_val, zd_val):
    gamma_inv_sq = np.clip(1.0 - zd_val**2, 0.0, None)
    return a * np.sqrt(gamma_inv_sq) * dirac_delta_approx(x_val, z_val, sigma_src)

# --- Force Calculation ---
def calculate_force(n, x0, V_field, U_field):
    z1_x0 = z_arr[n]
    zd_x0 = zd_arr[n]
    
    # 1. Background gradient & radiation damping (f0)
    dV_dx = np.gradient(V_field, dx)
    dV_dx_z = np.interp(z1_x0, x, dV_dx) 
    
    # Check for near-lightspeed safety 
    # it picked up the physic context :( 
    gamma_inv_sq = np.clip(1.0 - zd_x0**2, 1e-15, None) 
    
    # Damping proportional to \dot{z}/(1-\dot{z}^2)
    damping = (0.5 * a**2) * (zd_x0 / gamma_inv_sq)
    #massless force
    f0 = a * dV_dx_z - damping 
    
    # 2. History/Tail integral I1^2
    if n == 0:
        I1_2 = 0.0
    else:
        # History arrays up to current step n
        t_past = t_arr[:n+1]
        x1_past = z_arr[:n+1]
        zd_past = zd_arr[:n+1]
        
        # sigma = mu * sqrt((x^0 - t)^2 - (x^1 - z^1(x^0))^2)
        # Using z_arr[:n+1] directly here since x1_past IS z^1(t)
        sigma = mu * np.sqrt(np.maximum(0.0, (x0 - t_past)**2 - (x1_past - z1_x0)**2))
        
        # J1(sigma)/sigma -> 0.5 as sigma -> 0
        bessel_term = np.where(sigma < 1e-10, 0.5, j1(sigma) / sigma)
        
        gamma_inv_past = np.sqrt(np.maximum(0.0, 1.0 - zd_past**2))
        integrand = (z1_x0 - x1_past) * gamma_inv_past * bessel_term
        
        I1_2 = (a * (mu**2) / 2.0) * np.trapezoid(integrand, t_past)

    # 3. Static U field (\partial_1 U)
    dU_dx = np.gradient(U_field, dx)
    dU_dx_z = np.interp(z1_x0, x, dU_dx)
    
    # 4. Total Force Assembly
    gamma_inv_x0 = np.sqrt(np.maximum(0.0, 1.0 - zd_x0**2))
    # f = f0 + (I1_2 + \partial_1 V + \partial_1 U)(a * sqrt(1 - \dot{z}^2))
    f_total = f0 + (I1_2 + dV_dx_z + dU_dx_z) * (a * gamma_inv_x0)
    
    return f_total

# --- Field Initialization ---
u_prev, u_curr, u_next = np.zeros(nx), np.zeros(nx), np.zeros(nx)
V_prev, V_curr, V_next = np.zeros(nx), np.zeros(nx), np.zeros(nx)
U_prev, U_curr, U_next = np.zeros(nx), np.zeros(nx), np.zeros(nx)

# Initial Gaussian Pulse (V)
V0 = v0_amp * np.exp(-((x - v0_center)**2) / (2 * v0_sigma**2))
V1 = -((x - v0_center) / (v0_sigma**2)) * V0 # Initial velocity from derivative

# Initial Static Field (U) centered on given z_arr[0]
U0 = (a / (2 * mu)) * np.exp(-mu * np.abs(x - z_arr[0])) 

V_curr[:] = V0[:]
U_curr[:] = U0[:]
u_curr[:] = U0[:] + V0[:]

# Prepare histories for plotting/animation
frame_step = max(1, nt // 400) # Save every ~400th frame for efficiency
history_u = [u_curr.copy()]
times_anim = [0.0]

# --- THE MAIN LOOP ---
print("Simulating... Started from defined trajectory, updating dynamically.")
for n in range(nt - 1): # Corrected range: goes from 0 up to nt-2
    
    # 1. Start with current Z (seeded by initial trajectory at n=0, then dynamic)
    S = source(x, z_arr[n], zd_arr[n])
    
    # 2. Calculate fields (u_next) based on that Z position
    if n == 0:
        # Step 1 special case: uses given initial velocity V1
        V_next[1:-1] = (V_curr[1:-1] + dt * V1[1:-1] + 
                        0.5 * C2 * (V_curr[2:] - 2*V_curr[1:-1] + V_curr[:-2]) -
                        0.5 * (dt**2) * (mu**2) * V_curr[1:-1])
        U_next[1:-1] = (U_curr[1:-1] + 
                        0.5 * C2 * (U_curr[2:] - 2*U_curr[1:-1] + U_curr[:-2]) -
                        0.5 * (dt**2) * (mu**2) * U_curr[1:-1])
        u_next[1:-1] = (u_curr[1:-1] + dt * V1[1:-1] + 
                        0.5 * C2 * (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2]) -
                        0.5 * (dt**2) * (mu**2) * u_curr[1:-1] +
                        0.5 * (dt**2) * S[1:-1])
    else:
        # Standard Finite Difference steps
        V_next[1:-1] = (2 * V_curr[1:-1] - V_prev[1:-1] +
                        C2 * (V_curr[2:] - 2*V_curr[1:-1] + V_curr[:-2]) -
                        (dt**2) * (mu**2) * V_curr[1:-1])
        U_next[1:-1] = (2 * U_curr[1:-1] - U_prev[1:-1] +
                        C2 * (U_curr[2:] - 2*U_curr[1:-1] + U_curr[:-2]) -
                        (dt**2) * (mu**2) * U_curr[1:-1])
        u_next[1:-1] = (2 * u_curr[1:-1] - u_prev[1:-1] +
                        C2 * (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2]) -
                        (dt**2) * (mu**2) * u_curr[1:-1] +
                        (dt**2) * S[1:-1])
        
    # Boundary Conditions 
    V_next[0] = 0; V_next[-1] = 0
    U_next[0] = 0; U_next[-1] = 0
    u_next[0] = 0; u_next[-1] = 0

    # 3. Calculate Force using newly calculated fields for x^0 = t_arr[n+1]
    forces[n] = calculate_force(n, t_arr[n+1], V_next, U_next)

    # 4. Dynamically update Z for the next step (n+1) based entirety on the FORCE
    p_arr[n+1] = p_arr[n] + dt * forces[n]
    
    # Velocity \dot{z} = p / (m * sqrt(1 + (p/m)^2))
    # Corrected relativistic conversion handling clipping for stability
    p_over_m = p_arr[n+1] / m
    zd_arr[n+1] = p_over_m / np.sqrt(np.maximum(0.0, 1.0 + p_over_m**2))
    
    # Position update
    z_arr[n+1] = z_arr[n] + dt * zd_arr[n+1]

    # 5. Repeat: Rotate fields
    u_prev[:], u_curr[:] = u_curr[:], u_next[:]
    V_prev[:], V_curr[:] = V_curr[:], V_next[:]
    U_prev[:], U_curr[:] = U_curr[:], U_next[:]

    # Store animation frame
    if n % frame_step == 0:
        history_u.append(u_curr.copy())
        times_anim.append(t_arr[n+1])

# Calculate very last force point for consistency in graphs
forces[-1] = calculate_force(nt-1, t_arr[-1], V_curr, U_curr)
print("Simulation complete.")

# --- Static Plots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the particle's dynamic trajectory comparison
ax1.plot(z_arr, t_arr, color='green', lw=2, label='Dynamic Trajectory')
ax1.set_xlabel('Position $x^1$')
ax1.set_ylabel('Time ($x^0$)')
ax1.set_title('Trajectory')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot Required/Calculated Force
ax2.plot(t_arr, forces, color='purple', lw=2)
ax2.set_xlabel('Time ($x^0$)')
ax2.set_ylabel('Force $f(x^0, z, \\dot{z})$')
ax2.set_title('Calculated Force on Particle')
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# --- NEW: Animation Block ---
fig_ani, ax_ani = plt.subplots(figsize=(10, 6))
# Set appropriate limits for the scalar field u
ax_ani.set_xlim(-L, L)
ax_ani.set_ylim(-1.0, 2.5) # Based on U_static (a/2) + V_Gaussian (v0_amp)
ax_ani.set_xlabel('$x^1$')
ax_ani.set_ylabel('$u(x^0, x^1)$')
ax_ani.set_title('1D Klein-Gordon with Dynamic Source (Loop Sequence Optimized)')
ax_ani.grid(True, linestyle='--', alpha=0.6)

# Line object for the scalar field
line, = ax_ani.plot([], [], lw=2, color='royalblue', label='Total Scalar Field $u$')
# Marker object for the dynamic particle position z(t)
source_marker, = ax_ani.plot([], [], 'ro', markersize=6, label='Dynamic Particle $z(x^0)$')
# Text object for the timestamp
time_text = ax_ani.text(0.02, 0.95, '', transform=ax_ani.transAxes, fontsize=12)
ax_ani.legend(loc='upper right')

def init():
    line.set_data([], [])
    source_marker.set_data([], [])
    time_text.set_text('')
    return line, source_marker, time_text

def animate(i):
    # Map animation frame index to correct index in time array (t_arr)
    # Using 'history_u' and 'times_anim' index 'i' directly
    u_current = history_u[i]
    t_current = times_anim[i]
    
    # Need the precise position from the dynamic z_arr for this time
    # Rounding time slightly to find nearest step index
    idx_t_nearest = np.argmin(np.abs(t_arr - t_current))
    z_pos = z_arr[idx_t_nearest]
    
    # Update field line
    line.set_data(x, u_current)
    
    # Find nearest grid point for the marker (plotting on top of the field line)
    idx_x_marker = np.argmin(np.abs(x - z_pos))
    source_marker.set_data([z_pos], [u_current[idx_x_marker]])
    
    # Update timestamp
    time_text.set_text(f'Time: {t_current:.2f}')
    return line, source_marker, time_text

# Create the animation object
# Interval 30ms -> ~33 FPS
ani = FuncAnimation(fig_ani, animate, frames=len(history_u), init_func=init, blit=True, interval=30)
plt.tight_layout()
plt.show()