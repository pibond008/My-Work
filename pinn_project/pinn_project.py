import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.interpolate import interp1d

def eos(P, K, gamma):
    if P <= 0:
        return 0.0, 0.0
    rho_b = (P / K) ** (1 / gamma)
    epsilon = rho_b + P / (gamma - 1)
    return rho_b, epsilon

def tov_eqs(r, m, P, K, gamma):
    if P <= 0:
        return np.array([0.0, 0.0, 0.0])
    rho_b, epsilon = eos(P, K, gamma)
    dm_dr = 4 * np.pi * r**2 * epsilon
    dP_dr = -(epsilon + P) * (m + 4 * np.pi * r**3 * P) / (r * (r - 2 * m))
    dphi_dr = (m + 4 * np.pi * r**3 * P) / (r * (r - 2 * m))
    return np.array([dm_dr, dP_dr, dphi_dr])

def rk4_step(r, y, dr, K, gamma):
    m, P, phi = y
    k1 = tov_eqs(r, m, P, K, gamma)
    k2 = tov_eqs(r + dr/2, m + dr*k1[0]/2, P + dr*k1[1]/2, K, gamma)
    k3 = tov_eqs(r + dr/2, m + dr*k2[0]/2, P + dr*k2[1]/2, K, gamma)
    k4 = tov_eqs(r + dr, m + dr*k3[0], P + dr*k3[1], K, gamma)
    dm = (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6
    dP = (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6
    dphi = (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6
    P_new = max(P + dr*dP, 0.0)
    return np.array([m + dr*dm, P_new, phi + dr*dphi])

def solve_tov_full(rho_c=1.28e-3, K=100, gamma=2, dr=0.05, r_max=12):
    P_c = K * rho_c**gamma
    _, eps_c = eos(P_c, K, gamma)
    r_int, m_int, P_int, phi_int = [0.0], [0.0], [P_c], [0.0]
    r0 = 1e-9
    m0 = (4.0/3.0) * np.pi * eps_c * r0**3
    P0 = P_c - (2.0/3.0) * np.pi * (eps_c + P_c) * (eps_c + 3.0*P_c) * r0**2
    phi0 = (2.0*np.pi/3.0) * (eps_c + 3.0*P_c) * r0**2
    y = np.array([m0, P0, phi0])
    r = r0
    r_int.append(r)
    m_int.append(y[0])
    P_int.append(y[1])
    phi_int.append(y[2])
    while y[1] > 1e-15 and r < r_max:
        y = rk4_step(r, y, dr, K, gamma)
        r += dr
        r_int.append(r)
        m_int.append(y[0])
        P_int.append(y[1])
        phi_int.append(y[2])
    R = r_int[-1]
    M = m_int[-1]
    phi_R = 0.5 * np.log(1 - 2*M/R)
    phi_shift = phi_R - phi_int[-1]
    phi_int = np.array(phi_int) + phi_shift
    r_ext = np.linspace(R, r_max, 600)
    phi_ext = 0.5 * np.log(1 - 2*M/r_ext)
    R_iso = R / (1 + M/(2*R))**2
    return M, R, R_iso, np.array(r_int), np.array(m_int), np.array(P_int), phi_int, r_ext, phi_ext

M, R, R_iso, r_int, m_int, P_int, phi_int, r_ext, phi_ext = solve_tov_full(
    rho_c=1.28e-3, K=100, gamma=2, dr=0.05, r_max=15
)

r_min, r_max = r_int.min(), r_int.max()
m_min, m_max = m_int.min(), m_int.max()
P_min, P_max = P_int.min(), P_int.max()

r_norm = 2.0 * (r_int - r_min) / (r_max - r_min) - 1.0
m_norm = 2.0 * (m_int - m_min) / (m_max - m_min) - 1.0
P_norm = 2.0 * (P_int - P_min) / (P_max - P_min) - 1.0

# Skip r=0 point
start_idx = np.searchsorted(r_int, 1e-6)
r_tf = tf.Variable(r_norm[start_idx:].reshape(-1, 1), dtype=tf.float64, trainable=False)
m_tf = tf.constant(m_norm[start_idx:].reshape(-1, 1), dtype=tf.float64)
P_tf = tf.constant(P_norm[start_idx:].reshape(-1, 1), dtype=tf.float64)

norm = {
    'r_min': tf.constant(r_min, dtype=tf.float64),
    'r_max': tf.constant(r_max, dtype=tf.float64),
    'm_min': tf.constant(m_min, dtype=tf.float64),
    'm_max': tf.constant(m_max, dtype=tf.float64),
    'P_min': tf.constant(P_min, dtype=tf.float64),
    'P_max': tf.constant(P_max, dtype=tf.float64)
}

# Initialize closer to true values
class PINN_TOV(tf.keras.Model):
    def __init__(self):
        super(PINN_TOV, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(64, activation='tanh', dtype=tf.float64)
        self.hidden2 = tf.keras.layers.Dense(64, activation='tanh', dtype=tf.float64)
        self.hidden3 = tf.keras.layers.Dense(64, activation='tanh', dtype=tf.float64)
        self.out = tf.keras.layers.Dense(2, activation=None, dtype=tf.float64)
        
        # Initialize closer to true values
        self.log_K = tf.Variable(np.log(90.0), dtype=tf.float64, trainable=True)
        self.log_rho_c = tf.Variable(np.log(1.2e-3), dtype=tf.float64, trainable=True)
        self.gamma_raw = tf.Variable(np.log(1.5), dtype=tf.float64, trainable=True)
    
    @property
    def gamma(self):
        return tf.nn.softplus(self.gamma_raw) + 1.1
    
    def call(self, r):
        x = self.hidden1(r)
        x = self.hidden2(x)
        x = self.hidden3(x)
        return self.out(x)

model = PINN_TOV()

def tov_loss(model, r_tf, m_tf, P_tf, norm, lam1=1.0, lam2=1.0, lam3=10.0):
    one = tf.constant(1.0, dtype=tf.float64)
    two = tf.constant(2.0, dtype=tf.float64)
    three = tf.constant(3.0, dtype=tf.float64)
    four = tf.constant(4.0, dtype=tf.float64)
    pi = tf.constant(np.pi, dtype=tf.float64)
    eps = tf.constant(1e-10, dtype=tf.float64)
    
    K = tf.exp(model.log_K)
    gamma = tf.nn.softplus(model.gamma_raw) + 1.1
    rho_c = tf.exp(model.log_rho_c)
    
    r_min = norm['r_min']
    r_max = norm['r_max']
    m_min = norm['m_min']
    m_max = norm['m_max']
    P_min = norm['P_min']
    P_max = norm['P_max']
    
    def denorm_r(r_n):
        return (r_n + one) / two * (r_max - r_min) + r_min
    def denorm_m(m_n):
        return (m_n + one) / two * (m_max - m_min) + m_min
    def denorm_P(P_n):
        return (P_n + one) / two * (P_max - P_min) + P_min
    
    # Data loss
    pred_data = model(r_tf)
    loss_data = tf.reduce_mean(tf.square(pred_data[:, 0:1] - m_tf)) + \
                tf.reduce_mean(tf.square(pred_data[:, 1:2] - P_tf))
    
    # Physics loss
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(r_tf)
        pred = model(r_tf)
        m_n = pred[:, 0:1]
        P_n = pred[:, 1:2]
        r_ph = denorm_r(r_tf)
        m_ph = denorm_m(m_n)
        P_ph = tf.maximum(denorm_P(P_n), eps)
        rho_b = tf.maximum(P_ph / K, eps) ** (one / gamma)
        epsilon = rho_b + P_ph / tf.maximum(gamma - one, eps)
    
    dm_n_dr_n = tape.gradient(m_n, r_tf)
    dP_n_dr_n = tape.gradient(P_n, r_tf)
    del tape
    
    dm_n_dr_n = tf.where(tf.math.is_nan(dm_n_dr_n), tf.zeros_like(dm_n_dr_n), dm_n_dr_n)
    dP_n_dr_n = tf.where(tf.math.is_nan(dP_n_dr_n), tf.zeros_like(dP_n_dr_n), dP_n_dr_n)
    
    dm_dr = dm_n_dr_n * (m_max - m_min) / (r_max - r_min)
    dP_dr = dP_n_dr_n * (P_max - P_min) / (r_max - r_min)
    
    denom = tf.maximum(r_ph * (r_ph - two * m_ph), eps)
    res_m = dm_dr - four * pi * r_ph**two * epsilon
    res_P = dP_dr + (epsilon + P_ph) * (m_ph + four * pi * r_ph**three * P_ph) / denom
    
    center_mask = tf.cast(r_ph > 1e-6, dtype=tf.float64)
    loss_tov = tf.reduce_mean(tf.square(res_m * center_mask)) + \
               tf.reduce_mean(tf.square(res_P * center_mask))
    
    # Boundary condition (stronger weight)
    r_bc = r_tf[0:1]
    pred_bc = model(r_bc)
    m_bc = denorm_m(pred_bc[:, 0:1])
    P_bc = denorm_P(pred_bc[:, 1:2])
    P_c = K * rho_c ** gamma
    
    loss_bc = tf.square(m_bc) + tf.square(P_bc - P_c)
    
    # Total loss with balanced weights
    loss_total = lam1 * loss_data + lam2 * loss_tov + lam3 * loss_bc
    
    return loss_total, loss_data, loss_tov, loss_bc

# Separate optimizers with different learning rates
lr_net = 1e-3
lr_phys = 1e-2  # Higher LR for physics parameters

# Single optimizer for all variables
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)

# Loss weights
lam1, lam2, lam3 = 1.0, 1.0, 10.0

history = {
    'loss_total': [], 'loss_data': [], 'loss_tov': [], 'loss_bc': [],
    'K': [], 'gamma': [], 'rho_c': []
}

print("Training PINN for TOV")
print("-" * 70)
print(f"True: K=100, gamma=2.0, rho_c=1.28e-3")
print(f"Init: K={tf.exp(model.log_K).numpy():.1f}, gamma={model.gamma.numpy():.2f}, rho_c={tf.exp(model.log_rho_c).numpy():.2e}")
print("-" * 70)

epochs = 8000
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss_total, loss_data, loss_tov, loss_bc = tov_loss(
            model, r_tf, m_tf, P_tf, norm, lam1, lam2, lam3
        )
    
    grads = tape.gradient(loss_total, model.trainable_variables)
    grads = [tf.where(tf.math.is_nan(g), tf.zeros_like(g), g) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    history['loss_total'].append(loss_total.numpy())
    history['loss_data'].append(loss_data.numpy())
    history['loss_tov'].append(loss_tov.numpy())
    history['loss_bc'].append(loss_bc.numpy())
    history['K'].append(tf.exp(model.log_K).numpy())
    history['gamma'].append(model.gamma.numpy())
    history['rho_c'].append(tf.exp(model.log_rho_c).numpy())
    
    if epoch % 1000 == 0:
        K_val = tf.exp(model.log_K).numpy()
        gamma_val = model.gamma.numpy()
        rho_c_val = tf.exp(model.log_rho_c).numpy()
        
        # Use .item() to extract scalar from 0-d array
        print(f"Epoch {epoch:5d} | Loss: {loss_total.numpy().item():.4e} | Data: {loss_data.numpy().item():.4e} | "
            f"TOV: {loss_tov.numpy().item():.4e} | BC: {loss_bc.numpy().item():.4e}")
        print(f"           K: {K_val:.2f} (true: 100) | gamma: {gamma_val:.3f} (true: 2.0) | "
            f"rho_c: {rho_c_val:.2e} (true: 1.28e-3)")

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
K_final = tf.exp(model.log_K).numpy()
gamma_final = model.gamma.numpy()
rho_c_final = tf.exp(model.log_rho_c).numpy()
print(f"K     : {K_final:.2f}   (true: 100.0, error: {abs(K_final-100)/100*100:.1f}%)")
print(f"gamma : {gamma_final:.3f}   (true: 2.0, error: {abs(gamma_final-2.0)/2.0*100:.1f}%)")
print(f"rho_c : {rho_c_final:.2e}   (true: 1.28e-3, error: {abs(rho_c_final-1.28e-3)/1.28e-3*100:.1f}%)")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].plot(history['loss_total'], label='Total')
axes[0, 0].plot(history['loss_data'], label='Data')
axes[0, 0].plot(history['loss_tov'], label='TOV')
axes[0, 0].set_yscale('log')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()

axes[0, 1].plot(history['K'], label='K')
axes[0, 1].axhline(100, color='r', linestyle='--', label='True')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('K')
axes[0, 1].legend()

axes[1, 0].plot(history['gamma'], label='gamma')
axes[1, 0].axhline(2.0, color='r', linestyle='--', label='True')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('gamma')
axes[1, 0].legend()

axes[1, 1].plot(history['rho_c'], label='rho_c')
axes[1, 1].axhline(1.28e-3, color='r', linestyle='--', label='True')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('rho_c')
axes[1, 1].legend()

plt.tight_layout()
plt.show()