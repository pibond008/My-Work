import tensorflow as tf
tf.keras.backend.set_floatx('float32')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display

def eos(P, K, gamma):
    if P <= 0:
        return 0.0, 0.0
    rho_b = (P / K) ** (1 / gamma)
    epsilon = rho_b + P / (gamma - 1)
    return rho_b, epsilon

def solve_tov_full(rho_c=1.28e-3, K=100, gamma=2, dr=0.05, r_max=15):
    P_c = K * rho_c ** gamma
    _, eps_c = eos(P_c, K, gamma)

    r_int, m_int, P_int = [0.0], [0.0], [P_c]

    r0 = 1e-6
    m0 = (4.0 / 3.0) * np.pi * eps_c * r0 ** 3
    P0 = P_c - (2.0 / 3.0) * np.pi * (eps_c + P_c) * (eps_c + 3.0 * P_c) * r0 ** 2

    r, m, P = r0, m0, P0
    r_int.append(r); m_int.append(m); P_int.append(P)

    while P > 1e-12 and r < r_max:
        rho_b, epsilon = eos(P, K, gamma)
        dm_dr = 4 * np.pi * r ** 2 * epsilon
        denom = r * (r - 2 * m)
        denom = denom if abs(denom) > 1e-10 else 1e-10
        dP_dr = -(epsilon + P) * (m + 4 * np.pi * r ** 3 * P) / denom

        m += dr * dm_dr
        P = max(P + dr * dP_dr, 0.0)
        r += dr
        r_int.append(r); m_int.append(m); P_int.append(P)

    return m_int[-1], r_int[-1], np.array(r_int), np.array(m_int), np.array(P_int)


def gen_tov_data(rho_c, K, gamma, dr=0.05, r_max=15):
    M, R, r, m, P = solve_tov_full(rho_c, K, gamma, dr, r_max)
    mask = r > 1e-6
    r, m, P = r[mask], m[mask], P[mask]
    return r, m, P, M, R

A = np.array([[0.25, 0.25 - np.sqrt(3)/6],
              [0.25 + np.sqrt(3)/6, 0.25]])
b = np.array([0.5, 0.5])
c = np.array([0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6])
q = len(b)

lr_nn = tf.keras.optimizers.schedules.ExponentialDecay(
    2e-3,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=False
)

lr_phys = tf.keras.optimizers.schedules.ExponentialDecay(
    5e-5,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=False
)

optimiser_nn = tf.keras.optimizers.Adam(learning_rate=lr_nn)
optimiser_phys = tf.keras.optimizers.Adam(learning_rate=lr_phys)

def constrained_outputs(model, r_n, norm):

    raw = model(r_n)
    raw_m = raw[:, 0:1]
    raw_P = raw[:, 1:2]

    m_scale = norm['m_scale']
    P_scale = norm['P_scale']
    Pc_obs = norm['Pc_obs']

    # Mass: enforce m(0)=0.
    # Do NOT use softplus here; it was too restrictive.
    m_n = tf.pow(r_n, 3) * raw_m
    m_ph = m_n * m_scale

    # Pressure: enforce P(0)=Pc_obs and P(1)=0.
    # Positive pressure inside star.
    P_ph = Pc_obs * (1.0 - r_n) * tf.exp(r_n * raw_P)
    P_n = P_ph / P_scale

    return m_n, P_n, m_ph, P_ph

@tf.function(jit_compile=False)
def tov_inv_loss(model, r_n, m_n_train, P_n_train, norm,
                 lam_data, lam_tov, lam_pc):

    pi = tf.constant(np.pi, dtype=tf.float32)
    eps = tf.constant(1e-12, dtype=tf.float32)

    r_scale = norm['r_scale']
    m_scale = norm['m_scale']
    P_scale = norm['P_scale']
    Pc_obs = norm['Pc_obs']

    K, rho_c, gamma = model.physics_params()
    P_c = K * tf.pow(rho_c, gamma)

    Pc_obs = norm['Pc_obs']

    pc_loss = tf.reduce_mean(tf.square((P_c - Pc_obs) / Pc_obs))

    # Data loss
    m_n_pred, P_n_pred, _, _ = constrained_outputs(model, r_n, norm)

    data_loss = (
        tf.reduce_mean(tf.square(m_n_pred - m_n_train)) +
        tf.reduce_mean(tf.square(P_n_pred - P_n_train))
    )

    # Help identify central pressure combination K rho_c^gamma
    pc_loss = tf.reduce_mean(tf.square((P_c - Pc_obs) / Pc_obs))

    # Physics loss
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(r_n)
        m_n, P_n, m_ph, P_ph = constrained_outputs(model, r_n, norm)

    dm_n_dr_n = tape.gradient(m_n, r_n)
    dP_n_dr_n = tape.gradient(P_n, r_n)
    del tape

    r_ph = r_n * r_scale

    dm_dr = dm_n_dr_n * m_scale / r_scale
    dP_dr = dP_n_dr_n * P_scale / r_scale

    P_safe = tf.maximum(P_ph, eps)

    rho_b = tf.pow(P_safe / K, 1.0 / gamma)
    epsilon = rho_b + P_safe / (gamma - 1.0)

    denom = r_ph * (r_ph - 2.0 * m_ph)
    denom = tf.where(
        tf.abs(denom) < tf.constant(1e-10, dtype=tf.float32),
        tf.constant(1e-10, dtype=tf.float32) * tf.ones_like(denom),
        denom
    )

    res_m = dm_dr - 4.0 * pi * r_ph**2 * epsilon

    res_P = dP_dr + (epsilon + P_safe) * (
        m_ph + 4.0 * pi * r_ph**3 * P_safe
    ) / denom

    tov_loss = (
        tf.reduce_mean(tf.square(res_m / (m_scale / r_scale))) +
        tf.reduce_mean(tf.square(res_P / (P_scale / r_scale)))
    )

    total_loss = (
        lam_data * data_loss +
        lam_tov * tov_loss +
        lam_pc * pc_loss
    )

    return total_loss, data_loss, tov_loss, pc_loss

def logit(x):
    return np.log(x / (1.0 - x))


class TOVModelInverse(tf.keras.models.Model):
    def __init__(self, base_model, K_init, rho_c_init, gamma_init):
        super(TOVModelInverse, self).__init__()

        self.model = base_model

        self.log_K = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(np.log(K_init)),
            dtype=tf.float32,
            trainable=True,
            name='log_K'
        )

        self.log_rho_c = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(np.log(rho_c_init)),
            dtype=tf.float32,
            trainable=True,
            name='log_rho_c'
        )

        # Soft bounded gamma: gamma in [1.1, 3.0]
        gamma_min = 1.1
        gamma_max = 3.0
        y = (gamma_init - gamma_min) / (gamma_max - gamma_min)
        y = np.clip(y, 1e-6, 1.0 - 1e-6)

        self.gamma_raw = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(logit(y)),
            dtype=tf.float32,
            trainable=True,
            name='gamma_raw'
        )

    def physics_params(self):
        K = tf.exp(self.log_K)
        rho_c = tf.exp(self.log_rho_c)

        gamma_min = tf.constant(1.1, dtype=tf.float32)
        gamma_max = tf.constant(3.0, dtype=tf.float32)
        gamma = gamma_min + (gamma_max - gamma_min) * tf.sigmoid(self.gamma_raw)

        return K, rho_c, gamma

    def call(self, r_n, **kwargs):
        return self.model(r_n)


tovnet = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(1,), dtype=tf.float32),
    tf.keras.layers.Dense(128, activation='tanh', dtype=tf.float32),
    tf.keras.layers.Dense(128, activation='tanh', dtype=tf.float32),
    tf.keras.layers.Dense(128, activation='tanh', dtype=tf.float32),
    tf.keras.layers.Dense(128, activation='tanh', dtype=tf.float32),
    tf.keras.layers.Dense(2, dtype=tf.float32)
])

model = TOVModelInverse(
    tovnet,
    K_init=50.0,
    rho_c_init=4.8e-4,
    gamma_init=1.5
)

# Build once
_ = model(tf.constant([[0.0]], dtype=tf.float32))

@tf.function(jit_compile=False)
def inv_train_one_step_tov(model, r_n, m_n_train, P_n_train, norm,
                           lam_data, lam_tov, lam_pc,
                           opt_nn, opt_phys):

    nn_vars = model.model.trainable_variables
    phys_vars = [model.log_K, model.log_rho_c, model.gamma_raw]

    with tf.GradientTape() as tape:
        loss_value, data_l, tov_l, pc_l = tov_inv_loss(
            model,
            r_n,
            m_n_train,
            P_n_train,
            norm,
            lam_data,
            lam_tov,
            lam_pc
        )

    grads = tape.gradient(loss_value, nn_vars + phys_vars)

    grads_nn = grads[:len(nn_vars)]
    grads_phys = grads[len(nn_vars):]

    grads_nn = [
        tf.clip_by_norm(g, 10.0) if g is not None else None
        for g in grads_nn
    ]

    grads_phys = [
        tf.clip_by_norm(g, 1.0) if g is not None else None
        for g in grads_phys
    ]

    nn_pairs = [(g, v) for g, v in zip(grads_nn, nn_vars) if g is not None]
    phys_pairs = [(g, v) for g, v in zip(grads_phys, phys_vars) if g is not None]

    if len(nn_pairs) > 0:
        opt_nn.apply_gradients(nn_pairs)

    if len(phys_pairs) > 0:
        opt_phys.apply_gradients(phys_pairs)

    return loss_value, data_l, tov_l, pc_l

history = np.array([])

r_train, m_train, P_train, M_true, R_true = gen_tov_data(
    rho_c=1.28e-3, K=100, gamma=2, dr=0.05, r_max=15
)

r_scale = r_train.max()
m_scale = m_train.max()
P_scale = P_train.max()

norm_consts = {
    'r_scale': tf.constant(r_scale, dtype=tf.float32),
    'm_scale': tf.constant(m_scale, dtype=tf.float32),
    'P_scale': tf.constant(P_scale, dtype=tf.float32),
    'Pc_obs':  tf.constant(P_train[0], dtype=tf.float32)
}

R_train = tf.convert_to_tensor((r_train / r_scale).reshape(-1, 1), dtype=tf.float32)
M_train = tf.convert_to_tensor((m_train / m_scale).reshape(-1, 1), dtype=tf.float32)
P_train_n = tf.convert_to_tensor((P_train / P_scale).reshape(-1, 1), dtype=tf.float32)

r_test = np.linspace(0.0, r_train.max(), 300)
R_test = tf.convert_to_tensor((r_test / r_scale).reshape(-1, 1), dtype=tf.float32)

def irk_predict(model, r_test, norm, h, K, gamma, rho_c):
    """Integrate IRK from r=0 to r_test using learned network"""
    pi = tf.constant(np.pi, dtype=tf.float32)  # ← float32
    r_scale = norm['r_scale']
    m_scale = norm['m_scale']
    P_scale = norm['P_scale']

    r_curr = tf.constant(0.0, dtype=tf.float32)  # ← float32
    m_curr = tf.constant(0.0, dtype=tf.float32)
    P_curr = K * tf.pow(rho_c, gamma)

    r_test_list = []
    m_test_list = []
    P_test_list = []

    while r_curr < r_test[-1]:
        r_n = tf.constant([[r_curr / r_scale]], dtype=tf.float32)  # ← float32
        stages_n = model(r_n)
        stages_n = tf.reshape(stages_n, (1, q, 2))
        m_stages_n = stages_n[:, :, 0:1]
        P_stages_n = stages_n[:, :, 1:2]

        # Compute stage derivatives
        dm_dr_stages = []
        dP_dr_stages = []
        for i in range(q):
            r_stage = r_curr + c[i] * h  # ← All float32 now
            y_stage_m = m_stages_n[0, 0:1, :] + h * tf.reduce_sum(
                A[i, :] * tf.stack(dm_dr_stages, axis=0), axis=0
            ) if dm_dr_stages else m_stages_n[0, 0:1, :]
            y_stage_P = P_stages_n[0, 0:1, :] + h * tf.reduce_sum(
                A[i, :] * tf.stack(dP_dr_stages, axis=0), axis=0
            ) if dP_dr_stages else P_stages_n[0, 0:1, :]

            P_safe = tf.maximum(y_stage_P, 1e-12)
            rho_b = tf.pow(P_safe / K, 1.0 / gamma)
            epsilon = rho_b + P_safe / (gamma - 1.0)

            dm_dr = 4.0 * pi * r_stage**2 * epsilon
            denom = r_stage * (r_stage - 2.0 * y_stage_m)
            denom = tf.where(tf.abs(denom) < 1e-10, 1e-10, denom)
            dP_dr = -(epsilon + P_safe) * (y_stage_m + 4.0 * pi * r_stage**3 * P_safe) / denom

            dm_dr_stages.append(dm_dr)
            dP_dr_stages.append(dP_dr)

        dm_dr_stages = tf.stack(dm_dr_stages, axis=0)
        dP_dr_stages = tf.stack(dP_dr_stages, axis=0)

        # Update
        m_curr = m_curr + h * tf.reduce_sum(b * dm_dr_stages, axis=0)
        P_curr = P_curr + h * tf.reduce_sum(b * dP_dr_stages, axis=0)
        r_curr = r_curr + h

        r_test_list.append(r_curr.numpy())
        m_test_list.append(m_curr.numpy())
        P_test_list.append(P_curr.numpy())

    return np.array(r_test_list), np.array(m_test_list), np.array(P_test_list)


history = []
history_data = []
history_tov = []
history_pc = []

num_epochs = 10000

pretrain_epochs = 1000
ramp_epochs = 3000

lam_tov_max = 1e-3   # important: do NOT ramp to 1.0 yet

for epoch in range(num_epochs):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 3.5))

    if epoch < pretrain_epochs:
        ramp = 0.0
    else:
        ramp = min(1.0, (epoch - pretrain_epochs) / ramp_epochs)

    lam_data = tf.constant(100.0, dtype=tf.float32)
    lam_tov = tf.constant(lam_tov_max * ramp, dtype=tf.float32)
    lam_pc = tf.constant(1.0 * ramp, dtype=tf.float32)

    loss, data_l, tov_l, pc_l = inv_train_one_step_tov(
        model,
        R_train,
        M_train,
        P_train_n,
        norm_consts,
        lam_data,
        lam_tov,
        lam_pc,
        optimiser_nn,
        optimiser_phys
    )

    history.append(loss.numpy())
    history_data.append(data_l.numpy())
    history_tov.append(tov_l.numpy())
    history_pc.append(pc_l.numpy())

    if epoch % 100 == 0:
        K_val, rho_c_val, gamma_val = model.physics_params()
        K_val = K_val.numpy()
        rho_c_val = rho_c_val.numpy()
        gamma_val = gamma_val.numpy()

        m_n_test, P_n_test, m_ph_test, P_ph_test = constrained_outputs(
            model, R_test, norm_consts
        )

        m_int = m_ph_test.numpy()
        P_int = P_ph_test.numpy()

        ax1.cla()
        ax1.set_yscale('log')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax1.plot(history, label='total')
        ax1.plot(history_data, label='data', alpha=0.7)
        ax1.plot(history_tov, label='TOV', alpha=0.7)
        ax1.plot(history_pc, label='Pc', alpha=0.7)
        ax1.legend(fontsize=7)

        ax1.set_title(
            f'$K={K_val:.3f}$ ({abs(K_val-100)/100*100:.1f}%)\n'
            f'$\\rho_c={rho_c_val:.3e}$ ({abs(rho_c_val-1.28e-3)/1.28e-3*100:.1f}%)\n'
            f'$\\gamma={gamma_val:.4f}$ ({abs(gamma_val-2.0)/2.0*100:.1f}%)',
            fontsize=10
        )

        ax2.cla()
        ax2.set_xlabel('r')
        ax2.set_ylabel('m(r)')
        ax2.plot(r_test, m_int, label='m(r) NN')
        ax2.scatter(r_train, m_train, s=10, color='orange', label='m data')
        ax2.legend(fontsize=8)

        ax3.cla()
        ax3.set_xlabel('r')
        ax3.set_ylabel('P(r)')
        ax3.plot(r_test, P_int, label='P(r) NN', color='green')
        ax3.scatter(r_train, P_train, s=10, color='red', label='P data')
        ax3.legend(fontsize=8)

        display.display(plt.gcf())
        display.clear_output(wait=True)
        plt.tight_layout()