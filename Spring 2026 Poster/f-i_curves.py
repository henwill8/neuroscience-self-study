import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# ----------- USER-ADJUSTABLE PARAMETERS ------------------
# =========================================================

FI_TYPE = "threshold_linear"  # choose: "linear", "threshold_linear", "sigmoid", "power"

# Network weights
w_EE = 1.5
w_EI = 1.0
w_IE = 1.2
w_II = 0.5

I_E_ext = 0.5
I_I_ext = 0.2

# Gain parameters (used depending on FI_TYPE)
gain_E = 1
gain_I = 5
theta_E = 0.0
theta_I = 25.0
max_E = 50
max_I = 80
power_exp = 2.0


# =========================================================
# ------------------- f-I CURVES --------------------------
# =========================================================

def F(I, gain, theta, max_rate):
    if FI_TYPE == "linear":
        return gain * I

    elif FI_TYPE == "threshold_linear":
        return gain * np.maximum(0, I - theta)

    elif FI_TYPE == "sigmoid":
        return max_rate / (1 + np.exp(-gain * (I - theta)))

    elif FI_TYPE == "power":
        return gain * np.maximum(0, I - theta) ** power_exp


def F_inv(r, gain, theta, max_rate):
    r = np.clip(r, 1e-6, max_rate - 1e-6)

    if FI_TYPE == "linear":
        return r / gain

    elif FI_TYPE == "threshold_linear":
        return r / gain + theta

    elif FI_TYPE == "sigmoid":
        return theta + (1 / gain) * np.log(r / (max_rate - r))

    elif FI_TYPE == "power":
        return theta + (r / gain) ** (1 / power_exp)


# =========================================================
# ------------------ COMPUTE NULLCLINES -------------------
# =========================================================

rE_vals = np.linspace(0, max_E, 500)
rI_vals = np.linspace(0, max_I, 500)

# E-nullcline
FE_inv_vals = F_inv(rE_vals, gain_E, theta_E, max_E)
rI_null_E = (w_EE * rE_vals + I_E_ext - FE_inv_vals) / w_EI

# I-nullcline
FI_inv_vals = F_inv(rI_vals, gain_I, theta_I, max_I)
rE_null_I = (FI_inv_vals + w_II * rI_vals - I_I_ext) / w_IE

# =========================================================
# ---------------------- PLOTTING -------------------------
# =========================================================

plt.figure(figsize=(12, 5))

# ---- Plot f-I curves ----
plt.subplot(1, 2, 1)

I_vals = np.linspace(-1, 5, 500)
plt.plot(I_vals, F(I_vals, gain_E, theta_E, max_E), label="Excitatory f-I")
plt.plot(I_vals, F(I_vals, gain_I, theta_I, max_I), label="Inhibitory f-I")

plt.xlabel("Input Current")
plt.ylabel("Firing Rate")
plt.title(f"f-I Curves ({FI_TYPE})")
plt.legend()

# ---- Plot Nullclines ----
plt.subplot(1, 2, 2)

plt.plot(rE_vals, rI_null_E, label="E-nullcline")
plt.plot(rE_null_I, rI_vals, label="I-nullcline")

plt.xlabel("r_E")
plt.ylabel("r_I")
plt.title("E-I Phase Plane")
plt.xlim(0, max_E)
plt.ylim(0, max_I)
plt.legend()

plt.tight_layout()
plt.show()
