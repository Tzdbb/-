import numpy as np
from sklearn.preprocessing import StandardScaler

def nonlinear_mapping(Q, L=100, epsilon=1.0):
    """
    Non-linear mapping function: Maps quantized result Q to [0, L] range with rounding.
    """
    Q = np.array(Q, dtype=np.float64)
    Q_max = np.max(Q) + 1e-12  # Avoid division by zero
    M = np.round(L * (1 - np.exp(-epsilon * (Q / Q_max) ** 2)))  # Use rounding
    return M

class AdaptiveQuantizationKernel:
    """
    Kernel-driven adaptive quantization method as described in the document.
    """

    def __init__(self, M=100, alpha=0.3, L=100, epsilon=1.0,
                 beta1=2.0, beta2=2.0, beta3=2.0,
                 theta=0.3, d=0.8, t_max=100):
        self.M = M
        self.alpha = alpha
        self.L = L
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.theta = theta
        self.d = d
        self.t_max = t_max
        self.global_max = None
        self.scaler = StandardScaler()

    def fit(self, X_train):
        max_values = []
        for sample in X_train:
            x_abs = np.abs(sample.astype(np.complex128))
            if not np.allclose(x_abs, 0):
                max_values.append(np.max(x_abs))
        self.global_max = np.max(max_values) if max_values else 1.0
        return self

    def _build_histogram(self, x_abs, global_max):
        """
        Builds a histogram and computes the data density using the global max amplitude.
        """
        n0 = np.argmax(x_abs)
        x_max = global_max
        N = len(x_abs)

        delta_b = x_max / self.M
        bins = np.arange(0, x_max + delta_b, delta_b)

        bin_indices = np.digitize(x_abs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.M - 1)

        hist = np.zeros(self.M)
        for idx in bin_indices:
            hist[idx] += 1

        H = hist / (N * delta_b + 1e-12)
        rho = H[bin_indices]
        return delta_b, bins, bin_indices, rho, H, n0, x_max

    def _compute_kernel(self, x_abs, n0, x_max, rho):
        """
        Computes the combined kernel function.
        """
        N = len(x_abs)
        rho_n0 = rho[n0]
        f_rho = np.exp(-self.beta1 * ((rho - rho_n0) / (rho_n0 + 1e-12)) ** 2)
        distances = np.arange(N) - n0
        f_d = np.exp(-self.beta2 * (distances / N) ** 2)
        amplitude_diff = (x_abs - x_max) / (x_max + 1e-12)
        f_a = np.exp(-self.beta3 * amplitude_diff ** 2)

        combined_kernel = f_rho * f_d * f_a
        return combined_kernel
