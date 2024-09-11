import numpy as np
from scipy import optimize
from scipy.stats import laplace


def compute_mu_uniform_laplace(epochs, noise_multi, sample_rate):
    """Tính mu từ lấy mẫu con theo phân phối đều với nhiễu Laplace."""
    T = epochs / sample_rate
    c = np.sqrt(T) * sample_rate
    return np.sqrt(2) * c * np.sqrt(
        np.exp(noise_multi ** (-2)) * laplace.cdf(1.5 / noise_multi) + 3 * laplace.cdf(-0.5 / noise_multi) - 2
    )


def compute_mu_poisson_laplace(epochs, noise_multi, sample_rate):
    """Tính mu từ lấy mẫu con theo phân phối Poisson với nhiễu Laplace."""
    T = epochs / sample_rate
    return np.sqrt(np.exp(noise_multi ** (-2)) - 1) * np.sqrt(T) * sample_rate


def delta_eps_mu_laplace(eps, mu):
    """Tính dual giữa mu-GDP và (epsilon, delta)-DP cho nhiễu Laplace."""
    return laplace.cdf(-eps / mu + mu / 2) - np.exp(eps) * laplace.cdf(-eps / mu - mu / 2)


def eps_from_mu_laplace(mu, delta, bracket=(0, 500)):
    """Tính epsilon từ mu với delta thông qua dual ngược cho nhiễu Laplace."""

    def f(x):
        """Giải ngược dual bằng cách khớp delta."""
        return delta_eps_mu_laplace(x, mu) - delta

    return optimize.root_scalar(f, bracket=bracket, method='brentq').root


def compute_eps_uniform_laplace(epochs, noise_multi, sample_rate, delta):
    """Tính epsilon với delta từ dual ngược của lấy mẫu con theo phân phối đều với nhiễu Laplace."""
    return eps_from_mu_laplace(compute_mu_uniform_laplace(epochs, noise_multi, sample_rate), delta)


def compute_eps_poisson_laplace(epochs, noise_multi, sample_rate, delta):
    """Tính epsilon với delta từ dual ngược của lấy mẫu con theo phân phối Poisson với nhiễu Laplace."""
    return eps_from_mu_laplace(compute_mu_poisson_laplace(epochs, noise_multi, sample_rate), delta)


def get_noise_multiplier_laplace(
    sample_rate,
    epochs,
    target_epsilon,
    target_delta,
    sigma_min=0.01,
    sigma_max=10.0,
    threshold=1e-3,
):
    """Ước lượng hệ số nhiễu bằng tìm kiếm nhị phân cho nhiễu Laplace."""
    while sigma_max - sigma_min > threshold:
        sigma_mid = (sigma_min + sigma_max) / 2.
        epsilon = compute_eps_poisson_laplace(epochs, sigma_mid, sample_rate, target_delta)
        if epsilon > target_epsilon:
            sigma_min = sigma_mid
        else:
            sigma_max = sigma_mid
    return sigma_max
