import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 999)

from scipy.stats import norm


class Evaluator:
    """
    Evaluates DeepARCH predictions against realized volatility proxies.

    Parameters
    ----------
    compare_variance : bool
        If True, compare sigma^2 (variance) scale.
        If False (default), compare sigma (volatility) scale.
    bandwidth : int or None
        Bandwidth H for realized kernel estimators.
        If None, uses H = ceil(M^(2/3)) where M is the number of intraday returns.
    """

    def __init__(self, compare_variance: bool = False, bandwidth: int = None):
        self.compare_variance = compare_variance
        self.bandwidth = bandwidth

    # ------------------------------------------------------------------
    # Volatility proxies   (all return sigma^2 estimates, shape (N,1))
    # ------------------------------------------------------------------

    @staticmethod
    def realized_variance(intraday_returns: np.ndarray) -> np.ndarray:
        """
        RV_t = sum_{i=1}^{M} r_{t,i}^2
        intraday_returns : (N, M)  — 1-min returns within each 15-min bar.
        """
        return np.sum(intraday_returns ** 2, axis=1, keepdims=True)           # (N, 1)

    @staticmethod
    def bipower_variation(intraday_returns: np.ndarray) -> np.ndarray:
        """
        BV_t = (pi/2) * sum_{i=2}^{M} |r_{t,i}| * |r_{t,i-1}|
        Consistent estimator of IV robust to jumps.
        """
        mu1 = np.sqrt(2 / np.pi)                      # E[|Z|], Z~N(0,1)
        factor = 1.0 / mu1 ** 2                        # = pi/2
        abs_r  = np.abs(intraday_returns)
        bv     = factor * np.sum(abs_r[:, 1:] * abs_r[:, :-1], axis=1, keepdims=True)
        return bv                                      # (N, 1)

    @staticmethod
    def median_realized_volatility(intraday_returns: np.ndarray) -> np.ndarray:
        """
        MedRV_t = (pi / (6 - 4*sqrt(3) + pi)) * (M/(M-2))
                  * sum_{i=2}^{M-1} median(|r_{t,i-1}|, |r_{t,i}|, |r_{t,i+1}|)^2
        Andersen, Dobrev & Schaumburg (2012) — jump + noise robust.
        """
        M      = intraday_returns.shape[1]
        factor = (np.pi / (6 - 4 * np.sqrt(3) + np.pi)) * (M / (M - 2))
        abs_r  = np.abs(intraday_returns)

        # stack rolling triplets: shape (N, M-2, 3)
        triplets = np.stack(
            [abs_r[:, :-2], abs_r[:, 1:-1], abs_r[:, 2:]], axis=2
        )
        med_sq  = np.median(triplets, axis=2) ** 2      # (N, M-2)
        medrv   = factor * np.sum(med_sq, axis=1, keepdims=True)
        return medrv                                    # (N, 1)

    @staticmethod
    def _autocovariance(returns: np.ndarray, h: int) -> np.ndarray:
        """gamma_h = sum_i r_i * r_{i+h},  shape (N, 1)."""
        if h == 0:
            return np.sum(returns ** 2, axis=1, keepdims=True)
        return np.sum(returns[:, h:] * returns[:, :-h], axis=1, keepdims=True)

    @staticmethod
    def _parzen_kernel(x: np.ndarray) -> np.ndarray:
        """Parzen kernel  k: [0,1] -> [0,1]."""
        x = np.abs(x)
        return np.where(
            x <= 0.5,
            1 - 6 * x**2 + 6 * x**3,
            np.where(x <= 1.0, 2 * (1 - x)**3, 0.0)
        )

    @staticmethod
    def _tukey_hanning_kernel(x: np.ndarray) -> np.ndarray:
        """Tukey–Hanning kernel  k: [0,1] -> [0,1]."""
        x = np.abs(x)
        return np.where(x <= 1.0, 0.5 * (1 + np.cos(np.pi * x)), 0.0)

    def _realized_kernel(
        self,
        intraday_returns: np.ndarray,
        kernel_func
    ) -> np.ndarray:
        """
        RK_t = gamma_0 + sum_{h=1}^{H} k(h / (H+1)) * (gamma_h + gamma_{-h})
             = gamma_0 + 2 * sum_{h=1}^{H} k(h/(H+1)) * gamma_h
        Barndorff-Nielsen et al. (2008).
        """
        M = intraday_returns.shape[1]
        H = self.bandwidth if self.bandwidth is not None else int(np.ceil(M ** (2 / 3)))

        rk = self._autocovariance(intraday_returns, 0)    # gamma_0 = RV
        for h in range(1, H + 1):
            k_h   = kernel_func(np.array(h / (H + 1)))
            gamma = self._autocovariance(intraday_returns, h)
            rk   += 2 * k_h * gamma
        return rk                                          # (N, 1)

    def rk_parzen(self, intraday_returns: np.ndarray) -> np.ndarray:
        return self._realized_kernel(intraday_returns, self._parzen_kernel)

    def rk_tukey_hanning(self, intraday_returns: np.ndarray) -> np.ndarray:
        return self._realized_kernel(intraday_returns, self._tukey_hanning_kernel)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _mse(pred: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean((pred - target) ** 2))

    @staticmethod
    def _mae(pred: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean(np.abs(pred - target)))
    
    @staticmethod
    def _qloss(log_sigma2, returns_target, alpha_level: float) -> float:
        log_sigma2     = np.array(log_sigma2)
        returns_target = np.array(returns_target)

        z_alpha   = norm.ppf(alpha_level)
        sigma     = np.exp(0.5 * log_sigma2)
        var_t     = z_alpha * sigma

        indicator = (returns_target < var_t).astype(float)
        loss      = (alpha_level - indicator) * (returns_target - var_t)
        return float(np.sum(loss))

    @staticmethod
    def _jointloss(log_sigma2, returns_target, alpha_level: float) -> float:
        log_sigma2     = np.array(log_sigma2, dtype=np.float64)
        returns_target = np.array(returns_target, dtype=np.float64)

        z_alpha = norm.ppf(alpha_level)
        sigma   = np.clip(np.exp(0.5 * log_sigma2), 1e-8, None)   # prevent /0

        Q_t  = z_alpha * sigma
        ES_t = -sigma * norm.pdf(z_alpha) / alpha_level

        ratio     = (alpha_level - 1) / ES_t
        indicator = (returns_target < Q_t).astype(float)
        term1     = -np.log(ratio)
        term2     = -(returns_target - Q_t) * (alpha_level - indicator) / (alpha_level * ES_t)

        return float(np.mean(term1 + term2))

    @staticmethod
    def _nll(log_sigma2: np.ndarray, returns_target: np.ndarray) -> float:
        """
        Gaussian negative log-likelihood (matches training objective).

        NLL = (1/T) * sum_t [ log(sigma_t^2) + r_t^2 / sigma_t^2 ]

        Parameters
        ----------
        log_sigma2     : (N, 1)  model output
        returns_target : (N, 1)  realized returns r_t
        """
        sigma2 = np.exp(log_sigma2) + 1e-6

        loss = np.mean(log_sigma2 + returns_target**2 / sigma2)
        return float(loss)


    # ------------------------------------------------------------------
    # Main evaluation entry-point
    # ------------------------------------------------------------------

    def evaluate(self, log_sigma2,
                returns_target,
                alpha_levels: np.ndarray, 
                intraday_returns: np.ndarray) -> pd.DataFrame:
        """
        Parameters
        ----------
        log_sigma2: np.ndarray  (N, 1)
            Log-variance estimates.
        returns_target: np.ndarray (N, 1)
            real value of r(t+1)
        alpha_levels:
            for VaR and CVar calculation 
        intraday_returns: np.ndarray  (N, M)
            1-min returns for each 15-min observation window.
            M = 15 for 15-min bars built from 1-min data.

        Returns
        -------
        pd.DataFrame  — rows = proxies, columns = [MSE, MAE]
        """
        # ---- model prediction ----------------------------------------
        sigma2_hat = np.exp(log_sigma2)

        # ---- all proxies (variance scale) ----------------------------
        proxies = {
            "RV"              : self.realized_variance(intraday_returns),
            "BV"              : self.bipower_variation(intraday_returns),
            "MedRV"           : self.median_realized_volatility(intraday_returns),
            "RK_Parzen"       : self.rk_parzen(intraday_returns),
            "RK_TukeyHanning" : self.rk_tukey_hanning(intraday_returns),
        }

        proxies = {k: np.array(v, dtype=np.float64) for k, v in proxies.items()}

        # ---- optionally convert to volatility scale ------------------
        if self.compare_variance:
            pred = sigma2_hat
            targets = proxies
        else:
            pred    = np.sqrt(np.clip(sigma2_hat, 0, None))
            targets = {k: np.sqrt(np.clip(v, 0, None)) for k, v in proxies.items()}

        # ---- collect metrics -----------------------------------------
        records = []
        for name, tgt in targets.items():
            records.append({
                "Proxy" : name,
                "MSE"   : self._mse(pred, tgt),
                "MAE"   : self._mae(pred, tgt),
            })

        q_loss_arr = []
        jointloss_arr = []
        
        for alpha in alpha_levels:
            q_loss_arr.append(self._qloss(log_sigma2, returns_target, alpha))       
            jointloss_arr.append(self._jointloss(log_sigma2, returns_target, alpha))
                
        results_tail = pd.DataFrame(
            {"QLOSS": q_loss_arr, "JOINTLOSS": jointloss_arr},
            index = alpha_levels,                      
        )
        results_tail.index.name = "alpha"
        
        NLL = self._nll(log_sigma2, returns_target)

        results_error = pd.DataFrame(records).set_index("Proxy")
        return results_error, results_tail, NLL







        