"""Is there a single window whose averaged or smoothed periodogram equals Thomson's K-taper estimator?

Least-squares search (Frobenius norm on the quadratic form Q, N = 256, NW = 4, K = 7) over
  (b) Welch segment windows u of length L placed with a given step, with or without overhang, and
  (c) taper-and-kernel pairs, Q ~ diag(w) T diag(w) with T Toeplitz (alternating least squares).
Both searches return essentially the sliding sinc / rectangular taper + box, i.e. the leaky all-taper estimate:
no single window can subtract the leaky remainder of the K-taper estimator (paper, Sec. 5.4).
Run: python demos/fit_single_window.py  (about a minute).
"""
import sys, pathlib, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import minimize
import specsmooth as ss


def toeplitz_from_lags(h, N):
    tau = np.arange(N)[:, None] - np.arange(N)[None, :]
    return h[np.abs(tau)]


def fit_taper_kernel(Q, iters=200):
    """Alternating least squares for Q ~ diag(w) T diag(w). Returns unit-norm taper, lag window (h[0] = 1), rel. residual."""
    N = Q.shape[0]
    w = np.sqrt(np.maximum(np.diag(Q), 1e-12))
    for _ in range(iters):
        P = np.outer(w, w)
        num = np.array([(np.diagonal(Q, k) * np.diagonal(P, k)).sum() for k in range(N)])
        den = np.array([(np.diagonal(P, k) ** 2).sum() for k in range(N)])
        h = num / np.maximum(den, 1e-300)
        T = toeplitz_from_lags(h, N)
        for _ in range(3):
            w = (Q * T) @ w / np.maximum((T ** 2) @ (w ** 2), 1e-300)
    res = np.linalg.norm(Q - np.outer(w, w) * T) / np.linalg.norm(Q)
    return w / np.linalg.norm(w), h / h[0], res


def welch_Q(u, N, step, overhang):
    L = len(u); Q = np.zeros((N, N))
    offs = range(-L + 1, N, step) if overhang else range(0, N - L + 1, step)
    for s in offs:
        v = np.zeros(N); lo, hi = max(0, s), min(N, s + L); v[lo:hi] = u[lo - s:hi - s]; Q += np.outer(v, v)
    return Q, list(offs)


def fit_welch(Q, L, step, overhang, u0, iters=300):
    N = Q.shape[0]
    def fg(u):
        Qu, offs = welch_Q(u, N, step, overhang)
        c = np.trace(Q) / np.trace(Qu); R = Q - c * Qu
        g = np.zeros(L)
        for s in offs:
            lo, hi = max(0, s), min(N, s + L)
            g[lo - s:hi - s] += -4 * c * (R[lo:hi, lo:hi] @ u[lo - s:hi - s])
        return np.sum(R ** 2), g
    r = minimize(fg, u0, jac=True, method="L-BFGS-B", options={"maxiter": iters})
    Qu, _ = welch_Q(r.x, N, step, overhang); Qu *= np.trace(Q) / np.trace(Qu)
    return r.x, np.linalg.norm(Q - Qu) / np.linalg.norm(Q)


def dist(Q, Qc):
    Qc = Qc / np.trace(Qc) * np.trace(Q)
    return np.linalg.norm(Q - Qc) / np.linalg.norm(Q)


def main(N=256, NW=4):
    W = NW / N; K = 2 * NW - 1
    V, lam = ss.dpss_all(N, W)
    QK = V[:, :K] @ V[:, :K].T / K
    A = ss.sinc_toeplitz(N, W)
    hann = ss.unit_taper("hann", N); rect = np.ones(N) / np.sqrt(N); hb = ss.box_lag_window(N, W)
    print(f"Thomson K={K} (N={N}, NW={NW}); relative Frobenius distance of single-window estimators to Q_K:")
    print(f"  rect taper + box (= sliding sinc = all-taper lambda-weighted): {dist(QK, A):.3f}")
    print(f"  Hann taper + box:                                             {dist(QK, np.outer(hann, hann) * toeplitz_from_lags(hb, N)):.3f}")
    t0 = time.time(); w, h, res = fit_taper_kernel(QK)
    print(f"  best taper+kernel pair (ALS): {res:.3f}; taper max/mean = {w.max() / w.mean():.2f} (rect = 1.00); "
          f"first lags of kernel {np.round(h[:5], 3)} vs box {np.round(hb[:5], 3)}  [{time.time() - t0:.0f}s]")
    for L in [N // K, N // 4, N // 2, N]:
        for step, ov in [(1, True), (max(1, L // 2), False), (L, False)]:
            u0 = ss.unit_taper("hann", L)
            base = dist(QK, welch_Q(u0, N, step, ov)[0])
            t0 = time.time(); u, res = fit_welch(QK, L, step, ov, u0)
            print(f"  Welch window fit, L={L:3d} step={step:3d} overhang={str(ov):5s}: {res:.3f}  (Hann window: {base:.3f})  [{time.time() - t0:.0f}s]")
    for L in [N, 4 * N]:
        Qs, _ = welch_Q(ss.sinc_window(L, W), N, 1, True)
        print(f"  sliding sinc, L={L}: distance to A {dist(A, Qs):.3f}, to Q_K {dist(QK, Qs):.3f}")


if __name__ == "__main__":
    main()
