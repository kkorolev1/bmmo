import numpy as np
from scipy.special import softmax
from scipy.signal import fftconvolve

EPS = 1e-10

def calculate_log_probability(X, F, B, s):
    """
    Calculates log p(X_k|d_k,F,B,s) for all images X_k in X and
    all possible displacements d_k.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.

    Returns
    -------
    ll : array, shape(H-h+1, W-w+1, K)
        ll[dh,dw,k] - log-likelihood of observing image X_k given
        that the villain's face F is located at displacement (dh, dw)
    """
    H, W, K = X.shape
    h, w = F.shape

    ones_kernel = np.ones((h, w, 1))

    x_square_fa = fftconvolve(X ** 2, ones_kernel, mode="valid")
    xf_fa = fftconvolve(-2*X, np.expand_dims(F[::-1,::-1], axis=-1), mode="valid")
    face_square_residual = x_square_fa + xf_fa + (F ** 2).sum()
    
    bg_square_residual = (X - B[:, :, None]) ** 2
    bg_square_residual_fa = fftconvolve(bg_square_residual, ones_kernel, mode="valid")

    square_residual = face_square_residual + bg_square_residual.sum(axis=(0, 1)) - bg_square_residual_fa

    return -0.5 * square_residual / (s ** 2) - H * W * np.log(np.sqrt(2 * np.pi) * s + EPS)


def calculate_lower_bound(X, F, B, s, A, q, use_MAP=False):
    """
    Calculates the lower bound L(q,F,B,s,A) for the marginal log likelihood.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    L : float
        The lower bound L(q,F,B,s,A) for the marginal log likelihood.
    """
    log_prob = calculate_log_probability(X, F, B, s)
    if not use_MAP:
        return (q * log_prob).sum() + (q * np.log(A + EPS)[:, :, None]).sum() - (q[q > 0] * np.log(q[q > 0] + EPS)).sum()
    return (log_prob[q[0], q[1], np.arange(q.shape[1])]).sum() + (np.log(A + EPS)[q[0], q[1]]).sum()



def run_e_step(X, F, B, s, A, use_MAP=False):
    """
    Given the current esitmate of the parameters, for each image Xk
    esitmates the probability p(d_k|X_k,F,B,s,A).

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    F  : array_like, shape(h, w)
        Estimate of villain's face.
    B : array shape(H, W)
        Estimate of background.
    s : scalar, shape(1, 1)
        Eestimate of standard deviation of Gaussian noise.
    A : array, shape(H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    use_MAP : bool, optional,
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    """
    log_numero = calculate_log_probability(X, F, B, s) + np.log(A + EPS)[:, :, None]
    if not use_MAP:
        return softmax(log_numero, axis=(0, 1))
    max_idx = log_numero.reshape(-1, log_numero.shape[2]).argmax(axis=0)
    return np.stack(np.unravel_index(max_idx, (log_numero.shape[0], log_numero.shape[1])), axis=0)


def run_m_step(X, q, h, w, use_MAP=False):
    """
    Estimates F,B,s,A given esitmate of posteriors defined by q.

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    q  :
        if use_MAP = False: array, shape (H-h+1, W-w+1, K)
           q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
           of villain's face given image Xk
        if use_MAP = True: array, shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    h : int
        Face mask height.
    w : int
        Face mask width.
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    """
    H, W, K = X.shape
    ones_kernel = np.ones((h, w, 1))
    if use_MAP:
        indicator_MAP = np.zeros((H - h + 1, W - w + 1, K))
        indicator_MAP[q[0], q[1], np.arange(K)] = 1
        q = indicator_MAP
    A = q.sum(axis=2) / K
    F = fftconvolve(X, q[::-1, ::-1], axes=(0, 1), mode="valid").sum(axis=2) / K
    
    convolved_q = fftconvolve(q, ones_kernel, mode="full")
    B_numero = (X * (1 - convolved_q)).sum(axis=2)
    B_denumero = (1 - convolved_q).sum(axis=2)
    B_mask = B_denumero > 0
    B = np.zeros((H, W))
    B[B_mask] = B_numero[B_mask] / B_denumero[B_mask]

    x_square_fa = fftconvolve(X ** 2, ones_kernel, mode="valid")
    xf_fa = fftconvolve(-2*X, np.expand_dims(F[::-1,::-1], axis=-1), mode="valid")
    face_square_residual = x_square_fa + xf_fa + (F ** 2).sum()
    
    bg_square_residual = (X - B[:, :, None]) ** 2
    bg_square_residual_fa = fftconvolve(bg_square_residual, ones_kernel, mode="valid")

    square_residual = face_square_residual + bg_square_residual.sum(axis=(0, 1)) - bg_square_residual_fa
    s = np.sqrt((q * square_residual).sum() / (K * H * W) + EPS)
    
    return F, B, s, A

def run_EM(X, h, w, F=None, B=None, s=None, A=None, tolerance=0.001,
           max_iter=50, use_MAP=False):
    """
    Runs EM loop until the likelihood of observing X given current
    estimate of parameters is idempotent as defined by a fixed
    tolerance.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    F : array, shape (h, w), optional
        Initial estimate of villain's face.
    B : array, shape (H, W), optional
        Initial estimate of background.
    s : float, optional
        Initial estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1), optional
        Initial estimate of prior on displacement of face in any image.
    tolerance : float, optional
        Parameter for stopping criterion.
    max_iter  : int, optional
        Maximum number of iterations.
    use_MAP : bool, optional
        If true then after E-step we take only MAP estimates of displacement
        (dh,dw) of villain's face given image Xk.

    Returns
    -------
    F, B, s, A : trained parameters.
    LL : array, shape(number_of_iters,)
        L(q,F,B,s,A) after each EM iteration (1 iteration = 1 e-step + 1 m-step); 
        number_of_iters is actual number of iterations that was done.
    """
    H, W, K = X.shape
    if F is None:
        F = np.random.rand(h, w)
    if B is None:
        B = np.random.rand(H, W)
    if s is None:
        s = 1
    if A is None:
        A = np.ones((H - h + 1, W - w + 1), dtype=np.float64) / (H - h + 1) / (W - w + 1)
    history = []
    for i in range(max_iter):
        q = run_e_step(X, F, B, s, A, use_MAP)
        F, B, s, A = run_m_step(X, q, h, w, use_MAP)
        history.append(calculate_lower_bound(X, F, B, s, A, q, use_MAP))
        if len(history) > 1 and history[-1] - history[-2] < tolerance:
            break
    return F, B, s, A, np.array(history)

def run_EM_with_restarts(X, h, w, tolerance=0.001, max_iter=50, use_MAP=False,
                         n_restarts=10):
    """
    Restarts EM several times from different random initializations
    and stores the best estimate of the parameters as measured by
    the L(q,F,B,s,A).

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    tolerance, max_iter, use_MAP : optional parameters for EM.
    n_restarts : int
        Number of EM runs.

    Returns
    -------
    F : array,  shape (h, w)
        The best estimate of villain's face.
    B : array, shape (H, W)
        The best estimate of background.
    s : float
        The best estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        The best estimate of prior on displacement of face in any image.
    L : float
        The best L(q,F,B,s,A).
    """
    best_params = [None] * 4
    LL_best = float("-inf")
    for i in range(n_restarts):
        F, B, s, A, LL = run_EM(X, h, w, None, None, None, None, tolerance, max_iter, use_MAP)
        if LL > LL_best:
            LL_best = LL
            best_params = [F, B, s, A]
    return *best_params, LL_best
