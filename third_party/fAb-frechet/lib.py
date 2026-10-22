from collections.abc import Callable

import numpy as np


def arnoldi(
    mat: np.ndarray, vec: np.ndarray, num: int, reorth_its: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    _dtype = float if not np.any(np.iscomplex(mat)) else complex
    V = np.zeros((vec.shape[0], num + 1), dtype=_dtype)
    H = np.zeros((num + 1, num), dtype=_dtype)
    V[:, 0] = vec / np.linalg.norm(vec)
    for it in range(num):
        w = mat @ V[:, it]
        for prev in range(it + 1):
            H[prev, it] = V[:, prev].conj().T @ w
            w -= H[prev, it] * V[:, prev]
        # reorthogonalize against all existing basis vectors
        for _ in range(reorth_its):
            for prev in range(it + 1):
                fac = V[:, prev].conj().T @ w
                H[prev, it] += fac
                w -= fac * V[:, prev]
        nw = np.linalg.norm(w)
        H[it + 1, it] = nw
        if np.abs(H[it + 1, it]) < 1e-16:  # check for lucky breakdown
            print(f"Lucky breakdown at iteration {it + 1}")
            V = V[:, : it + 2]
            H = H[: it + 2, : it + 1]
            break
        V[:, it + 1] = w / nw
    return V, H


def lanczos(
    mat: np.ndarray, vec: np.ndarray, num: int, reorth_its: int = 2
) -> tuple[list, list, list]:
    alpha, beta = [], []
    v = vec.copy() / np.linalg.norm(vec)
    V = [v]
    for index in range(num):
        w = mat @ v
        alpha.append(v @ w)
        # reorthogonalize against all existing basis vectors
        for _ in range(reorth_its):
            for reorthind in range(index + 1):
                w -= (V[reorthind] @ w) * V[reorthind]
        beta.append(np.linalg.norm(w))
        if np.abs(beta[-1]) < 1e-16:  # check for lucky breakdown
            print(f"Lucky breakdown at iteration {index + 1}")
            break
        v = w / beta[-1]
        V.append(v)
    return V, alpha, beta


def unit(size: int, pos: int = 0) -> np.ndarray:
    return np.eye(1, size, k=pos).T[:, 0]


def eval_fAb_krylov(
    func: Callable, A: np.ndarray, b: np.ndarray, E: np.ndarray, krylov_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute f(A) b and Df[E](A) b simultaneously.
    """
    n = A.shape[0]
    bp = np.pad(b, (n, 0))
    AA = np.block([[A, E], [np.zeros_like(A, dtype=float), A]])
    V, H = arnoldi(AA, bp, krylov_steps)
    zs = np.zeros((n, V.shape[1]))
    UU = np.linalg.qr(np.block([[V[:n], zs], [zs, V[n:]]]))[0]
    xx = UU @ func(UU.conj().T @ AA @ UU) @ UU.conj().T @ bp
    return xx[n:], xx[:n], V, H


def eval_fAb_krylov_sparse(
    func: Callable, A: np.ndarray, b: np.ndarray, E: np.ndarray, krylov_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute f(A) b and Df[E](A) b simultaneously.
    """
    n = A.shape[0]

    X, Y, XAX, YAY, XEY = sep_orth(A, E, b, krylov_steps, return_projs=True)
    Y = Y[:, :-1]
    U = np.block([[X, np.zeros_like(Y)], [np.zeros_like(X), Y]])
    AA = np.block(
        [
            [XAX, XEY],
            [np.zeros_like(XEY), YAY],
        ]
    )

    # bp = np.pad(b, (n, 0))
    # AA = sps.block_array([[A, E], [None, A]])
    # V, H = arnoldi(AA, bp, krylov_steps)
    # zs = np.zeros((n, V.shape[1]))
    # UU = np.linalg.qr(np.block([[V[:n], zs], [zs, V[n:]]]))[0]
    # xx = UU @ func(UU.conj().T @ AA @ UU) @ UU.conj().T @ bp

    xx = U @ (func(AA) @ np.pad(Y.conj().T @ b, (krylov_steps, 0)))
    return xx[n:], xx[:n]


def armijo(
    func: Callable,
    f_val: float,
    df_val: np.ndarray,
    param: float,
    step_size: float,
    tau: float = 0.5,
    c: float = 0.5,
) -> float:
    t = c * df_val
    step_size /= tau
    nf = func(param - step_size * df_val)
    of = nf
    while np.isnan(nf) or f_val - nf < step_size * t:
        step_size *= tau
        of = nf
        nf = func(param - step_size * df_val)
        print(f"ARMIJO SET STEP_SIZE TO {step_size: 1.4e}")
        if f_val - of > f_val - nf:
            break
    return step_size


def scalar_gradient_descent(
    func: Callable,
    func_and_grad: Callable,
    initial_guess: float,
    step_size: float = 5e-1,
    stoptol: float = 1e-8,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    The function `func` needs to return both f(theta) and f'(theta).
    """
    phist, fhist = [], []

    param = initial_guess
    f, df = func_and_grad(param)
    fhist.append(f)
    phist.append(param)

    print(f"F: {f:1.4e} @ P: {param:1.4f}")
    while f > stoptol:
        step_size = armijo(func, f, df, param, step_size)
        param -= step_size * df
        f, df = func_and_grad(param)
        print(f"F: {f:1.4e} @ P: {param:1.4f}")
        fhist.append(f)
        phist.append(param)

    return param, np.array(phist), np.array(fhist)


def arnoldi_update(
    func: Callable,
    mat: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    num: int,
    h: float,
    reorth_its: int = 2,
) -> np.ndarray:
    """
    Compute f(A + b c^*) - f(A).
    """
    zeros = np.zeros((num, num))
    e1 = unit(num)
    nb, nc = np.linalg.norm(b), np.linalg.norm(c)
    Vb, Hb = arnoldi(mat, b, num, reorth_its)
    Vc, Hc = arnoldi(mat.conj().T, c, num, reorth_its)
    Hb, Hc = Hb[:-1], Hc[:-1]
    Vb, Vc = Vb[:, :-1], Vc[:, :-1]
    bmat = np.block(
        [
            [Hb, h * nb * nc * (e1[:, np.newaxis] @ e1[np.newaxis, :])],
            [
                zeros,
                Hc.conj().T
                + h * nc * (Vc.conj().T @ b)[:, np.newaxis] @ e1[np.newaxis, :],
            ],
        ]
    )
    fb = Vb @ func(bmat)[:num, num:] @ Vc.conj().T
    return fb


def sep_orth(
    A: np.ndarray,
    E: np.ndarray,
    b: np.ndarray,
    num: int,
    reorth_its: int = 2,
    return_projs: bool = False,
) -> tuple:
    n = A.shape[0]

    X, Y, AX, EY = (
        np.zeros((n, num)),
        np.zeros((n, num + 1)),
        np.zeros((n, num)),
        np.zeros((n, num + 1)),
    )
    R, H = np.zeros((num, num + 1)), np.zeros((num + 1, num))

    Y[:, 0] = b / np.linalg.norm(b)

    for it in range(num):
        wA = AX[:, :it] @ R[:it, it]
        wE = E @ Y[:, it]
        EY[:, it] = wE

        wX = wA + wE
        wY = A @ Y[:, it]

        # orth for wY
        h = Y[:, : it + 1].conj().T @ wY
        H[: it + 1, it] = h
        wY -= Y[:, : it + 1] @ h
        for _ in range(reorth_its):
            wY -= Y[:, : it + 1] @ (Y[:, : it + 1].conj().T @ wY)

        # update Y and R
        beta = np.linalg.norm(wY)
        H[it + 1, it] = beta
        Y[:, it + 1] = wY / beta
        R[:it, it + 1] = -R[:it, : it + 1] @ h / beta
        R[it, it + 1] = 1 / beta

        # orth for wX
        g = X[:, :it].conj().T @ wX
        wX -= X[:, :it] @ g
        for _ in range(reorth_its):
            wX -= X[:, :it] @ (X[:, :it].conj().T @ wX)

        # update X, AX and R
        alpha = np.linalg.norm(wX)
        X[:, it] = wX / alpha
        AX[:, it] = A @ X[:, it]
        preR = (
            np.array([[alpha]])
            if it == 0
            else np.block([[np.eye(it), g[..., np.newaxis]], [np.zeros(it), alpha]])
        )
        R[: it + 1, it + 1] = preR @ R[: it + 1, it + 1]

    if return_projs:
        return X, Y, X.conj().T @ AX, H[:-1], X.conj().T @ EY[:, :-1]
    return X, Y
