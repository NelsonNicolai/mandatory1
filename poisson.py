from __future__ import annotations

import numpy as np
import sympy as sp
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

x = sp.Symbol("x")


class Poisson:
    """Solve Poisson's equation in 1D::

        u''(x) = f(x), x in [0, L], u(0) = a, u(L) = b

    where a and b are numbers.
    """

    def __init__(self, L: float = 1):
        self.L = L

    def D2(self, N: int, dx: float) -> sparse.lil_matrix:
        """Return second order differentiation matrix"""
        # scipy's untyped stubs infer offsets as int from its default value,
        # so a sequence of offsets is flagged even though it's exactly what
        # the docs ask for.
        D = sparse.diags([1., -2., 1.], [-1, 0, 1], (N + 1, N + 1), format="lil")
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        D /= dx**2
        return D

    def assemble(
        self, N: int, f: sp.Expr, bc: tuple[float, float] = (0, 0)
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Assemble coefficient matrix and right hand side vector

        Parameters
        ----------
        N : int
            The number of uniform intervals
        f : Sympy expression
            The right hand side as a Sympy expression in x
        bc : 2-tuple of numbers
            The boundary conditions at x=0 and x=L

        Returns
        -------
        A : scipy sparse CSR matrix
            Coefficient matrix
        b : 1D array
            Right hand side vector
        """
        mesh = self.create_mesh(N)
        D = self.D2(N, self.L / N)
        D[0, :4] = 1, 0, 0, 0
        D[-1, -4:] = 0, 0, 0, 1
        b = np.zeros(N + 1)
        b[1:-1] = sp.lambdify(x, f)(mesh[1:-1])
        b[0] = bc[0]
        b[-1] = bc[1]
        return D.tocsr(), b

    def create_mesh(self, N: int) -> np.ndarray:
        """Uniform discretization of the line [0, L]

        Parameters
        ----------
        N : int
            The number of uniform intervals

        Returns
        -------
        x : array
            The mesh
        """
        return np.linspace(0, self.L, N + 1)

    def __call__(
        self, N: int, f: sp.Expr, bc: tuple[float, float] = (0, 0)
    ) -> np.ndarray:
        """Solve Poisson's equation

        Parameters
        ----------
        N : int
            The number of uniform intervals
        f : Sympy expression
            The right hand side as a Sympy expression in x
        bc : 2-tuple of numbers
            The boundary conditions at x=0 and x=L

        Returns
        -------
        The solution as a Numpy array

        """
        A, b = self.assemble(N, bc=bc, f=f)
        return sparse_linalg.spsolve(A, b)

    def l2_error(self, u: np.ndarray, ue: sp.Expr) -> float:
        """Return l2-error

        Parameters
        ----------
        u : array_like
            Numerical solution
        ue : Sympy expression
            The analytical solution as a Sympy expression in x
        Returns
        -------
        The l2-error as a number

        """
        N = len(u) - 1
        mesh = self.create_mesh(N)
        dx = self.L / N
        uj = sp.lambdify(x, ue)(mesh)
        return np.sqrt(dx * np.sum((uj - u) ** 2))

    def convergence_rates(
        self, ue: sp.Expr, m: int = 6
    ) -> tuple[list[float], np.ndarray, np.ndarray]:
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            u = self(
                N0,
                f=sp.diff(ue, x, 2),
                bc=(float(ue.subs(x, 0)), float(ue.subs(x, self.L))),
            )
            E.append(self.l2_error(u, ue))
            h.append(self.L / N0)
            N0 *= 2
        r = [np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i]) for i in range(1, m, 1)]
        return r, np.array(E), np.array(h)


def test_poisson():
    sol = Poisson(1)
    ue = sp.exp(4 * sp.cos(x))
    r, _, _ = sol.convergence_rates(ue)
    assert abs(r[-1] - 2) < 1e-2, r


if __name__ == "__main__":
    L = 2
    N = 1000
    sol = Poisson(L=L)
    ue = sp.exp(4 * sp.cos(x))
    # ue = x**2
    bc = (float(ue.subs(x, 0)), float(ue.subs(x, L)))
    u = sol(N, ue.diff(x, 2), bc=bc)
    print("Manufactured solution: ", ue)
    print(f"Boundary conditions: u(0)={bc[0]:2.4f}, u(L)={bc[1]:2.2f}")
    print(f"Discretization: N = {N}")
    print(f"L2-error {sol.l2_error(u, ue)}")
