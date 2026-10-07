import numpy as np
import sympy as sp
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from poisson import Poisson

x, y = sp.symbols("x,y")

# Below we create a solver that reuses some of the implementation from
# the 1D solver in poisson.py.


class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y |in [0, L] x [0, L]

    with Dirichlet boundary conditions.
    """

    def __init__(self, L: float):
        self.L = L
        self.p = Poisson(L)  # we can reuse some of the code from the 1D case

    def create_mesh(self, N: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a 2D Cartesian mesh

        Parameters
        ----------
        N : int
            The number of uniform intervals in both x and y directions

        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh
        """
        xi = self.p.create_mesh(N)
        xij, yij = np.meshgrid(xi, xi, indexing="ij", sparse=True)
        return xij, yij

    def laplace(self, N: int) -> sparse.lil_matrix:
        """Return a vectorized Laplace operator

        Parameters
        ----------
        N : int
            The number of uniform intervals in both x and y directions

        Returns
        -------
        A : scipy sparse LIL matrix
            The vectorized Laplace operator
        """
        dx = self.L / N
        dy = self.L / N

        D2x = self.p.D2(N, dx)
        D2y = self.p.D2(N, dy)

        d2x2 = sparse.kron(D2x, sparse.eye(N + 1))
        d2y2 = sparse.kron(sparse.eye(N + 1), D2y)

        return d2x2 + d2y2

    def assemble(
        self, N: int, f: sp.Expr, ue: sp.Expr
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Return assembled coefficient matrix A and right hand side vector b

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in both x and y directions
        f : Sympy expression
            The right hand side as a Sympy expression in x and y
        ue : Sympy expression
            The exact solution as a Sympy expression in x and y

        Returns
        -------
        A : scipy sparse CSR matrix
            Coefficient matrix
        b : 1D array
            Right hand side vector

        Note
        ----
        Compute the Kronecker product of the 1D Laplace operator with itself
        to create the 2D Laplace operator. Then, assemble the right-hand side
        vector b by evaluating the function f at the mesh points and applying
        Dirichlet boundary conditions using the exact solution ue.
        """

        # creating mesh
        xij, yij = self.create_mesh(N)

        # Making U
        Index = self.get_boundary_indices(N)
        meshfunc = self.meshfunction(f, xij, yij)
        meshue = self.meshfunction(ue, xij, yij)

        # vectorizing
        vecue = meshue.ravel()
        vecmesh = meshfunc.ravel()

        b = vecmesh.copy()
        b[Index] = vecue[Index]

        # making A
        A = self.laplace(N)
        A = A.tolil()
        for i in Index:
            A[i]=0
            A[i,i]=1
        A = A.tocsr()
        return A, b

    def meshfunction(
        self, u: sp.Expr, xij: np.ndarray, yij: np.ndarray
    ) -> np.ndarray:
        """Return Sympy function as mesh function

        Parameters
        ----------
        u : Sympy function

        Returns
        -------
        array - The input function as a mesh function
        """
        U = sp.lambdify((x, y), u, "numpy")
        return U(xij, yij)

    def get_boundary_indices(self, N: int) -> np.ndarray:
        """Return indices of vectorized matrix that belongs to the boundary."""
        n = N + 1
        
        top = np.arange(n)
        bot = np.arange(n * (n - 1), n * n)
        lef = np.arange(0, n * n, n)
        rig = np.arange(n - 1, n * n, n)

        index = np.concatenate([top,lef,rig,bot])
        return np.unique(index)

    def l2_error(self, u: np.ndarray, ue: sp.Expr) -> float:
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy expression
            The exact solution

        Returns
        -------
        float - The l2-error
        """

        # setting up mesh functions for exact solution
        N = len(u) - 1

        xij, yij = self.create_mesh(N)
        meshue = self.meshfunction(ue, xij, yij)

        # defining Delta x & Delta y
        dx = self.L / N
        dy = self.L / N
        errsum = 0

        errsum = np.sum((u - meshue) ** 2)

        errnorm = (dx * dy * errsum) ** 0.5

        return errnorm

    def __call__(self, N: int, ue: sp.Expr) -> np.ndarray:
        """Solve Poisson's equation with a given manufactured solution

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in both x and y directions
        ue : Sympy expression
            The exact solution

        Returns
        -------
        The solution as a Numpy array
        """

        A, b = self.assemble(
            N,
            sp.diff(ue, x, 2) + sp.diff(ue, y, 2),
            ue
        )

        return sparse_linalg.spsolve(
            A,
            b.ravel()
        ).reshape((N + 1, N + 1))

    def convergence_rates(self, ue: sp.Expr, m: int = 6):
        E = []
        h = []

        N0 = 8

        for _ in range(m):
            u = self(N0, ue)
            E.append(self.l2_error(u, ue))
            h.append(self.p.L / N0)
            N0 *= 2

        r = [
            np.log(E[i - 1] / E[i])
            / np.log(h[i - 1] / h[i])
            for i in range(1, m, 1)
        ]

        return r, np.array(E), np.array(h)

    def eval(self, U: np.ndarray, xval: float, yval: float) -> float:
        """Return interpolated u(xval, yval)."""

        N = U.shape[0] - 1
        dx = self.L / N

        # Find the nearest grid point
        ix = int(round(xval / dx))
        iy = int(round(yval / dx))

        # Use 4 points in each direction
        ix = max(1, min(ix, N - 2))
        iy = max(1, min(iy, N - 2))

        xij, yij = self.create_mesh(N)

        xs = xij[ix - 1:ix + 3, 0]
        ys = yij[0, iy - 1:iy + 3]

        basisx = Lagrangebasis(xs, x=x)
        basisy = Lagrangebasis(ys, x=y)

        L2 = Lagrangefunction2D(
            U[ix - 1:ix + 3, iy - 1:iy + 3],
            basisx,
            basisy
        )

        return float(L2.subs({x: xval, y: yval}).evalf())

def Lagrangebasis(xj, x=x):
    """Construct Lagrange basis for points in xj

    Parameters
    ----------
    xj : array
        Interpolation points (nodes)
    x : Sympy Symbol

    Returns
    -------
    Lagrange basis as a list of Sympy functions
    """
    from sympy import Mul

    n = len(xj)
    ell = []

    numert = Mul(*[x - xj[i] for i in range(n)])

    for i in range(n):
        numer = numert / (x - xj[i])
        denom = Mul(
            *[
                (xj[i] - xj[j])
                for j in range(n)
                if i != j
            ]
        )
        ell.append(numer / denom)

    return ell


def Lagrangefunction2D(u, basisx, basisy):
    N, M = u.shape
    f = 0

    for i in range(N):
        for j in range(M):
            f += basisx[i] * basisy[j] * u[i, j]

    return f


def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(
        sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y)
    )

    sol = Poisson2D(1)

    r, _, _ = sol.convergence_rates(ue)

    assert abs(r[-1] - 2) < 1e-2


def test_interpolation():
    ue = sp.exp(
        sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y)
    )

    sol = Poisson2D(1)

    N = 100
    U = sol(N, ue)
    h = sol.p.L / N

    assert abs(
        sol.eval(U, 0.52, 0.63)
        - ue.subs({x: 0.52, y: 0.63}).n()
    ) < 1e-3

    assert abs(
        sol.eval(U, h / 2, 1 - h / 2)
        - ue.subs({x: h, y: 1 - h / 2}).n()
    ) < 1e-3


if __name__ == "__main__":
    test_convergence_poisson2d()
    test_interpolation()
    print("All tests passed!")
