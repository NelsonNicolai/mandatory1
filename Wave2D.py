import numpy as np
import sympy as sp
from scipy import sparse


x, y, t = sp.symbols("x y t")


class Wave2D:
    """Class for solving the 2D wave equation."""

    def create_mesh(
        self, N: int, sparse: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return 2D mesh created using np.meshgrid."""
        xi = np.linspace(0, 1, N + 1)
        xij, yij = np.meshgrid(
            xi, xi, indexing="ij", sparse=sparse
        )
        return xij, yij

    def D2(self, N: int) -> sparse.lil_matrix:
        """Return second-order differentiation matrix."""
        dx = 1 / N
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D / dx**2

    def ue(self, mx: int, my: int, c: float, t0: float) -> np.ndarray:
        """Return the exact standing wave at time t0."""

        w = c * np.pi * np.sqrt(mx**2 + my**2)

        xij, yij = self.create_mesh(self.N)

        return (
            np.sin(mx * np.pi * xij)
            * np.sin(my * np.pi * yij)
            * np.cos(w * t0)
        )

    def initialize(
        self,
        N: int,
        mx: int,
        my: int,
        c: float,
    ) -> np.ndarray:
        """Initialize the solution at t=0."""

        xij, yij = self.create_mesh(N)

        w = c * np.pi * np.sqrt(mx**2 + my**2)

        U = (
            np.sin(mx * np.pi * xij)
            * np.sin(my * np.pi * yij)
            * np.cos(w * 0.0)
        )

        return U

    def l2_error(
        self,
        u: np.ndarray,
        t0: float,
        mx: int,
        my: int,
        c: float,
    ) -> float:
        """Return the L2-error norm."""

        N = len(u) - 1

        xij, yij = self.create_mesh(N)

        w = c * np.pi * np.sqrt(mx**2 + my**2)

        exact = (
            np.sin(mx * np.pi * xij)
            * np.sin(my * np.pi * yij)
            * np.cos(w * t0)
        )

        dx = 1 / N
        dy = 1 / N

        errsum = np.sum((u - exact) ** 2)

        return np.sqrt(dx * dy * errsum)

    def apply_bcs(self, u: np.ndarray):
        """Apply Dirichlet boundary conditions."""

        u[0, :] = 0
        u[-1, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0

    def __call__(
        self,
        N: int,
        Nt: int,
        cfl: float = 0.5,
        c: float = 1.0,
        mx: int = 3,
        my: int = 3,
        store_data: int = -1,
    ):
        """Solve the wave equation."""

        # Time step
        dt = cfl / (c * N)

        # Spatial differentiation matrix
        D = self.D2(N)

        # Initial condition
        Un = self.initialize(N, mx, my, c)

        # Initial velocity is zero.
        # For u_tt = c^2 Laplace(u), use a Taylor expansion
        # to obtain the first time step.
        laplace_Un = D @ Un + Un @ D.T

        Unp1 = Un + 0.5 * (c * dt) ** 2 * laplace_Un

        self.apply_bcs(Unp1)

        Unm1 = Un.copy()
        Un = Unp1.copy()

        if store_data > 0:
            keys = [0]
            values = [Un.copy()]

            for n in range(1, Nt):
                laplace_Un = D @ Un + Un @ D.T

                Unp1 = (
                    2 * Un
                    - Unm1
                    + (c * dt) ** 2 * laplace_Un
                )

                self.apply_bcs(Unp1)

                keys.append(n)
                values.append(Unp1.copy())

                Unm1 = Un.copy()
                Un = Unp1.copy()

            return dict(zip(keys, values))

        elif store_data == -1:
            # If the first step has already been performed,
            # take the remaining Nt-1 steps.
            for _ in range(1, Nt):
                laplace_Un = D @ Un + Un @ D.T

                Unp1 = (
                    2 * Un
                    - Unm1
                    + (c * dt) ** 2 * laplace_Un
                )

                self.apply_bcs(Unp1)

                Unm1 = Un.copy()
                Un = Unp1.copy()

            h = 1 / N
            T = Nt * dt

            l2 = self.l2_error(
                Un,
                T,
                mx,
                my,
                c,
            )

            return h, l2

        else:
            raise ValueError("store_data must be -1 or greater than 0")

    def convergence_rates(
        self,
        m: int = 4,
        cfl: float = 0.1,
        Nt: int = 10,
        mx: int = 3,
        my: int = 3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute convergence rates for a range of discretizations."""

        E = []
        h = []

        N0 = 8

        for _ in range(m):
            dx, err = self(
                N0,
                Nt,
                cfl=cfl,
                mx=mx,
                my=my,
                store_data=-1,
            )

            E.append(err)
            h.append(dx)

            N0 *= 2
            Nt *= 2

        r = [
            np.log(E[i - 1] / E[i])
            / np.log(h[i - 1] / h[i])
            for i in range(1, m)
        ]

        return np.array(r), np.array(E), np.array(h)


class Wave2D_Neumann(Wave2D):
    """2D wave equation with homogeneous Neumann boundary conditions."""

    def D2(self, N: int) -> sparse.lil_matrix:
        dx = 1 / N
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2

        D[0, :] = 0
        D[0, 0] = -2
        D[0, 1] = 2

        D[-1, :] = 0
        D[-1, -2] = 2
        D[-1, -1] = -2

        return D / dx**2

    def initialize(
        self,
        N: int,
        mx: int,
        my: int,
        c: float,
    ) -> np.ndarray:
        """Initialize cosine standing wave."""

        xij, yij = self.create_mesh(N)

        return (
            np.cos(mx * np.pi * xij)
            * np.cos(my * np.pi * yij)
        )

    def l2_error(
        self,
        u: np.ndarray,
        t0: float,
        mx: int,
        my: int,
        c: float,
    ) -> float:
        """Return the L2-error for the Neumann solution."""

        N = len(u) - 1

        xij, yij = self.create_mesh(N)

        w = c * np.pi * np.sqrt(mx**2 + my**2)

        exact = (
            np.cos(mx * np.pi * xij)
            * np.cos(my * np.pi * yij)
            * np.cos(w * t0)
        )

        dx = 1 / N
        dy = 1 / N

        errsum = np.sum((u - exact) ** 2)

        return np.sqrt(dx * dy * errsum)

    def apply_bcs(self, u: np.ndarray):
        """Apply homogeneous Neumann boundary conditions."""
        pass


def test_convergence_wave2d():
    sol = Wave2D()

    r, _, _ = sol.convergence_rates(
        m=5,
        mx=2,
        my=3,
    )

    assert abs(r[-1] - 2) < 0.05, r


def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()

    r, _, _ = solN.convergence_rates(
        mx=3,
        my=3,
    )

    assert abs(r[-1] - 2) < 0.05, r


def test_exact_wave2d():
    #constants, defined
    N  = 10
    Nt = 10
    cfl= 1 / (2**0.5)
    c  = 1
    mx = 2
    my = 2
    store_data = -1

    #constants, derived
    dt = cfl / (c * N)
    T  = Nt * dt

    #initialising class
    dir = Wave2D()
    neu = Wave2D_Neumann()

    #l2
    hdir, l2dir = dir(N,Nt,cfl,c,mx,my,store_data)
    hneu, l2neu = neu(N,Nt,cfl,c,mx,my,store_data)
    #metrics
    assert l2dir < 1e-12
    assert l2neu < 1e-12

if __name__ == "__main__":
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    test_exact_wave2d()
    print("All tests passed!")