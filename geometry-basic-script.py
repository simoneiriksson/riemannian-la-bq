import numpy as np
import sys
from scipy.integrate import solve_bvp
from scipy.integrate import solve_ivp
import scipy.integrate as integrate
import matplotlib.pyplot as plt


# =========================================================== #
# ==================== UTILITY FUNCTIONS ==================== #
# =========================================================== #

def my_meshgrid(x1min, x1max, x2min, x2max, N=10):
    X1, X2 = np.meshgrid(np.linspace(x1min, x1max, N), np.linspace(x2min, x2max, N))
    X = np.concatenate((np.expand_dims(X1.flatten(), axis=1), np.expand_dims(X2.flatten(), axis=1)), axis=1)
    return X


# ============================================================= #
# ==================== GEOMETRIC FUNCTIONS ==================== #
# ============================================================= #

# This function evaluates the differential equation c'' = f(c, c') given a metric and the derivative
def geodesic_system(manifold, c, dc):
    # Input: c, dc (DxN)

    D, N = c.shape
    if (dc.shape[0] != D) | (dc.shape[1] != N):
        print('geodesic_system: second and third input arguments must have same dimensionality\n')
        sys.exit(1)

    # Evaluate the metric and the derivative
    M, dM = manifold.metric_tensor(c, nargout=2)

    # Prepare the output (D x N)
    ddc = np.zeros((D, N))

    # Diagonal Metric Case, M (N x D), dMdc_d (N x D x d=1,...,D) d-th column derivative with respect to c_d
    if manifold.diagonal:
        for n in range(N):
            dMn = np.squeeze(dM[n, :, :])
            ddc[:, n] = -0.5 * (2 * np.matmul(dMn * dc[:, n].reshape(-1, 1), dc[:, n])
                                - np.matmul(dMn.T, (dc[:, n] ** 2))) / M[n, :]

    # Non-Diagonal Metric Case, M ( N x D x D ), dMdc_d (N x D x D x d=1,...,D)
    else:
        M_inv = np.linalg.inv(M)  # N x D x D
        Term1 = dM.reshape(N, D, D * D, order='F')  # N x D x D^2
        Term2 = dM.reshape(N, D * D, D, order='F')  # N x D^2 x D

        for n in range(N):
            ddc[:, n] = -0.5 * M_inv[n, :, :] @ ((2 * Term1[n, :, :] - Term2[n, :, :].T) @ np.kron(dc[:, n], dc[:, n]))
    return ddc


# This function makes transforms a 2nd order ODE system to 1st order system
def second2first_order(manifold, state):
    # Input: state [c; dc] (2D x N)
    # Output: state y=[dc; ddc] (2D x N)
    D = int(state.shape[0] / 2)
    print(f"ODE_FUN here! {state.shape = }")
    # TODO: Something better for this? Is it necessary?
    if state.ndim == 1:
        state = state.reshape(-1, 1)  # (2D,) -> (2D, 1)

    # c is c(t)
    c = state[:D, :]  # D x N

    # cm is c'(t)
    cm = state[D:, :]  # D x N

    # cmm is c''(t)
    if manifold.analytic:
        cmm = manifold.geodesic_system(c, cm)  # D x N, Implements analytically the ODE
    else:
        cmm = geodesic_system(manifold, c, cm)  # D x N, Implements the general ODE
    
    y = np.concatenate((cm, cmm), axis=0)
    return y


# This function implements the exponential map
def expmap(manifold, x, v):
    # Input: v, x (Dx1)
    x = x.reshape(-1, 1)
    v = v.reshape(-1, 1)
    D = x.shape[0]

    ode_fun = lambda t, c_dc: second2first_order(manifold, c_dc).flatten()  # The solver needs this shape (D,)
    if np.linalg.norm(v) > 1e-5:
        curve, failed = solve_expmap(x, v, ode_fun)
    else:
        curve = lambda t: (x.reshape(D, 1).repeat(np.size(t), axis=1),
                           v.reshape(D, 1).repeat(np.size(t), axis=1))  # Return tuple (2D x T)
        failed = True

    return curve, failed


# This function solves the initial value problem for the implementation of the expmap
def solve_expmap(x, v, ode_fun):
    # Is the vector in normal coordinates or not? i.e. Exp(Log_x(y)) == y ?

    init = np.concatenate((x, v), axis=0).flatten()  # 2D x 1 -> (2D, ), the solver needs this shape
    failed = False

    t0 = 0
    t1 = 1
    solution = solve_ivp(ode_fun, [t0, t1], init, dense_output=True, rtol=1e-3, atol=1e-3)  # First solution of the IVP problem ,,
    curve = lambda t: evaluate_solution(solution, t)

    return curve, failed


# Structure the solution
def evaluate_solution(solution, t):
    t = np.array([t]).reshape(-1,)
    c_dc = solution.sol(t)
    D = int(c_dc.shape[0] / 2)

    c = c_dc[:D, :]  # D x T
    dc = c_dc[D:, :]  # D x T
    return c, dc

# These are the boundary condition for computing the geodesic
def boundary_conditions(ya, yb, c0, c1):
    D = c0.shape[0]
    retVal = np.zeros(2 * D)
    retVal[:D] = ya[:D] - c0.flatten()
    retVal[D:] = yb[:D] - c1.flatten()
    return retVal


# If the solver failed provide the linear interpolation as the solution
def evaluate_failed_solution(p0, p1, t):
    # Input: p0, p1 (D x 1), t (T x 0)
    c = (1 - t) * p0 + t * p1  # D x T
    dc = np.repeat(p1 - p0, np.size(t), 1)  # D x T
    return c, dc


# This function computes the infinitesimal small length on a curve
def local_length(manifold, curve, t):
    # Input: curve function of t returns (D X T), t (T x 0)
    c, dc = curve(t)  # [D x T, D x T]
    D = c.shape[0]
    M = manifold.metric_tensor(c, nargout=1)
    if manifold.diagonal:
        dist = np.sqrt(np.sum(M.transpose() * (dc ** 2), axis=0))  # T x 1, c'(t) M(c(t)) c'(t)
    else:  # TODO: Use np.einsum?
        dc = dc.T  # D x N -> N x D
        dc_rep = np.repeat(dc[:, :, np.newaxis], D, axis=2)  # N x D -> N x D x D
        Mdc = np.sum(M * dc_rep, axis=1)  # N x D
        dist = np.sqrt(np.sum(Mdc * dc, axis=1))  # N x 1
    return dist


# This function computes the length of the geodesic curve
def curve_length(manifold, curve, a=0, b=1, tol=1e-5, limit=50):
    # Input: curve a function of t returns (D x ?), [a,b] integration interval, tol error of the integration
    if callable(curve):
        curve_length_eval = integrate.quad(lambda t: local_length(manifold, curve, t), a, b, epsabs=tol, limit=limit)
    else:
        print("TODO: Not implemented yet integration for discrete curve!\n")
        sys.exit(1)

    return curve_length_eval[0]


# Master function that selects the solver
def compute_geodesic(solver, manifold, c0, c1, solution=None):
    if solver.name == 'bvp':
        geodesic_solution = solver_bvp(solver, manifold, c0, c1, solution)
    else:
        print("TODO: Not supported solver!\n")
        sys.exit(1)

    return geodesic_solution


# Construct an object for running a solver
class SolverBVP:

    def __init__(self, NMax=1000, tol=1e-1):
        self.NMax = NMax
        self.tol = tol
        self.name = 'bvp'


# This is the default solver that is a build-in python BVP solver.
def solver_bvp(solver, manifold, c0, c1, init_solution):
    # c0, c1: Dx1
    c0 = c0.reshape(-1, 1)
    c1 = c1.reshape(-1, 1)
    D = c0.shape[0]

    # The functions that we need for the bvp solver
    # ode_fun = lambda t, c_dc: second2first_order_MONGE(manifold, c_dc)  # D x T, implements c'' = f(c, c')
    ode_fun = lambda t, c_dc: second2first_order(manifold, c_dc)  # D x T, implements c'' = f(c, c')
    bc_fun = lambda ya, yb: boundary_conditions(ya, yb, c0, c1)  # 2D x 0, what returns?

    # Initialize the curve with straight line or with another given curve
    T = 10
    t_init = np.linspace(0, 1, T, dtype=np.float32)  # T x 0
    if init_solution is None:
        c_init = np.outer(c0, (1.0 - t_init.reshape(1, T))) + np.outer(c1, t_init.reshape(1, T))  # D x T
        dc_init = (c1 - c0).reshape(D, 1).repeat(T, axis=1)  # D x T
    else:
        if init_solution['solver'] == 'bvp':
            c_init, dc_init = init_solution['curve'](t_init)  # D x T, D x T
        else:
            print('The initial curve solution to the solver does not exist (bvp)!')
            sys.exit(1)
    c_dc_init = np.concatenate((c_init, dc_init), axis=0)  # 2D x T

    # Solve the geodesic problem
    result = solve_bvp(ode_fun, bc_fun, t_init.flatten(), c_dc_init, tol=solver.tol, max_nodes=solver.NMax)


    # Provide the output, if solver failed return the straight line as solution
    if result.success:
        curve = lambda t: evaluate_solution(result, t)
        logmap = result.y[D:, 0]  # D x 1, the initial velocity of the curve
        solution = {'solver': 'bvp', 'curve': curve}
        failed = False
    else:
        print('Geodesic solver (bvp) failed!')
        curve = lambda t: evaluate_failed_solution(c0, c1, t)
        logmap = (c1 - c0)  # D x 1
        solution = None
        failed = True

    # Compute the curve length under the Riemannian measure and compute the logarithmic map
    curve_length_eval = curve_length(manifold, curve)
    # logmap = curve_length_eval * logmap.reshape(-1, 1) / np.linalg.norm(logmap)  # Scaling for normal coordinates (v.T @ M @ v = length^2)?

    geodesic_res = {'curve': curve, 'logmap': logmap, 'length': curve_length_eval, 'solution': solution, 'failed': failed}

    return geodesic_res


# An object from this class corresponds to a Riemannian manifold.
class loss_manifold():
    def __init__(self, analytic=False):
        self.analytic = analytic  # Use the analytic system or not?
        self.diagonal = False

    # Returns the geodesic system if it is analytically tractable
    def geodesic_system(self, c, dc):
        # c, dc: D x N
        c = c.T  # N x D
        dc = dc.T  # N x D
        N, D = c.shape
        ddc = np.zeros((N, D))

        for n in range(N):
            cn = c[n, :].reshape(-1, 1)
            dcn = dc[n, :].reshape(-1, 1)

            Grad_val = grad_fun(cn.T).reshape(-1, 1)
            Hess_val = Hess_fun(cn.T).reshape(D, D)
            
            ddc[n, :] = -(Grad_val * (1 / (1 + Grad_val.T @ Grad_val)) * (dcn.T @ Hess_val @ dcn)).flatten()

        return ddc.T


    # Returns the metric tensor
    def metric_tensor(self, c, nargout=1):
        # c, dc: D x N
        c = c.T  # N x D
        N, D = c.shape
        M = np.zeros((N, D, D))
        Id = np.eye(D)

        if nargout == 2:
            dM = np.zeros((N, D, D, D))

        for n in range(N):
            cn = c[n, :].reshape(-1, 1)

            Grad_val = grad_fun(cn.T).reshape(-1, 1)
            M[n, :, :] = Id + Grad_val * Grad_val.T

            # TODO: Check again the computation of the derivative. Can be done faster?
            if nargout == 2:
                Hess_val = Hess_fun(cn.T).reshape(D, D)
                for dd in range(D):
                    dM[n, :, :, dd] = Hess_val[:, dd].reshape(-1, 1) * Grad_val.T + Grad_val * Hess_val[:, dd].reshape(1, -1)

        if nargout == 1:
            return M
        else:
            return M, dM




# ================================================= #
# ==================== EXAMPLE ==================== #
# ================================================= #

# TODO: Change the functions bellow to implement different function-driven manifolds.
# Define a loss function e.g. f(x,y) = 0.5 * (x1^2 + x2^2)
def loss_fun(X):
    # input: X (NxD)
    return 0.5 * np.sum(X**2, 1)
    

# Define the gradient of the given function
def grad_fun(X):
    # input: X (NxD)
    return X


# Define the Hessian of the given function
def Hess_fun(X):
    # input: X (NxD)
    N, D = X.shape
    return np.tile(np.eye(D), (N, 1, 1))


# ===================================================== #
# ==================== MAIN SCRIPT ==================== #
# ===================================================== #
# Define the manifold class
manifold = loss_manifold(analytic=False)  # Return the analytic geodesic equation
# manifold = loss_manifold(analytic=False)  # Compute the geodesic equation using the general formulation


# Compute and plot the metric magnitude
N_grid = 30
X_grid = my_meshgrid(-3, 3, -3, 3, N=N_grid)
volM = np.sqrt(np.sum(grad_fun(X_grid) ** 2, axis=1))  # vol(M(x)) = sqrt*1 + ||Grad(x)||^2)
plt.contourf(X_grid[:, 0].reshape(N_grid, N_grid), X_grid[:, 1].reshape(N_grid, N_grid), volM.reshape(N_grid, N_grid))
plt.colorbar()

# Compute and plot some exponential maps (IVP) and the corresponding geodesic as BVP

# Exponential map
x = np.random.randn(2, 1)  # Base point
v = 2 * np.random.randn(2, 1)  # Initial velocity

c, failed = expmap(manifold, x, v)
T = np.linspace(0, 1, 100)
CURVE, DCURVE = c(T)
y = CURVE[:, -1]  # The last point on the curve
plt.plot(CURVE[0, :], CURVE[1, :], 'r', label='Expmap')

# Compute the geodesic between two points
solver = SolverBVP()
geodesic_solution = compute_geodesic(solver, manifold, x, y, solution=None)
CURVE_, DCURVE_ = geodesic_solution['curve'](T)
plt.plot(CURVE_[0, :], CURVE_[1, :], '--k', label='Geodesic')
plt.legend()

# Test the numerical difference in the length computation.
M = manifold.metric_tensor(x)
print(np.abs(np.sqrt(v.T @ M @ v) - geodesic_solution['length']))  # the length of the curve is equal to the tangent vector magnitude under the Riemannian metric.
