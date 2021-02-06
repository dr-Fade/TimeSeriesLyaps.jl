import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import lyapunovs as ls

def lorenz_example():
    σ, β, ρ = 10, 8/3, 28
    x0 = (12.5, 2.5, 1.5)
    T, n = 100, 10000
    def lorenz(t, X, σ, β, ρ):
        u, v, w = X
        up = -σ*(u - v)
        vp = ρ*u - v - u*w
        wp = -β*w + u*v
        return up, vp, wp
    soln = solve_ivp(lorenz, (0, T), x0, args=(σ, β, ρ), dense_output=True)
    t = np.linspace(0, T, n)
    attractor = soln.sol(t).T[1000:]

    # call ls.lyapunovs on the attractor and ask to return a convergence graph
    lyaps = ls.lyapunovs(attractor, 500, True)

    fig = plt.figure()
    trajectory = fig.add_subplot(1, 2, 1, projection='3d')
    lyaps_convergence = fig.add_subplot(1, 2, 2)
    trajectory.plot(attractor[:,0], attractor[:,1], attractor[:,2])
    lyaps_convergence.plot(lyaps[:,0])
    lyaps_convergence.plot(lyaps[:,1])
    lyaps_convergence.plot(lyaps[:,2])
    lyaps_convergence.grid(True)
    fig.show()
    print("The lyapunov spectrum is: " + str(lyaps[-1]))
    input("Press Enter to quit...")

def logistic_demo():
    def logistic(x0: float, r: float, n: int) -> np.ndarray:
        res = np.zeros((n,1))
        res[0] = x0
        for i in range(1,n):
            res[i] = r * res[i-1] * (1 - res[i-1])
        return res
    r_min = 1
    step = 0.01
    r_max = 4
    res = np.zeros(round((r_max-r_min)/step))
    current_point = 0
    r_range = np.arange(r_min, r_max, step)
    for r in r_range:
        trajectory = logistic(0.5, r, 100)[2:]

        # call ls.lyapunovs on the trajectory without returning convergence graph
        new_lyap = ls.lyapunovs(trajectory, 1)

        res[current_point] = new_lyap
        current_point += 1
        fig = plt.figure()
    fig = plt.figure()
    lyaps = fig.add_subplot(1, 1, 1)
    lyaps.plot(r_range, res)
    lyaps.grid(True)
    fig.show()
    input("Press Enter to quit...")
    plt.clf()
    plt.close('all')

lorenz_example()
logistic_demo()
