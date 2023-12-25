module TimeSeriesLyaps

using DynamicalSystems, LinearAlgebra, ChaosTools
include("./local_jacobian.jl")

export calculate_spectrum

"""
Estimate a spectrum of Lyapunov exponents for the time series using a QR-based algorithm with numerical local Jacobian estimation.
"""
function calculate_spectrum(attractor::Dataset, theiler_window::Int; ks=[100], return_convergence = false, normalize_spectrum=true)
    # total number of points and a dimension of the problem
    N, m = size(attractor)
    # k-d tree for quick search for neighboring points
    # the last point is excluded because it doesn't havea successor and it's
    # impossible to estimate local jacobian in it
    kd_tree = KDTree(attractor[begin:end-1])
    # initial values for the exponents are zeros
    λ = zeros(m)
    res = zeros(1, m)
    # initial value the orthogonal deviation vectors
    Y = I(m)
    for t = 1 : N - 1
        # calculate the local expansion rate
        J = local_jacobian(t, attractor, kd_tree, theiler_window)
        # apply the local jacobian by the tangent vectors and reorthogonalize them
        Q, R = LinearAlgebra.qr(J*Y)
        # take the logarithms of the new tangent vectors' norms and accumulate their values
        for i in 1:m
            λ[i] += R[i, i] |> abs |> log
        end
        # initialize the tangent vectors set with the new vectors of unit length
        Y = Q
        if return_convergence
            res = [res; λ' / t]
        end
    end
    λ = λ / (N-1)
    η = if normalize_spectrum
            _η = lyapunov_from_data(attractor, ks; w=theiler_window)[begin] / λ[begin]
            if _η * λ[begin] < 0
                error("Failed to estimate the Largest Lyapunov Exponent correctly! Please, provide different ks argument.")
            end
            _η
        else
            1
        end
    if return_convergence
        return  η * res
    end
    return η * λ
end

end #module
