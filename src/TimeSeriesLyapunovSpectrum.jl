module TimeSeriesLyapunovSpectrum

using DynamicalSystems, LinearAlgebra, ChaosTools
include("./LocalJacobian.jl")

function calculate_spectrum(attractor::Dataset, theiler_window::Int; return_convergence = true)
    # total number of points and a dimension of the problem
    N, m = size(attractor)
    # k-d tree for quick search for neighboring points
    # the last point is excluded because it doesn't havea successor and it's
    # impossible to estimate local jacobian in it
    kd_tree = KDTree(attractor[begin:end-1])
    # initial values for the exponents are zeros
    λ = zeros(m)
    res = zeros(1, m)
    successful_estimations = 0
    # initial value the orthogonal deviation vectors
    Y = I(m)
    for t = 1 : N - 1
        # calculate the local expansion rate
        J = LocalJacobian.local_jacobian(t, attractor, kd_tree, theiler_window)
        # apply the local jacobian by the tangent vectors and reorthogonalize them
        Q, R = LinearAlgebra.qr(J*Y)
        # take the logarithms of the new tangent vectors' norms and accumulate their values
        for i in 1:m
            λ[i] += R[i, i] |> abs |> log
        end
        successful_estimations += 1
        # initialize the tangent vectors set with the new vectors of unit length
        Y = Q
        if return_convergence
            res = [res; λ' / successful_estimations]
        end
    end
    # normalize the exponents
    if return_convergence
        return res
    end
    return λ / successful_estimations
end

end #module
