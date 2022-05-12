module LocalJacobian

using DynamicalSystems, DifferentialEquations, ChaosTools, LinearAlgebra, NearestNeighbors

function local_jacobian(t::Int64, attractor::Dataset, tree::KDTree, theiler_window::Int64)
    # attractor dimensions
    _, m = size(attractor)
    # find 2*m nearest neighbors and add the current point to the pool so that
    # we will have 2*m+1 points to work with
    indxs, = knn(tree, attractor[t], 2*m, false, i -> abs(i-t) < theiler_window)
    neighbors = [t; indxs]
    # matrix of neighbors (X) and their successors (Y)
    X = attractor[neighbors[1],:]
    Y = attractor[neighbors[1]+1,:]
    for i in neighbors[2:end]
        X = [X Array(attractor[i,:])]
        Y = [Y Array(attractor[i+1,:])]
    end
    X =  [ones(length(neighbors),1) Array(X')]
    Y = Array(Y')
    # estimate the local jacobian
    J = Array{Float64}(undef, 0, m)
    for i = 1:m
        J = [J; Array(X \ Y[:,i])'[:,2:end]]
    end
    return J
end

end # module
