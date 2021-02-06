module LyapunovSpectrumExample

using DynamicalSystems, DifferentialEquations, ChaosTools, LinearAlgebra
include("TimeSeriesLyapunovSpectrum.jl")

function lorenz_real_lyaps()
    T = 25
    dt = 0.001
    σ = 10.0
    ρ = 28.0
    β = 8/3
    lorenz = DynamicalSystems.Systems.lorenz([12.5, 2.5, 1.5]; σ = σ, ρ = ρ, β = β)
    trajectory = DynamicalSystems.trajectory(lorenz, T; dt=dt)
    N, m = size(trajectory)
    # initial values for the exponents are zeros
    λ = zeros(m)
    res = zeros(1, m)
    successful_estimations = 0
    # initial value the orthogonal deviation vectors
    Y = I(m)
    for t = 1 : N-1
        # calculate the local expansion rate
        J(x) = I(3) + [
            (-σ)        σ       0;
            (ρ-x[3])    (-1)    (-x[1]);
            x[2]        x[1]    (-β)
        ] * dt
        # apply the local jacobian by the tangent vectors and reorthogonalize them
        Q, R = LinearAlgebra.qr(J(trajectory[t])*Y)
        # take the logarithms of the new tangent vectors' norms and accumulate their values
        for i in 1:m
            λ[i] += R[i, i] |> abs |> log
        end
        successful_estimations += 1
        # initialize the tangent vectors set with the new vectors of unit length
        Y = Q
        res = [res; λ' / successful_estimations]
    end
    # normalize the exponents
    return (res / dt)
end

function lorenz_demo()
    T = 25
    dt = 0.001
    lorenz = DynamicalSystems.Systems.lorenz([12.5, 2.5, 1.5]; σ = 10.0, ρ = 28.0, β = 8/3)
    embedded = DynamicalSystems.trajectory(lorenz, T; dt=dt)[500:end,:]
    return TimeSeriesLyapunovSpectrum.calculate_spectrum(embedded, 500)
end

function roessler_demo()
    T = 500
    dt = 0.01
    roessler = DynamicalSystems.Systems.roessler([.1, .1, .1]; a = 0.15, b = 0.2, c = 10)
    embedded = DynamicalSystems.trajectory(roessler, T; dt=dt)[500:end,:]
    return TimeSeriesLyapunovSpectrum.calculate_spectrum(embedded, 500)
end

function roessler_real_lyaps()
    T = 500
    dt = 0.01
    a = 0.15
    b = 0.2
    c = 10
    roessler = DynamicalSystems.Systems.roessler([.1, .1, .1]; a = a, b = b, c = c)
    trajectory = DynamicalSystems.trajectory(roessler, T; dt=dt)
    N, m = size(trajectory)
    # initial values for the exponents are zeros
    λ = zeros(m)
    res = zeros(1, m)
    successful_estimations = 0
    # initial value the orthogonal deviation vectors
    Y = I(m)
    for t = 1 : N-1
        # calculate the local expansion rate
        J(x) = I(3) + [
            0       -1      -1;
            1       a       0;
            x[3]    0    (x[1]-c)
        ] * dt
        # apply the local jacobian by the tangent vectors and reorthogonalize them
        Q, R = LinearAlgebra.qr(J(trajectory[t])*Y)
        # take the logarithms of the new tangent vectors' norms and accumulate their values
        for i in 1:m
            λ[i] += R[i, i] |> abs |> log
        end
        successful_estimations += 1
        # initialize the tangent vectors set with the new vectors of unit length
        Y = Q
        res = [res; λ' / successful_estimations]
    end
    # normalize the exponents
    return (res / dt)
end

function logistic_demo()
    res = Array{Int}(undef, 0, 1)
    for r in 1 : 0.001 : 4
        logistic = DynamicalSystems.Systems.logistic(0.5; r = r)
        trajectory = DynamicalSystems.trajectory(logistic, 100)[2:end]
        new_lyap = TimeSeriesLyapunovSpectrum.calculate_spectrum(Dataset(trajectory), 1; return_convergence = false)
        res = [res; new_lyap]
    end
    return res
end

function hyperchaos_roessler_demo()
    a = 0.25
    b = 3
    c = 0.5
    d = 0.05
    function hyper_roessler(du,u,p,t)
        x, y, z, w = u
        a, b, c, d = p
        du[1] = -y - z
        du[2] = x + a*y + w
        du[3] = b + x * z
        du[4] = -c * z + d * w
    end
    u0 = [-10.,-6.,0,10.]
    tspan = (0.00,500.0)
    p = [a, b, c, d]
    prob = ODEProblem(hyper_roessler, u0, tspan, p)
    sol = solve(prob)
    data = map(x -> x[1], sol.u)
    τ = estimate_delay(data, "mi_min")
    D = estimate_dimension(data, τ, 1:10) |> x -> findfirst(y -> y >= 0.95, x)
    embedded = embed(data, D, τ)
    return TimeSeriesLyapunovSpectrum.calculate_spectrum(embedded, 500)
end

function eeg_demo(filename::String)
    raw_data = read(filename, String) |>
        x -> replace(x, r"\t" => " ") |>
        x -> replace(x, r"\d:\d:\d\d?.\d\d\d? " => "") |>
        x -> split(x, "\n") |>
        x -> filter(str -> str != "", x) |>
        x -> map(line -> split(line, " ") |>
        x -> map(num -> parse(Float64, num), x), x)

    n = 2
    data = raw_data[1][n]
    for array in raw_data[2:end]
        data = [data; array[n]]
    end
    τ = estimate_delay(data, "mi_min")
    D = estimate_dimension(data, τ, 1:10) |> x -> findfirst(y -> y >= 0.9, x)
    embedded = embed(data, D, τ)
    return TimeSeriesLyapunovSpectrum.calculate_spectrum(embedded, 10)
end

end # module
