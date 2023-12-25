using DynamicalSystems, DifferentialEquations, ChaosTools, LinearAlgebra, DelimitedFiles, Smoothing, Plots
include("TimeSeriesLyaps.jl")

function lorenz_real_lyaps() 
    T = 25
    Δt = 0.01
    σ = 10.0
    ρ = 28.0
    β = 8/3
    lorenz = DynamicalSystems.Systems.lorenz([12.5, 2.5, 1.5]; σ = σ, ρ = ρ, β = β)
    trajectory = DynamicalSystems.trajectory(lorenz, T; Δt=Δt)[1000:end,:]
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
        ] * Δt
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
    return (res / Δt)
end

function lorenz_demo()
    T = 25
    Δt = 0.01
    lorenz = DynamicalSystems.Systems.lorenz([12.5, 2.5, 1.5]; σ = 10.0, ρ = 28.0, β = 8/3)
    embedded = DynamicalSystems.trajectory(lorenz, T; Δt=Δt)[1][1000:end,:]
    return TimeSeriesLyaps.calculate_spectrum(embedded, 500; ks=[200], normalize_spectrum = true, return_convergence = true)
end

function embedded_lorenz_demo()
    T = 25
    Δt = 0.01
    lorenz = DynamicalSystems.Systems.lorenz([12.5, 2.5, 1.5]; σ = 10.0, ρ = 28.0, β = 8/3)
    embedded = embed(vec(DynamicalSystems.trajectory(lorenz, T; Δt=Δt)[1][1000:end,1]), 3, 14)
    return TimeSeriesLyaps.calculate_spectrum(embedded, 500)
end

function roessler_demo()
    T = 500
    Δt = 0.01
    roessler = DynamicalSystems.Systems.roessler([.1, .1, .1]; a = 0.15, b = 0.2, c = 10)
    embedded = DynamicalSystems.trajectory(roessler, T; Δt=Δt)[500:end,:]
    return TimeSeriesLyaps.calculate_spectrum(embedded, 500)
end

function roessler_real_lyaps()
    T = 500
    Δt = 0.01
    a = 0.15
    b = 0.2
    c = 10
    roessler = DynamicalSystems.Systems.roessler([.1, .1, .1]; a = a, b = b, c = c)
    trajectory = DynamicalSystems.trajectory(roessler, T; Δt=Δt)[500:end,:]
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
        ] * Δt
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
    return (res / Δt)
end

function logistic_demo()
    res = Array{Int}(undef, 0, 1)
    for r in 1 : 0.001 : 4
        logistic = DynamicalSystems.Systems.logistic(0.5; r = r)
        trajectory = DynamicalSystems.trajectory(logistic, 100)[2:end]
        new_lyap = TimeSeriesLyaps.calculate_spectrum(Dataset(trajectory), 1; return_convergence = false)
        res = [res; new_lyap]
    end
    return res
end

lm(r) = begin
    logistic = DynamicalSystems.Systems.logistic(0.5; r = r)
    [[r x] for x in unique(DynamicalSystems.trajectory(logistic, 200)[100:end])]
end

logistic_map() = [lm(r) for r in 1 : 0.01 : 4]

function hyperchaos_roessler_demo()
    function chen(du,u,p,t)
        x, y, z = u
        du[1] = -0.4x + y
        du[2] = x + 0.3y - x*z
        du[3] = -0.1z + y^2 - 1
    end
    u0 = [-10.,-6.,0]
    tspan = (0.00,2000.0)
    prob = ODEProblem(chen, u0, tspan, [])
    sol = solve(prob)
    data = sol.u[7000:end] |> Dataset
    τ = estimate_delay(data, "mi_min")
    D = estimate_dimension(data, τ, 1:10) |> x -> findfirst(y -> y >= 0.95, x)
    embedded = embed(data, D, τ)
    return TimeSeriesLyaps.calculate_spectrum(embedded, 500)
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
    return TimeSeriesLyaps.calculate_spectrum(embedded, 500)
end

get_data_array(filename::String, point::String) =
    open(filename, "r") do file
        while !eof(file)
            line = readline(file)
            if length(line) > length(point) && line[begin:length(point)] == point
                return line[length(point)+1:end] |>
                    x -> split(x, "\t") |>
                    x -> filter(y -> y != "", x) |>
                    x -> [parse(Float32, y) for y in x]
            end
        end
    end

function eeg_demo(filename::String, point::String; smoothing::Int64 = 1)
    data = Smoothing.binomial(get_data_array(filename, point), smoothing)
    τ = estimate_delay(data, "mi_min")
    indicator = stochastic_indicator(data, τ, 1:10)
    D = indicator |> findmin |> x -> x[end]
    embedded = embed(data, D, τ)
    return (TimeSeriesLyaps.calculate_spectrum(embedded, 10), indicator)
end

function test_all_eeg(data_dir::String, points::Vector{String}; file_extension::String = "")
    if !isdir(data_dir)
        @error "Directory not found $data_dir"
        return
    end
    for f in readdir(data_dir; join=true)
        dir_name, extn = splitext(f)
        if extn != file_extension
            @warn "Skipping file with incorrect extension: $f"
            continue
        end
        if !isdir(dir_name)
            mkdir(dir_name)
        end
        for p in points
            for s in 1:5
                lyaps = eeg_demo(f, p; smoothing=s);
                plot(lyaps[begin]); # suppressing the output to not draw the actual plot
                res_file_name = "$(dir_name)/$(p)_[smoothing=$s]_[lyaps=$(lyaps[begin][end,:])].png"
                savefig(res_file_name)
            end
        end
    end
end

function conservative_4d_demo()
    A = [
        0 -3 1 1;
        3 0 2 0;
        -1 -2 0 4;
        -1 0 -4 0;
    ]
    g = (f, k) -> begin
        x, y, z, u = f
        [
            x * abs(x) + k[1]*x
            y * abs(y) + k[2]*y
            z * abs(z) + k[3]*z
            u * abs(u) + k[4]*u
        ]
    end
    function ode(du,u,p,t)
        du .= A * g(u, p)
    end
    tspan = (0.00,10.0)
    map(
        k -> "k = $k:\n" * "$(map(
            u0 -> "\tf₀ = $u0: λ = $(lyapunovspectrum(
                ODEProblem(ode, u0, tspan, k) |> ContinuousDynamicalSystem, 10000
            ) |> λ -> map(x -> round(x; digits = 2), λ))\n",
            [
                [0.1, 0.1, 0.1, 0.1],
                [-0.1, 0.1, 0.1, 0.1],
                [-0.1, 0.1, -0.1, 0.1],
                [-0.1, -0.1, -0.1, 0.1],
            ]
        ) |> s -> reduce(*, s))",
        [
            [1, 1, 1, 1],
            [-1, 1, 1, 1],
            [-1, 1, -1, 1],
            [-1, -1, -1, 1],
        ]
    ) |> s -> reduce(*, s)
end
