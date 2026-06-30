#  Single Mass-Spring-Damper

using Corleone
using LuxCore
using Random
using OrdinaryDiffEqTsit5
using SymbolicIndexingInterface
using Optimization
using OptimizationMOI
using Ipopt
using ComponentArrays
using CairoMakie  

function crane_dynamics!(du, u, p, t)
    x_trolley, x_hook, x_payload  = u[1], u[2], u[3]    # positions
    v_trolley, v_hook, v_payload  = u[4], u[5], u[6]    # velocities

    f     = p[1]    # motor force on trolley
    fe    = p[2]    # exteral force
    ff    = p[3]    # friction force 
    k_th  = p[4]    # trolley-hook link stiffness
    c_th  = p[5]    # trolley-hook link damping
    k_hp  = p[6]    # hoist cable stiffness
    c_hp  = p[7]    # hoist cable damping
    m     = p[8]    # mass
    l_th  = p[9]    # trolley-hook link rest length
    l_hp  = p[10]   # hoist cable rest length

    du[1] = v_trolley
    du[2] = v_hook
    du[3] = v_payload

    du[4] = (k_th*((x_hook - x_trolley) - l_th) + c_th*(v_hook - v_trolley) + f + fe + ff) / m
    du[5] = (-c_th*(v_hook - v_trolley) - k_th*((x_hook - x_trolley) - l_th) + k_hp*((x_payload - x_hook) - l_hp) + c_hp*(v_payload - v_hook)) / m
    du[6] = (-c_hp*(v_payload - v_hook) - k_hp*((x_payload - x_hook) - l_hp)) / m

    #du[7] = f * v_trolley
    du[7] = 0.5 * f^2
end

tspan = (0.0, 10.0)
u0 = [0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0]
p0 = [0.0, 0.0, 0.0, 3.2, 2.3, 5.8, 4.5, 1.0, 1.0, 1.0]
# f, fr, ff, k_th, c_th, k_hp, c_hp, m, l_th, l_hp
# (using Parsi et al., 2023 values: k12=3.2,c12=2.3,k23=5.8,c23=4.5,m=1.0, but with m = 1.0 kg for simplicity)
prob = ODEProblem(crane_dynamics!, u0, tspan, p0)


wx = 100.0
wv = 20.0
xstar = u0[1:3] .+ 1.0

loss_expr = :(x₇ + $(wx/2)*((x₁-$(xstar[1]))^2 + (x₂-$(xstar[2]))^2 + (x₃-$(xstar[3]))^2) + $(wv/2)*(x₄^2 + x₅^2 + x₆^2))
# --- 3. Convergence Study Loop ---
# Test these control discretization step sizes
dt_candidates = [1.0, 0.5, 0.25, 0.1, 0.05, 0.025]

objective_values = Float64[]
solve_times      = Float64[]

println("Starting Single Shooting Convergence Study...")

for dt in dt_candidates
    println("Testing dt = ", dt)
    
    # Define control grid based on current dt
    cgrid = collect(0.0 : dt : 10.0)
    
    control_f = ControlParameter(
        cgrid,
        name     = :f,
        bounds   = (-20.0, 20.0),
        controls = zeros(length(cgrid))
    )

    # Build the single shooting layer with the dynamic control grid and safe bounds
    layer = Corleone.SingleShootingLayer(
        prob, Tsit5();
        controls = (1 => control_f,),
        bounds_p = ([0.0, 0.0, 3.2, 2.3, 5.8, 4.5, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 3.2, 2.3, 5.8, 4.5, 1.0, 1.0, 1.0])
    )

    ps, st = LuxCore.setup(Random.default_rng(), layer)
    optprob = OptimizationProblem(layer, AutoForwardDiff(), Val(:ComponentArrays), loss = loss_expr)

    # Time the optimization
    time_taken = @elapsed begin
        uopt = solve(
            optprob, Ipopt.Optimizer(),
            tol                   = 1.0e-5,
            hessian_approximation = "limited-memory",
            max_iter              = 3000,
            mu_strategy           = "adaptive",
            print_level           = 0  # Silence Ipopt to keep the console clean
        )
    end
    
    # Save the metrics
    push!(objective_values, uopt.objective)
    push!(solve_times, time_taken)
end

# --- 4. Plot Convergence Analysis ---
fig_conv = Figure(size = (800, 400))

# Left Plot: Cost vs dt
ax1 = CairoMakie.Axis(fig_conv[1, 1], title = "Objective Cost vs. Step Size", xlabel = "Step Size (dt)", ylabel = "Cost")
lines!(ax1, dt_candidates, objective_values, color = :blue, linewidth = 2)
scatter!(ax1, dt_candidates, objective_values, color = :blue, markersize = 12)

# Right Plot: Solve Time vs dt
ax2 = CairoMakie.Axis(fig_conv[1, 2], title = "Solve Time vs. Step Size", xlabel = "Step Size (dt)", ylabel = "Time (seconds)")
lines!(ax2, dt_candidates, solve_times, color = :red, linewidth = 2)
scatter!(ax2, dt_candidates, solve_times, color = :red, markersize = 12)

# Reverse x-axis so the graph reads left-to-right as the grid gets finer (smaller dt)
ax1.xreversed = true
ax2.xreversed = true

mkpath("results_crane_convergence_analysis")
save("results_crane_convergence_analysis/convergence_analysis.png", fig_conv)

# Display convergence figure
display(fig_conv)

function plot_msd(sol)
    fig = Figure(size = (800, 900)) # Sized appropriately for 3 stacked plots

    colors = Makie.wong_colors()

    # Position Subplot
    ax1 = CairoMakie.Axis(fig[1, 1], title = "Position (m)", ylabel = "Distance")
    scatterlines!(ax1, sol, vars = [:x₁], label = "Trolley (x1)", color = colors[1], markersize = 6)
    scatterlines!(ax1, sol, vars = [:x₂], label = "Hook (x2)", color = colors[2], markersize = 6)
    scatterlines!(ax1, sol, vars = [:x₃], label = "Payload (x3)", color = colors[3], markersize = 6)
    fig[1, 2] = Legend(fig, ax1, framevisible = false)

    # Velocity Subplot
    ax2 = CairoMakie.Axis(fig[2, 1], title = "Velocity (m/s)", ylabel = "Speed")
    scatterlines!(ax2, sol, vars = [:x₄], label = "v1", color = colors[1], markersize = 6)
    scatterlines!(ax2, sol, vars = [:x₅], label = "v2", color = colors[2], markersize = 6)
    scatterlines!(ax2, sol, vars = [:x₆], label = "v3", color = colors[3], markersize = 6)
    fig[2, 2] = Legend(fig, ax2, framevisible = false)

    # Control Force Subplot
    ax3 = CairoMakie.Axis(fig[3, 1], title = "Control Force (N)", xlabel = "Time (s)", ylabel = "Force")
    stairs!(ax3, sol, vars = [:f], label = "Motor Force", color = colors[1], linewidth = 2)
    fig[3, 2] = Legend(fig, ax3, framevisible = false)

    return fig
end

# Generate and save the plot
mkpath("results_crane")
fig_optimal = plot_msd(optsol)
save("results_crane/optimal_trajectory_single_shooting.png", fig_optimal)

# Display the figure (if running in a notebook/REPL)
fig_optimal