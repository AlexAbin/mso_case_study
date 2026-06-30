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

dt_list = [1.0, 0.5, 0.25, 0.1, 0.05, 0.025]

objective_values = Float64[]
solve_times      = Float64[]

println("Convergence Study")
for dt in dt_list
    println("Testing dt = ", dt)
    
    # Defining timegrid based on current dt
    cgrid = collect(0.0 : dt : 10.0)
    
    control_f = ControlParameter(
        cgrid,
        name     = :f,
        bounds   = (-20.0, 20.0),
        controls = zeros(length(cgrid))
    )

    layer = Corleone.SingleShootingLayer(
        prob, Tsit5();
        controls = (1 => control_f,),
        bounds_p = ([0.0, 0.0, 3.2, 2.3, 5.8, 4.5, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 3.2, 2.3, 5.8, 4.5, 1.0, 1.0, 1.0])
    )

    ps, st = LuxCore.setup(Random.default_rng(), layer)
    optprob = OptimizationProblem(layer, AutoForwardDiff(), Val(:ComponentArrays), loss = loss_expr)

    time_taken = @elapsed begin
        uopt = solve(
            optprob, Ipopt.Optimizer(),
            tol                   = 1.0e-5,
            hessian_approximation = "limited-memory",
            max_iter              = 3000,
            mu_strategy           = "adaptive",
            print_level           = 0  
        )
    end
    
    push!(objective_values, uopt.objective)
    push!(solve_times, time_taken)
end

function plot_convergence(dts, objs, times)
    fig = Figure(size = (800, 400))

    # Cost 
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel = "Step Size (dt)", ylabel = "Objective Value")
    lines!(ax1, dts, objs, color = :blue, linewidth = 2)
    scatter!(ax1, dts, objs, color = :blue, markersize = 12)
    ax1.xreversed = true

    # Time
    ax2 = CairoMakie.Axis(fig[1, 2], xlabel = "Step Size (dt)", ylabel = "Time (s)")
    lines!(ax2, dts, times, color = :red, linewidth = 2)
    scatter!(ax2, dts, times, color = :red, markersize = 12)
    ax2.xreversed = true

    return fig
end

mkpath("results_crane")
fig_conv = plot_convergence(dt_list, objective_values, solve_times)
save("results_crane/convergence_analysis.png", fig_conv)
display(fig_conv)

dt = dt_list[5] 
cgrid = collect(0.0 :dt :10)

control_f = ControlParameter(
    cgrid,
    name     = :f,
    bounds   = (-20.0, 20.0),
    controls = zeros(length(cgrid))
)

layer = Corleone.SingleShootingLayer(
    prob, Tsit5();
    controls = (1 => control_f,),
    bounds_p = ([0.0, 0.0, 3.2, 2.3, 5.8, 4.5, 1.0, 1.0, 1.0],
                [0.0, 0.0, 3.2, 2.3, 5.8, 4.5, 1.0, 1.0, 1.0])
)

 function plot_msd(sol)
     fig = Figure()

     colors = Makie.wong_colors()

     ax1 = CairoMakie.Axis(fig[1, 1], title = "Position (m)")
     lines!(ax1, sol, vars = [:x₁], label = "Trolley Position (x1)", color = colors[1])
     lines!(ax1, sol, vars = [:x₂], label = "Hook Position (x2)"   , color = colors[2])
     lines!(ax1, sol, vars = [:x₃], label = "Payload Position (x3)", color = colors[3])
     fig[1, 2] = Legend(fig, ax1, framevisible = false)

     ax2 = CairoMakie.Axis(fig[2, 1], title = "Velocity (m/s)")
     lines!(ax2, sol, vars = [:x₄], label = "Trolley Velocity (x4)", color = colors[1])
     lines!(ax2, sol, vars = [:x₅], label = "Hook Velocity (x5)"   , color = colors[2])
     lines!(ax2, sol, vars = [:x₆], label = "Payload Velocity (x6)", color = colors[3])
     fig[2, 2] = Legend(fig, ax2, framevisible = false)

     ax3 = CairoMakie.Axis(fig[3, 1],xlabel = "Time (s)", title = "Control Force (N)")
     stairs!(ax3, sol, vars = [:f], label = "Motor Force", color = colors[1])
     fig[3, 2] = Legend(fig, ax3, framevisible = false)

     return fig
 end

optprob = OptimizationProblem(layer, AutoForwardDiff(), Val(:ComponentArrays), loss = :x₇)

uopt = solve(
    optprob, Ipopt.Optimizer(),
    tol                   = 1.0e-5,
    hessian_approximation = "limited-memory",
    max_iter              = 3000,
    mu_strategy           = "adaptive"
)

ps, st = LuxCore.setup(Random.default_rng(), layer)
ax = getaxes(ComponentArray(ps))

wx = 100.0
wv = 20.0
xstar = u0[1:3] .+ 1.0

loss_expr = :(x₇ + $(wx/2)*((x₁-$(xstar[1]))^2 + (x₂-$(xstar[2]))^2 + (x₃-$(xstar[3]))^2) + $(wv/2)*(x₄^2 + x₅^2 + x₆^2))

println(loss_expr)

optprob = OptimizationProblem(layer, AutoForwardDiff(), Val(:ComponentArrays), loss = loss_expr)

uopt = solve(
    optprob, Ipopt.Optimizer(),
    tol                   = 1.0e-5,
    hessian_approximation = "limited-memory",
    max_iter              = 3000,
    mu_strategy           = "adaptive"
)

optsol, _ = layer(nothing, uopt + zero(ComponentArray(ps)), st)

println("Single shooting finished.")
println("Single shooting objective = ", uopt.objective)
println("Single shooting retcode = ", uopt.retcode)
println("Single shooting final state = ", optsol.u[end])

mkpath("results_crane")

 fig_1 = plot_msd(optsol)
save("results_crane/single_shooting_result.png", fig_1)
 fig_1

# Multiple Shooting

shooting_points = collect(0.0:1.0:10.0)

mslayer = MultipleShootingLayer(
    prob, Tsit5(), shooting_points...;
    controls = (1 => control_f,),
    bounds_p = ([0.0, 0.0, 3.2, 2.3, 5.8, 4.5, 1.0, 1.0, 1.0],
                [0.0, 0.0, 3.2, 2.3, 5.8, 4.5, 1.0, 1.0, 1.0])
)

msps, msst = LuxCore.setup(Random.default_rng(), mslayer)
msax = getaxes(ComponentArray(msps))

optprob_ms = OptimizationProblem(mslayer, AutoForwardDiff(), Val(:ComponentArrays), loss = loss_expr)

uopt_ms = solve(
    optprob_ms, Ipopt.Optimizer(),
    tol                   = 1.0e-5,
    hessian_approximation = "limited-memory",
    max_iter              = 3000,
    mu_strategy           = "adaptive"
)

mssol, _ = mslayer(nothing, uopt_ms + zero(ComponentArray(msps)), msst)

println("Multiple shooting finished.")
println("Multiple shooting objective = ", uopt_ms.objective)
println("Multiple shooting retcode = ", uopt_ms.retcode)
println("Multiple shooting final state = ", mssol.u[end])

println("First time = ", first(mssol.t))
println("Last time  = ", last(mssol.t))
println("Stored times = ", mssol.t)

 fig_2 = plot_msd(mssol)
 save("results_crane/multiple_shooting_result.png", fig_2)
 fig_2

