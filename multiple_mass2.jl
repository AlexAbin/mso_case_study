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

function mass_spring_damper!(du, u, p, t)
    x1, x2, x3  = u[1], u[2], u[3]
    v1, v2, v3  = u[4], u[5], u[6]

    f    = p[1]
    fe   = p[2]
    ff   = p[3]
    k    = p[4]
    c    = p[5]
    m    = p[6]
    l_12 = p[7]
    l_23 = p[8]

    du[1] = v1
    du[2] = v2
    du[3] = v3

    du[4] = (-c*v1 - k*x1 + k*((x2 - x1) - l_12) + c*(v2 -v1) + f + fe + ff) / m
    du[5] = (-c*(v2 - v1) - k*((x2 - x1) - l_12) + k*((x3 - x2) - l_23) + c*(v3 -v2)) / m
    du[6] = (-c*(v3 - v2) - k*((x3 - x2) - l_23)) / m

    du[7] = f * v1
end

tspan = (0.0, 10.0)
u0 = [0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0]
p0 = [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 1.0]
prob = ODEProblem(mass_spring_damper!, u0, tspan, p0)

cgrid = collect(0.0:0.2:10)

control_f = ControlParameter(
    cgrid,
    name     = :f,
    bounds   = (-20.0, 20.0),
    controls = zeros(length(cgrid))
)

layer = Corleone.SingleShootingLayer(
    prob, Tsit5();
    controls = (1 => control_f,),
    bounds_p = ([0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 1.0])
)

 function plot_msd(sol)
     fig = Figure()

     colors = Makie.wong_colors()

     ax1 = CairoMakie.Axis(fig[1, 1], title = "Position (m)")
     scatterlines!(ax1, sol, vars = [:x₁], label = "x1", color = colors[1])
     scatterlines!(ax1, sol, vars = [:x₂], label = "x2", color = colors[2])
     scatterlines!(ax1, sol, vars = [:x₃], label = "x3", color = colors[3])
     fig[1, 2] = Legend(fig, ax1, framevisible = false)

     ax2 = CairoMakie.Axis(fig[2, 1], title = "Velocity (m/s)")
     scatterlines!(ax2, sol, vars = [:x₄], label = "v1", color = colors[1])
     scatterlines!(ax2, sol, vars = [:x₅], label = "v2", color = colors[2])
     scatterlines!(ax2, sol, vars = [:x₆], label = "v3", color = colors[3])
     fig[2, 2] = Legend(fig, ax2, framevisible = false)

     ax3 = CairoMakie.Axis(fig[3, 1], title = "Control Force (N)")
     stairs!(ax3, sol, vars = [:f], label = "Force", color = colors[1])
     fig[3, 2] = Legend(fig, ax3, framevisible = false)

     return fig
 end

optprob = OptimizationProblem(layer, AutoForwardDiff(), Val(:ComponentArrays), loss = :x₃)

uopt = solve(
    optprob, Ipopt.Optimizer(),
    tol                   = 1.0e-5,
    hessian_approximation = "limited-memory",
    max_iter              = 1000,
    mu_strategy           = "adaptive"
)

ps, st = LuxCore.setup(Random.default_rng(), layer)
ax = getaxes(ComponentArray(ps))

wx    = 100.0
wv    = 20.0
xstar = u0[1:3] .+ 1.0

loss_expr = :(x₇ + $(wx/2)*((x₁-$(xstar[1]))^2 + (x₂-$(xstar[2]))^2 + (x₃-$(xstar[3]))^2) + $(wv/2)*(x₄^2 + x₅^2 + x₆^2))

println(loss_expr)

optprob = OptimizationProblem(layer, AutoForwardDiff(), Val(:ComponentArrays), loss = loss_expr)

uopt = solve(
    optprob, Ipopt.Optimizer(),
    tol                   = 1.0e-5,
    hessian_approximation = "limited-memory",
    max_iter              = 5000,
    mu_strategy           = "adaptive"
)

optsol, _ = layer(nothing, uopt + zero(ComponentArray(ps)), st)

println("Single shooting finished.")
println("Single shooting objective = ", uopt.objective)
println("Single shooting retcode = ", uopt.retcode)
println("Single shooting final state = ", optsol.u[end])

mkpath("results")

 fig_1 = plot_msd(optsol)
save("results/single_shooting_result.png", fig_1)
 fig_1

# Multiple Shooting

shooting_points = [0.0, 2.5, 5.0, 7.5, 10]

mslayer = MultipleShootingLayer(
    prob, Tsit5(), shooting_points...;
    controls = (1 => control_f,),
    bounds_p = ([0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 1.0])
)

msps, msst = LuxCore.setup(Random.default_rng(), mslayer)
msax = getaxes(ComponentArray(msps))

optprob_ms = OptimizationProblem(mslayer, AutoForwardDiff(), Val(:ComponentArrays), loss = loss_expr)

uopt_ms = solve(
    optprob_ms, Ipopt.Optimizer(),
    tol                   = 1.0e-5,
    hessian_approximation = "limited-memory",
    max_iter              = 5000,
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
save("results/multiple_shooting_result.png", fig_2)
 fig_2

