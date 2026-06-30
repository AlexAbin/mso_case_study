import Pkg
# Activates the local environment containing the Project.toml file
Pkg.activate(".") 
# Downloads and builds any missing dependencies listed in Project.toml
Pkg.instantiate() 

using CairoMakie

# Configuration flags for your solvers
const AMPL = false 

println("Environment instantiated and CairoMakie loaded successfully.")