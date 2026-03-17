#=
Function to easily compare both results with one unified plot.

TO-DO: improve this description
=#

using Plots
using DataFrames
using CSV
using LaTeXStrings

# Open the data
df_bb84 = CSV.read("SSPvsBB84v2_18-03/data/BB84_depolarized_data.csv", DataFrame)
df_ssp  = CSV.read("SSPvsBB84v2_18-03/data/SSP_depolarized_data.csv", DataFrame)

# Extract the x-axis (distance)
D_mod = df_bb84.D_km

# Define the ϵ values we've used for the simulations (to be
# able to access the dictionary's values).
ϵ_mod = [0.0, 1e-6, 1e-3] 

# Style configuration
default(
    fontfamily = "Computer Modern", # LaTeX typography
    framestyle = :box,              # Boxed style
    minorgrid = true,               # Detailed grid
    gridalpha = 0.3,                # Less intense grid (translucent)
    tickfontsize = 10,              # Axis' font size
    guidefontsize = 12,             # Title's font size
    legendfontsize = 10             # Llegend's font size
)

# Add the axis labels, the legend and configure the logaritmic scale
plot(
    xlabel = L"\textrm{Distance~} D \textrm{~(km)}",
    ylabel = L"\textrm{Secret-key~rate~} R_\infty \textrm{~(bits~per~pulse)}",
    title = L"\textrm{\textbf{BB84 vs SSP:~}} \mathbf{p_{\textrm{\textbf{dep.}}} = 0.1}",
    yscale = :log10,
    ylims = (1e-7, 1.0),
    legend = :bottomleft
)

# Choose a color palette to be sure each ϵ gets the same color
colors = palette(:default)

for (i, ϵ) in enumerate(ϵ_mod)
    # Rebuild the column's name as we saved it in the CSV
    col_name = "SKR_$ϵ"
    
    # 1. Plot BB84's line (solid line: linestyle=:solid)
    plot!(D_mod, df_bb84[!, col_name], 
          label="BB84 (ϵ = $ϵ)", 
          linewidth=1, 
          linestyle=:solid,
          color=colors[i], 
          marker=:none)
          
    # 2. Plot SSP's line (dashed line: linestyle=:dash)
    plot!(D_mod, df_ssp[!, col_name], 
          label="SSP (ϵ = $ϵ)", 
          linewidth=1, 
          linestyle=:dash,
          color=colors[i], 
          marker=:none)
end

# Show the graph
display(current())

# Save the graph in .pdf format (optional)
# savefig("./plots/plot_comparison.pdf")