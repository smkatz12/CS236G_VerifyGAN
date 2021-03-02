using PGFPlots
using Colors
using ColorBrewer

function plot_prob(tree)
    nbin = 200
    xmin = tree.lbs[1]
    xmax = tree.ubs[1]
    ymin = tree.lbs[2]
    ymax = tree.ubs[2]

    safe_color = RGB(0.0, 0.0, 1.0) # blue
    unsafe_color = RGB(1.0, 0.0, 0.0) # red

    colors = [safe_color, unsafe_color]

    ax = Axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, width="7cm", height="8cm", 
    ylabel=L"$\theta$ (degrees)", xlabel=L"$x$ (meters)", title="Safe Cells")

    function get_heat(x, y)
        leaf = get_leaf(tree.root_node, [x, y])
        return leaf.prob
    end

    push!(ax, Plots.Image(get_heat, (xmin, xmax), (ymin, ymax), zmin = 0, zmax = 1,
        xbins = nbin, ybins = nbin, colormap = ColorMaps.RGBArrayMap(colors), colorbar=false))

    return ax
end

function plot_control_range(tree; nbin = 200)
    xmin = tree.lbs[1]
    xmax = tree.ubs[1]
    ymin = tree.lbs[2]
    ymax = tree.ubs[2]


    ax = Axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, width="7cm", height="8cm", 
    ylabel=L"$\theta$ (degrees)", xlabel=L"$x$ (meters)", title="Control Range")

    function get_heat(x, y)
        leaf = get_leaf(tree.root_node, [x, y])
        return leaf.max_control - leaf.min_control
    end

    push!(ax, Plots.Image(get_heat, (xmin, xmax), (ymin, ymax),
        xbins = nbin, ybins = nbin, colormap = pasteljet, colorbar=false))

    return ax
end

function plot_reachable(tree)
    nbin = 200
    xmin = tree.lbs[1]
    xmax = tree.ubs[1]
    ymin = tree.lbs[2]
    ymax = tree.ubs[2]

    unreach_color = RGB(1.0, 1.0, 1.0) # white
    reach_color = RGB(0.0, 0.0, 0.502) # navy blue

    colors = [unreach_color, reach_color]

    ax = Axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, width="7cm", height="8cm", 
    ylabel=L"$\theta$ (degrees)", xlabel=L"$x$ (meters)", title="Reachable Cells")

    function get_heat(x, y)
        leaf = get_leaf(tree.root_node, [x, y])
        return leaf.prob
    end

    push!(ax, Plots.Image(get_heat, (xmin, xmax), (ymin, ymax), zmin = 0, zmax = 1,
        xbins = nbin, ybins = nbin, colormap = ColorMaps.RGBArrayMap(colors), colorbar=false))

    return ax
end