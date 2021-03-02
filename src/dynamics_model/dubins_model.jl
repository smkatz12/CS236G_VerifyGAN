# Parameters

L = 5 # m
v = 5 # m/s

# Dynamics model functions

function next_state(x, θ, ϕ; dt = 0.05)
    # x is in meters and θ is in degrees
	# ϕ is steering angle input (deg)
	
	# Dynamics model
	ẋ = v * sind(θ)
	θ̇ = (v / L) * tand(ϕ)
	
	x′ = x + ẋ * dt
	θ′ = θ + rad2deg(θ̇) * dt
	
	return x′, θ′
end

function next_state(x, θ, y, ϕ; dt = 0.05)
    # for plotting purposes, we might also want the y value
    # x is in meters and θ is in degrees
	# ϕ is steering angle input (deg)
	
	# Dynamics model
	ẋ = v * sind(θ)
	ẏ = v * cosd(θ)
	θ̇ = (v / L) * tand(ϕ)
	
	x′ = x + ẋ * dt
	y′ = y + ẏ * dt
	θ′ = θ + rad2deg(θ̇) * dt
	
	return x′, θ′, y′
end

function sim_steps(x, θ, ϕ; dt = 0.05, num_steps = 20)
    for i = 1:num_steps
		x, θ = next_state(x, θ, ϕ)
	end
    return x, θ
end

# NOTE: this function is only valid for θs between -90 and 90 degrees
function reachable_cell(lbs, ubs, ϕ_min, ϕ_max)
	next_lb_x, next_lb_θ = sim_steps(lbs[1], lbs[2], ϕ_min)
	next_ub_x, next_ub_θ = sim_steps(ubs[1], ubs[2], ϕ_max)
	return [next_lb_x, next_lb_θ], [next_ub_x, next_ub_θ]
end