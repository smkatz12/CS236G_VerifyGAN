include("verify.jl")

network = read_nnet("../../models/full_big_normal_v2.nnet")

num_inp = size(network.layers[1].weights, 2)
strategy = MIPVerify.mip
timeout_per_node = 0.5
main_timeout = 10.0
mipverify_network = network_to_mipverify_network(network, "test", strategy)

main_solver = GurobiSolver(Gurobi.Env(), OutputFlag = 0, TimeLimit=main_timeout)
tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag = 0, TimeLimit=timeout_per_node)

function get_control_optima(mipverify_network, num_inp, lbs, ubs, main_solver, tightening_solver)
    p1 = get_optimization_problem(
          (num_inp,),
          mipverify_network,
          main_solver,
          lower_bounds=lbs,
          upper_bounds=ubs,
          tightening_solver=tightening_solver
          )

    @objective(p1.model, Max, -0.74p1.output_variable[1] - 0.44p1.output_variable[2])
    solve(p1.model)
    max_control = getobjectivevalue(p1.model)

    @objective(p1.model, Min, -0.74p1.output_variable[1] - 0.44p1.output_variable[2])
    solve(p1.model)
    min_control = getobjectivevalue(p1.model)

    return min_control, max_control
end

min_cte = -0.1
max_cte = 0.1

min_he = -0.1
max_he = 0.1

lbs = [-1.0, -1.0, min_cte / 6.366468343804353, min_he / 17.248858791583547]
ubs = [1.0, 1.0, max_cte / 6.366468343804353, max_he / 17.248858791583547]

#min_control, max_control = get_control_optima(mipverify_network, num_inp, lbs, ubs, main_solver, tightening_solver)