from master import *
from sub import *
from compact import Problem
import pandas as pd

# **** Prerequisites ****
I, T, K = list(range(1,26)), list(range(1,15)), list(range(1,4))
data = pd.DataFrame({
    'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
    'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
    'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
})

# Parameter
time_Limit, max_itr, output_len, mue, threshold = (4, 100, 100, 1e-4, 5e-4)
demand_dict = {(1, 1): 9, (1, 2): 11, (1, 3): 4, (2, 1): 10, (2, 2): 9, (2, 3): 9, (3, 1): 8, (3, 2): 16, (3, 3): 2, (4, 1): 4, (4, 2): 21, (4, 3): 3, (5, 1): 2, (5, 2): 7, (5, 3): 13, (6, 1): 7, (6, 2): 5, (6, 3): 13, (7, 1): 14, (7, 2): 1, (7, 3): 5, (8, 1): 7, (8, 2): 12, (8, 3): 3, (9, 1): 3, (9, 2): 11, (9, 3): 8, (10, 1): 1, (10, 2): 13, (10, 3): 6, (11, 1): 19, (11, 2): 8, (11, 3): 4, (12, 1): 11, (12, 2): 4, (12, 3): 15, (13, 1): 6, (13, 2): 6, (13, 3): 12, (14, 1): 4, (14, 2): 10, (14, 3): 9}
dir, dir1 = 'right', 'right'

# **** Compact Solver ****
problem = Problem(data, demand_dict)
problem.buildLinModel()
problem.solveModel()

# **** Column Generation ****
modelImprovable, reached_max_itr = True, False

# Get Starting Solutions
problem_start = Problem(data, demand_dict)
problem_start.buildLinModel()
problem_start.model.Params.MIPGap = 0.8
problem_start.model.update()
problem_start.model.optimize()
start_values_moti = {(t, s): problem_start.x[1, t, s].x for t in T for s in K}
print(start_values_moti)
while True:
    # Initialize iterations
    itr = 0
    last_itr = 0

    # Create empty results lists
    master = MasterProblem(data, demand_dict, max_itr, itr, start_values_moti)
    master.buildModel()
    master.initCoeffs()
    print("*{:^{output_len}}*".format("Restricted Master Problem successfully built!", output_len=output_len))

    # Initialize and solve relaxed model
    master.startSol(start_values_moti)
    master.addSchedule({(t, s, 1): problem_start.moti[1, t, s].x for t in T for s in K})
    master.solveRelaxModel()

    while (modelImprovable) and itr < max_itr:
        # Start
        itr += 1

        # Solve RMP and get duals
        master.current_iteration = itr + 1
        master.solveRelaxModel()
        duals_ts, duals_i, duals_bl, duals_br = master.getDuals()

        # Solve SPs
        modelImprovable = False
        subproblem = Subproblem(duals_i, duals_ts, data, itr, duals_bl, duals_br)
        subproblem.buildModel()
        subproblem.solveSP()
        last_itr = itr + 1

        # Generate and add columns with reduced cost
        if subproblem.model.objval < -threshold:
            Schedules = subproblem.model.getAttr("X", subproblem.moti)
            master.addCol(itr, Schedules)
            master.addSchedule(Schedules)
            master.model.update()
            modelImprovable = True

        if not modelImprovable:
            day, shift, value, filtered_list = master.find_and_print_fractional_betas(master.exportSchedules(), master.getLambdas())
            print(f'Branching Info: {day, shift, value, filtered_list}')
            print("*{:^{output_len}}*".format("No more improvable columns found.", output_len=output_len))
            break

    if modelImprovable and itr == max_itr:
        print("*{:^{output_len}}*".format("More iterations needed. Increase max_itr and restart the process.", output_len=output_len))
        max_itr *= 2
    else:
        break

# Solve Master Problem with integrality restored
master.finalSolve(time_Limit)

print('First Branch - Left')

modelImprovable1, reached_max_itr1 = True, False
while True:
    # Initialize iterations
    itr1, last_itr1 = last_itr - 1, 0


    # Create empty results lists
    master1 = MasterProblem(data, demand_dict, max_itr, itr1, None)
    master1.buildModel()
    master1.initCoeffs()
    master1.addColsFromDict(master.exportSchedules())
    master1.addSchedule(master.exportSchedules())
    master1.propagate_constraints(master)
    master1.add_branch_constraint(day, shift, dir, value, master.exportSchedules())
    master1.model.write('master.lp')

    print("*{:^{output_len}}*".format("Restricted Master Problem successfully built!", output_len=output_len))

    # Initialize and solve relaxed model
    master1.solveRelaxModel()

    while (modelImprovable1) and itr1 < max_itr:
        # Start
        itr1 += 1

        # Solve RMP and get duals
        master1.current_iteration = itr1 + 1
        master1.solveRelaxModel()
        duals_ts1, duals_i1, duals_bl1, duals_br1 = master1.getDuals()

        # Solve SPs
        modelImprovable1 = False
        print("*{:^{output_len}}*".format(f"Begin Column Generation Iteration {itr1}", output_len=output_len))

        # Build SP
        subproblem1 = Subproblem(duals_i1, duals_ts1, data, itr1, duals_bl1, duals_br1)
        subproblem1.propagate_constraints(subproblem)
        subproblem1.buildModel()
        subproblem1.add_branch_constraint(day, shift, dir)
        subproblem1.solveSP()
        redCost1 = subproblem1.model.objval
        last_itr1 = itr1 + 1

        # Generate and add columns with reduced cost
        if redCost1 < -threshold:
            Schedules1 = subproblem1.model.getAttr("X", subproblem1.moti)
            master1.addCol(itr1, Schedules1)
            master1.addSchedule(Schedules1)
            master1.model.update()
            modelImprovable1 = True

        if not modelImprovable1:
            day1, shift1, value1, filtered_list1 = master1.find_and_print_fractional_betas(master1.exportSchedules(), master1.getLambdas())
            print(f'Branching Info1: {day1, shift1, value1, filtered_list1}')
            print("*{:^{output_len}}*".format("No more improvable columns found.", output_len=output_len))
            break

    if modelImprovable1 and itr1 == max_itr:
        break
    else:
        break

# Solve Master Problem with integrality restored
master1.finalSolve(time_Limit)

print('Second Branch - Left')

modelImprovable2, reached_max_itr2 = True, False
while True:
    # Initialize iterations
    itr2, last_itr2 = last_itr1 - 1, 0

    # Create empty results lists
    master2 = MasterProblem(data, demand_dict, max_itr, itr2, None)
    master2.buildModel()
    master2.initCoeffs()
    master2.addColsFromDict(master1.exportSchedules())
    master2.addSchedule(master1.exportSchedules())
    master2.propagate_constraints(master1)
    master2.add_branch_constraint(day1, shift1, dir1, value1, master1.exportSchedules())

    print("*{:^{output_len}}*".format("Restricted Master Problem successfully built!", output_len=output_len))

    # Initialize and solve relaxed model
    master2.solveRelaxModel()

    # Retrieve dual values

    while (modelImprovable2) and itr2 < max_itr:
        # Start
        itr2 += 1

        # Solve RMP and get duals
        master2.current_iteration = itr2 + 1
        master2.solveRelaxModel()
        duals_ts2, duals_i2, duals_bl2, duals_br2 = master2.getDuals()


        # Solve SPs
        modelImprovable2 = False
        print("*{:^{output_len}}*".format(f"Begin Column Generation Iteration {itr2}", output_len=output_len))
        subproblem2 = Subproblem(duals_i2, duals_ts2, data, itr2, duals_bl2, duals_br2)
        subproblem2.buildModel()
        subproblem2.propagate_constraints(subproblem1)
        subproblem2.add_branch_constraint(day1, shift1, dir1)
        subproblem2.model.update()

        # Save time to solve SP
        subproblem2.solveSP()
        last_itr2 = itr2 + 1

        # Generate and add columns with reduced cost
        if subproblem2.model.objval < -threshold:
            Schedules2 = subproblem2.model.getAttr("X", subproblem2.moti)
            master2.addCol(itr2, Schedules2)
            print(f'New columdn in {itr2}: {Schedules2}')
            master2.addSchedule(Schedules2)
            master2.model.update()
            modelImprovable2 = True

        if not modelImprovable2:
            day2, shift2, value2, filtered_list2 = master2.find_and_print_fractional_betas(master2.exportSchedules(), master2.getLambdas())
            print(f'Branching Info2: {day2, shift2, value2, filtered_list2}')
            print("*{:^{output_len}}*".format("No more improvable columns found.", output_len=output_len))
            break

    if modelImprovable2 and itr2 == max_itr:
        break
    else:
        break

# Solve Master Problem with integrality restored
master2.finalSolve(time_Limit)
