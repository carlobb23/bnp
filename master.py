import gurobipy as gu
import statistics
import numpy as np
import math


class MasterProblem:
    def __init__(self, df, Coverage, max_iteration, current_iteration, start):
        self.max_iteration = max_iteration
        self.nurses = df['I'].dropna().astype(int).unique().tolist()
        self.days = df['T'].dropna().astype(int).unique().tolist()
        self.shifts = df['K'].dropna().astype(int).unique().tolist()
        self._current_iteration = current_iteration
        self.roster = [i for i in range(1, self.max_iteration + 2)]
        self.cover = Coverage
        self.model = gu.Model("MasterProblem")
        self.cons_cover = {}
        self.cons_lmbda = {}
        self.cover_values = [self.cover[key] for key in self.cover.keys()]
        self.start = start
        self.all_schedules = {}
        self.branch_constraints = []  
        self.branch_constraints_mp = []  
        self.branch_history = {}  

    def buildModel(self):
        self.generateVariables()
        self.generateConstraints()
        self.generateObjective()
        self.model.update()

    def generateVariables(self):
        self.u = self.model.addVars(self.days, self.shifts, vtype=gu.GRB.CONTINUOUS, lb=0, name='u')
        self.lmbda = self.model.addVars(self.roster, vtype=gu.GRB.INTEGER, lb=0, name='lmbda')

    def generateConstraints(self):
        self.cons_lmbda = self.model.addLConstr(len(self.nurses) == gu.quicksum(self.lmbda[r] for r in self.roster), name="lmb")
        for t in self.days:
            for s in self.shifts:
                self.cons_cover[t, s] = self.model.addConstr(gu.quicksum(self.lmbda[r] for r in self.roster) + self.u[t, s] >= self.cover[t, s], "cover(" + str(t) + "," + str(s) + ")")
        return self.cons_lmbda, self.cons_cover

    def generateObjective(self):
        self.model.setObjective(gu.quicksum(self.u[t, s] for t in self.days for s in self.shifts), sense=gu.GRB.MINIMIZE)

    def getDuals(self):
        dual_sum_left = 0
        dual_sum_right = 0
        formula_parts_left = []
        formula_parts_right = []

        for constr in self.model.getConstrs():
            if 'left' in constr.ConstrName:
                value = constr.Pi
                formula_parts_left.append(f"{value} + ")
                print(f'Left {constr, value}')
                dual_sum_left += value*(-1)
            elif 'right' in constr.ConstrName:
                value = constr.Pi
                formula_parts_right.append(f"{value} - ")
                print(f'Right {constr, value}')
                dual_sum_right += value
            else:
                dual_sum_left, dual_sum_right = 0, 0
        print(f'Left, Right: {dual_sum_left, dual_sum_right}')
        formula = " ".join(formula_parts_left) + " - " + " ".join(formula_parts_right)
        print(f'Dual formula {formula}')
        dual_sum_total = dual_sum_left + dual_sum_right

        print(dual_sum_total)
        return {(d, s): self.cons_cover[d, s].Pi for d in self.days for s in self.shifts}, self.cons_lmbda.Pi, dual_sum_left, dual_sum_right

    def startSol(self, start_vals):
        for t in self.days:
            for s in self.shifts:
                if (t, s) in start_vals:
                    value_cons = start_vals[t, s]
                else:
                    value_cons = 0

                if (t, s) in start_vals:
                    self.model.chgCoeff(self.cons_cover[t, s], self.lmbda[1], value_cons)
        self.model.update()

    def initCoeffs(self):
        for t in self.days:
            for s in self.shifts:
                for r in self.roster:
                    self.model.chgCoeff(self.cons_cover[t, s], self.lmbda[r], 0)
        self.model.update()

    def addCol(self, itr, schedules_perf, index=None):
        for t in self.days:
            for s in self.shifts:
                if (t, s, itr + 1) in schedules_perf:
                    value_cons = schedules_perf[t, s, itr + 1]
                else:
                    value_cons = 0

                if (t, s, itr + 1) in schedules_perf:
                    self.model.chgCoeff(self.cons_cover[t, s], self.lmbda[itr + 1], value_cons)
        self.model.update()

    def finalSolve(self, timeLimit):
        self.model.setParam('TimeLimit', timeLimit)
        self.model.Params.OutputFlag = 1
        self.model.setAttr("vType", self.lmbda, gu.GRB.INTEGER)
        self.model.update()
        self.model.optimize()
        if self.model.status == gu.GRB.OPTIMAL:
            print("***** Optimal solution found *****")
        else:
            print("***** No optimal solution found *****")


    def solveModel(self, timeLimit):
        try:
            self.model.setParam('TimeLimit', timeLimit)
            self.model.Params.OutputFlag = 0
            self.model.Params.IntegralityFocus = 1
            self.model.Params.MIPGap = 1e-5
            self.model.setParam('ConcurrentMIP', 2)
            self.model.optimize()
        except gu.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))

    def solveRelaxModel(self):
        try:
            self.model.Params.OutputFlag = 0
            self.model.Params.MIPGap = 1e-6
            self.model.Params.Method = 2
            self.model.Params.Crossover = 0
            for v in self.model.getVars():
                v.setAttr('vtype', 'C')
                v.setAttr('lb', 0.0)
            self.model.optimize()
        except gu.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))

    def addSchedule(self, schedule):
        self.all_schedules.update(schedule)

    def exportSchedules(self):
        return self.all_schedules


    def addColsFromDict(self, all_schedules):
        for (t, s, it), value_cons in all_schedules.items():
            if (t, s) in self.cons_cover:
                self.model.chgCoeff(self.cons_cover[t, s], self.lmbda[it], value_cons)
        self.model.update()


    def add_branch_constraint(self, day, shift, direction, bound_value, schedules):
        """
        Adds a branching constraint to the model and tracks it in branch_constraints.

        Args:
            day (int): Der Tag der Constraint.
            shift (int): Die Schicht der Constraint.
            direction (str): 'left' für <= oder 'right' für >=.
            bound_value (float): Der Wert, der als Grenze genutzt wird.
            relevant (list): Liste der relevanten Variablen.
        """



        expr = gu.quicksum(self.lmbda[r] * schedules.get((day, shift, r), 0) for r in self.roster)


        if direction == 'left':
            constr = self.model.addConstr(expr <= math.floor(bound_value),
                                          name=f"branch_left_{day}_{shift}")
        else:
            constr = self.model.addConstr(expr >= math.ceil(bound_value),
                                          name=f"branch_right_{day}_{shift}")

        self.branch_constraints.append({
            'day': day,
            'shift': shift,
            'direction': direction,
            'bound': bound_value,
            'relevant': schedules,
            'constraint': constr
        })

        self.model.update()

    def propagate_constraints(self, parent_problem):
        """Propagates branching constraints from parent node"""
        for constraint_info in parent_problem.branch_constraints:
            self.add_branch_constraint(
                constraint_info['day'],
                constraint_info['shift'],
                constraint_info['direction'],
                constraint_info['bound'],
                constraint_info['relevant']
            )

        self.model.update()

    def add_branch_constraint_mp(self, ind, bound_value, direction):
        """
        Adds a branching constraint to the model and tracks it in branch_constraints.

        Args:
            day (int): Der Tag der Constraint.
            shift (int): Die Schicht der Constraint.
            direction (str): 'left' für <= oder 'right' für >=.
            bound_value (float): Der Wert, der als Grenze genutzt wird.
            relevant (list): Liste der relevanten Variablen.
        """


        # Neues Constraint hinzufügen
        if direction == 'left':
            constr = self.model.addConstr(-1 * self.lmbda[ind] >= -1 * math.floor(bound_value),
                                          name=f"branch_left_{ind}")
        else:
            constr = self.model.addConstr(self.lmbda[ind] >= math.ceil(bound_value),
                                          name=f"branch_right_{ind}")

        # Neues Constraint zu `branch_constraints` hinzufügen
        self.branch_constraints.append({
            'day': None,
            'shift': None,
            'direction': direction,
            'bound': bound_value,
            'relevant': None,
            'constraint': constr
        })

        self.model.update()

    def find_and_print_fractional_betas(self, schedule, lmb_vals):

        beta_values = {}
        for d in self.days:
            for s in self.shifts:
                beta_d_s = sum(schedule.get((d, s, k), 0) * lmb_vals.get((k), 0) for k in self.roster)
                beta_values[d, s] = beta_d_s
                print(f'Beta{d}{s} = {beta_d_s}')

        fractional_betas = {}
        for (d, s), beta in beta_values.items():
            if abs(beta - round(beta)) > 1e-6:
                fractional_betas[(d, s)] = beta

        if not fractional_betas:
            None
        else:
            for (d, s), beta in fractional_betas.items():
                fractional_part = abs(beta - round(beta))
                print(f"beta_{d}_{s} = {beta}, Fraktionsanteil = {fractional_part}")

        if not fractional_betas:
            most_fractional_beta = None
        else:
            most_fractional_beta = max(fractional_betas.items(), key=lambda x: abs(x[1] - round(x[1])))
            print(
                f"beta_{most_fractional_beta[0][0]}_{most_fractional_beta[0][1]} = {most_fractional_beta[1]}")

        k_values = []
        for (key_t, key_s, key_k), value in schedule.items():
            if key_t == most_fractional_beta[0][0] and key_s == most_fractional_beta[0][1] and value > 0:
                k_values.append(key_k)

        return most_fractional_beta[0][0], most_fractional_beta[0][1], most_fractional_beta[1], [k for k in k_values if lmb_vals.get(k, 0) > 0]

    def getLambdas(self):
        return self.model.getAttr("X", self.lmbda)
