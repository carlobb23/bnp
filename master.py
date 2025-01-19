import gurobipy as gu
import math

class Subproblem:
    def __init__(self, duals_i, duals_ts, df, iteration, branch_duals_left, branch_duals_right):
        itr = iteration + 1
        self.days = df['T'].dropna().astype(int).unique().tolist()
        self.shifts = df['K'].dropna().astype(int).unique().tolist()
        self.duals_i = duals_i
        self.duals_ts = duals_ts
        self.duals_branch_left = branch_duals_left
        self.duals_branch_right = branch_duals_right
        self.model = gu.Model("Subproblem")
        self.itr = itr
        self.F_S = [(3, 1), (3, 2), (2, 1)]
        self.Min_WD = 3
        self.Max_WD = 5
        self.M = 100
        self.alpha = {1: 0.87, 2: 0.83, 3: 0.92, 4: 0.91, 5: 0.88, 6: 1, 7: 0.93, 8: 0.91, 9: 0.93, 10: 0.84, 11: 0.82, 12: 0.92, 13: 1, 14: 0.95}
        self.branch_constraints = []
        self.branch_constraints_mp = []


    def buildModel(self):
        self.generateVariables()
        self.generateConstraints()
        self.generateObjective()
        self.model.update()

    def generateVariables(self):
        self.x = self.model.addVars(self.days, self.shifts, vtype=gu.GRB.BINARY, name="x")
        self.y = self.model.addVars(self.days, vtype=gu.GRB.BINARY, name="y")
        self.mood = self.model.addVars(self.days, vtype=gu.GRB.BINARY, name="mood")
        self.moti = self.model.addVars(self.days, self.shifts, [self.itr], vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name="moti")

    def generateConstraints(self):
        for t in self.days:
            self.model.addLConstr(gu.quicksum(self.x[t, k] for k in self.shifts) == self.y[t], name = f"day_assignment_{t}")
            for s in self.shifts:
                self.model.addConstr(self.alpha[t] * gu.quicksum(self.x[t, s] for s in self.shifts)  == self.moti[t, s, self.itr], name="Mood_constraint")
        for k1, k2 in self.F_S:
            for t in range(1, len(self.days)):
                self.model.addLConstr(self.x[t, k1] + self.x[t + 1, k2] <= 1, name = f"rotation_{t}")
        for t in range(1, len(self.days) - self.Max_WD + 1):
            self.model.addLConstr(gu.quicksum(self.y[u] for u in range(t, t + 1 + self.Max_WD)) <= self.Max_WD, name = f"max_days_{t}")
        for t in range(2, len(self.days) - self.Min_WD + 1):
            self.model.addLConstr(gu.quicksum(self.y[u] for u in range(t + 1, t + self.Min_WD + 1)) >= self.Min_WD * (self.y[t + 1] - self.y[t]), name = f"min_days_{t}")
        self.model.update()

    def generateObjective(self):
        self.model.setObjective(0 - gu.quicksum(self.moti[t, s, self.itr] * self.duals_ts[t, s] for t in self.days for s in self.shifts) - self.duals_i + self.duals_branch_left - self.duals_branch_right, sense=gu.GRB.MINIMIZE)

    def solveSP(self):
        self.model.Params.OutputFlag = 1
        self.model.optimize()
        if self.model.status != gu.GRB.OPTIMAL:
            self.model.computeIIS()
            print('\nThe following constraints and variables are in the IIS:')
            for c in self.model.getConstrs():
                if c.IISConstr: print(f'\t{c.constrname}: {self.model.getRow(c)} {c.Sense} {c.RHS}')

            for v in self.model.getVars():
                if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
                if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')
            else:
                None


    def add_branch_constraint(self, day, shift, direction):
        """Adds branbetang constraint to subproblem."""

        if direction == 'left':
            self.model.addLConstr(0 == self.x[day, shift], name=f"sp_branch_left_{day}_{shift}")
        else:
            self.model.addConstr(1 == self.x[day, shift], name=f"sp_branch_right_{day}_{shift}")

        self.branch_constraints.append({
            'day': day,
            'shift': shift,
            'direction': direction
        })

        self.model.update()

    def propagate_constraints(self, parent_subproblem):
        """Propagates all branbetang constraints from parent"""
        for constraint_info in parent_subproblem.branch_constraints:
            self.add_branch_constraint(
                constraint_info['day'],
                constraint_info['shift'],
                constraint_info['direction']
            )
