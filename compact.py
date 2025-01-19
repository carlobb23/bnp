import gurobipy as gu
import math
import time

class Problem:
    def __init__(self, dfData, Cover):
        self.I = dfData['I'].dropna().astype(int).unique().tolist()
        self.T = dfData['T'].dropna().astype(int).unique().tolist()
        self.K = dfData['K'].dropna().astype(int).unique().tolist()
        self.cover = Cover
        self.model = gu.Model("MasterProblem")
        self.Min_WD = 3
        self.Max_WD = 5
        self.alpha = {1: 0.87, 2: 0.83, 3: 1, 4: 0.91, 5: 0.88, 6: 1, 7: 1, 8: 0.91, 9: 1, 10: 0.84, 11: 0.82, 12: 0.92, 13: 1, 14: 0.95}
        self.M = 100
        self.F_S = [(3, 1), (3, 2), (2, 1)]

    def buildLinModel(self):
        self.t0 = time.time()
        self.generateVariables()
        self.genGenCons()
        self.generateObjective()
        self.model.update()


    def generateVariables(self):
        self.x = self.model.addVars(self.I, self.T, self.K, vtype=gu.GRB.BINARY, name="x")
        self.y = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="y")
        self.u = self.model.addVars(self.T, self.K, lb = 0, vtype=gu.GRB.CONTINUOUS, name="u")
        self.moti = self.model.addVars(self.I, self.T, self.K, vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name="perf")
        self.mood = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="mood")


    def genGenCons(self):
        for i in self.I:
            for t in range(1, len(self.T) - self.Max_WD + 1):
                self.model.addLConstr(
                    gu.quicksum(self.y[i, u] for u in range(t, t + 1 + self.Max_WD)) <= self.Max_WD)
            for t in range(2, len(self.T) - self.Min_WD + 1):
                self.model.addLConstr(
                    gu.quicksum(self.y[i, u] for u in range(t + 1, t + self.Min_WD + 1)) >= self.Min_WD * (
                                self.y[i, t + 1] - self.y[i, t]))
            for k1, k2 in self.F_S:
                for t in range(1, len(self.T)):
                    self.model.addLConstr(self.x[i, t, k1] + self.x[i, t + 1, k2] <= 1)
            for t in self.T:
                self.model.addLConstr(gu.quicksum(self.x[i, t, k] for k in self.K) == self.y[i, t])
                for k in self.K:
                    self.model.addLConstr(self.moti[i, t, k] == self.alpha[t] * self.x[i, t, k], name=f"moti_{t}_{k}")
        for t in self.T:
            for k in self.K:
                self.model.addLConstr(
                    gu.quicksum(self.x[i, t, k] for i in self.I) + self.u[t, k] >= self.cover[t, k])
        #for i in self.I:
            #for t in self.T:
                #self.model.addConstr(self.alpha[t] * gu.quicksum(self.x[i, t, k] for k in self.K) + self.mood[i, t] == 1, name="Mood_constraint")
                #for k in self.K:
                    #self.model.addConstr(-self.M * (1 - self.x[i, t, k]) <= self.moti[i, t, k] - self.mood[i, t], name="Motivation_lower_bound")
                    #self.model.addConstr(self.moti[i, t, k] - self.mood[i, t] <= self.M * (1 - self.x[i, t, k]), name="Motivation_upper_bound")
                    #self.model.addConstr(self.moti[i, t, k] <= self.x[i, t, k], name="Motivation_upper_limit")
        self.model.update()

    def generateObjective(self):
        self.model.setObjective(gu.quicksum(self.u[t, k] for k in self.K for t in self.T), sense=gu.GRB.MINIMIZE)

    def solveModel(self):
        self.model.optimize()
