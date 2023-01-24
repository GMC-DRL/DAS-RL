import time

import numpy as np
import scipy.stats as stats
from env.cec_test_func import *


class JADE:
    def __init__(self):
        self.uCR = 0.5
        self.uF = 0.5
        self.p = 0.05
        self.c = 0.1
        self.archive = np.array([])

    def step(self, population, cost, problem):
        NP, dim = population.shape
        cr = np.random.normal(loc=self.uCR, scale=0.1, size=NP)
        cr = np.clip(cr, 0, 1)
        f = stats.cauchy.rvs(loc=self.uF, scale=0.1, size=NP)
        err = np.where(f < 0)[0]
        f[err] = 2 * self.uF - f[err]
        f[f > 1] = 1

        idx = np.argsort(cost)
        sorted_pop = population[idx]

        pb = max(int(NP * self.p), 1)
        rb = np.random.randint(pb, size=NP)

        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where(r1 == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where(r1 == np.arange(NP))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP + len(self.archive), size=NP)
        duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r2[duplicate] = np.random.randint(NP + len(self.archive), size=duplicate.shape[0])
            duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        xb = sorted_pop[rb]
        x1 = population[r1]
        if len(self.archive) > 0:
            x2 = np.concatenate((population, self.archive), 0)[r2]
        else:
            x2 = population[r2]

        fs = f.repeat(dim).reshape(NP, dim)
        v = population + fs * (xb - population) + fs * (x1 - x2)

        crs = cr.repeat(dim).reshape(NP, dim)
        jrand = np.random.randint(dim, size=NP)
        u = np.where(np.random.rand(NP, dim) < crs, v, population)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        u = np.clip(u, -100, 100)

        new_cost = problem.func(u)

        replace = np.where(new_cost < cost)[0]

        if len(self.archive) >= NP:
            self.archive[np.random.randint(NP, size=len(replace))] = population[replace]
        else:
            self.archive = np.append(self.archive, population[replace]).reshape(-1, dim)
        if len(self.archive) > NP:
            self.archive = self.archive[:NP]
        SF = f[replace]
        SCR = cr[replace]
        population[replace] = u[replace]
        cost[replace] = new_cost[replace]

        self.uCR = (1 - self.c) * self.uCR + self.c * np.mean(SCR)
        if np.sum(SF) < 1e-8:
            self.uF = 0.5
        else:
            self.uF = (1 - self.c) * self.uF + self.c * np.sum(SF ** 2) / (np.sum(SF))

        return population, cost


class CoDE:
    def __init__(self):
        self.pool = np.array([[1.0, 0.1], [1.0, 0.9], [0.8, 0.2]])

    def rand1bin(self, population):
        NP, dim = population.shape

        paras = self.pool[np.random.randint(3, size=NP)]
        F, Cr = paras[:, 0], paras[:, 1]
        Fs = F.repeat(dim).reshape(NP, dim)
        Crs = Cr.repeat(dim).reshape(NP, dim)

        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where(r1 == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where(r1 == np.arange(NP))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP, size=NP)
        duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r2[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        count = 0
        r3 = np.random.randint(NP, size=NP)
        duplicate = np.where((r3 == np.arange(NP)) + (r3 == r1) + (r3 == r2))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r3[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r3 == np.arange(NP)) + (r3 == r1) + (r3 == r2))[0]
            count += 1

        x1 = population[r1]
        x2 = population[r2]
        x3 = population[r3]
        v = x1 + Fs * (x2 - x3)

        jrand = np.random.randint(dim, size=NP)
        u = np.where(np.random.rand(NP, dim) < Crs, v, population)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        u = np.clip(u, -100, 100)

        return u

    def rand2bin(self, population):
        NP, dim = population.shape

        paras = self.pool[np.random.randint(3, size=NP)]
        F, Cr = paras[:, 0], paras[:, 1]
        Fs = F.repeat(dim).reshape(NP, dim)
        Crs = Cr.repeat(dim).reshape(NP, dim)

        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where(r1 == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where(r1 == np.arange(NP))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP, size=NP)
        duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r2[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        count = 0
        r3 = np.random.randint(NP, size=NP)
        duplicate = np.where((r3 == np.arange(NP)) + (r3 == r1) + (r3 == r2))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r3[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r3 == np.arange(NP)) + (r3 == r1) + (r3 == r2))[0]
            count += 1

        count = 0
        r4 = np.random.randint(NP, size=NP)
        duplicate = np.where((r4 == np.arange(NP)) + (r4 == r1) + (r4 == r2) + (r4 == r3))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r4[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r4 == np.arange(NP)) + (r4 == r1) + (r4 == r2) + (r4 == r3))[0]
            count += 1

        count = 0
        r5 = np.random.randint(NP, size=NP)
        duplicate = np.where((r5 == np.arange(NP)) + (r5 == r1) + (r5 == r2) + (r5 == r3) + (r5 == r4))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r5[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r5 == np.arange(NP)) + (r5 == r1) + (r5 == r2) + (r5 == r3) + (r5 == r4))[0]
            count += 1

        x1 = population[r1]
        x2 = population[r2]
        x3 = population[r3]
        x4 = population[r4]
        x5 = population[r5]
        v = x5 + Fs * (x1 - x2) + Fs * (x3 - x4)

        jrand = np.random.randint(dim, size=NP)
        u = np.where(np.random.rand(NP, dim) < Crs, v, population)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        u = np.clip(u, -100, 100)

        return u

    def current2rand(self, population):
        NP, dim = population.shape

        paras = self.pool[np.random.randint(3, size=NP)]
        F, Cr = paras[:, 0], paras[:, 1]
        Fs = F.repeat(dim).reshape(NP, dim)

        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where(r1 == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where(r1 == np.arange(NP))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP, size=NP)
        duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r2[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        count = 0
        r3 = np.random.randint(NP, size=NP)
        duplicate = np.where((r3 == np.arange(NP)) + (r3 == r1) + (r3 == r2))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r3[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r3 == np.arange(NP)) + (r3 == r1) + (r3 == r2))[0]
            count += 1

        x1 = population[r1]
        x2 = population[r2]
        x3 = population[r3]
        u = population + Fs * (x1 - population) + Fs * (x2 - x3)
        u = np.clip(u, -100, 100)

        return u

    def step(self, population, cost, problem):
        u1 = self.rand1bin(population)
        u2 = self.rand2bin(population)
        u3 = self.current2rand(population)
        cost1 = problem.func(u1)
        cost2 = problem.func(u2)
        cost3 = problem.func(u3)
        u1[cost2 < cost1] = u2[cost2 < cost1]
        new_cost = np.min([cost1, cost2], 0)
        u1[cost3 < new_cost] = u3[cost3 < new_cost]
        new_cost = np.min([new_cost, cost3], 0)
        population[new_cost < cost] = u1[new_cost < cost]
        cost = np.min([new_cost, cost], 0)

        return population, cost


class EPSDE:
    def __init__(self):
        self.Cr_pool = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        self.F_pool = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    def best2bin(self, population, best):
        NP, dim = population.shape

        F = np.random.choice(self.F_pool, size=NP)
        Cr = np.random.choice(self.Cr_pool, size=NP)
        Fs = F.repeat(dim).reshape(NP, dim)
        Crs = Cr.repeat(dim).reshape(NP, dim)

        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where(r1 == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where(r1 == np.arange(NP))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP, size=NP)
        duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r2[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        r3 = np.random.randint(NP, size=NP)
        duplicate = np.where((r3 == np.arange(NP)) + (r3 == r1) + (r3 == r2))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r3[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r3 == np.arange(NP)) + (r3 == r1) + (r3 == r2))[0]
            count += 1

        r4 = np.random.randint(NP, size=NP)
        duplicate = np.where((r4 == np.arange(NP)) + (r4 == r1) + (r4 == r2) + (r4 == r3))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r4[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r4 == np.arange(NP)) + (r4 == r1) + (r4 == r2) + (r4 == r3))[0]
            count += 1

        x1 = population[r1]
        x2 = population[r2]
        x3 = population[r3]
        x4 = population[r4]
        v = best + Fs * (x1 - x2) + Fs * (x3 - x4)

        jrand = np.random.randint(dim, size=NP)
        u = np.where(np.random.rand(NP, dim) < Crs, v, population)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        u = np.clip(u, -100, 100)

        return u

    def rand1bin(self, population):
        NP, dim = population.shape

        F = np.random.choice(self.F_pool, size=NP)
        Cr = np.random.choice(self.Cr_pool, size=NP)
        Fs = F.repeat(dim).reshape(NP, dim)
        Crs = Cr.repeat(dim).reshape(NP, dim)

        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where(r1 == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where(r1 == np.arange(NP))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP, size=NP)
        duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r2[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        count = 0
        r3 = np.random.randint(NP, size=NP)
        duplicate = np.where((r3 == np.arange(NP)) + (r3 == r1) + (r3 == r2))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r3[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r3 == np.arange(NP)) + (r3 == r1) + (r3 == r2))[0]
            count += 1

        x1 = population[r1]
        x2 = population[r2]
        x3 = population[r3]
        v = x1 + Fs * (x2 - x3)

        jrand = np.random.randint(dim, size=NP)
        u = np.where(np.random.rand(NP, dim) < Crs, v, population)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        u = np.clip(u, -100, 100)

        return u

    def current2rand1bin(self, population):
        NP, dim = population.shape

        F = np.random.choice(self.F_pool, size=NP)
        Cr = np.random.choice(self.Cr_pool, size=NP)
        Fs = F.repeat(dim).reshape(NP, dim)
        Crs = Cr.repeat(dim).reshape(NP, dim)

        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where(r1 == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where(r1 == np.arange(NP))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP, size=NP)
        duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r2[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        count = 0
        r3 = np.random.randint(NP, size=NP)
        duplicate = np.where((r3 == np.arange(NP)) + (r3 == r1) + (r3 == r2))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r3[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r3 == np.arange(NP)) + (r3 == r1) + (r3 == r2))[0]
            count += 1

        x1 = population[r1]
        x2 = population[r2]
        x3 = population[r3]
        v = population + Fs * (x1 - population) + Fs * (x2 - x3)

        jrand = np.random.randint(dim, size=NP)
        u = np.where(np.random.rand(NP, dim) < Crs, v, population)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        u = np.clip(u, -100, 100)

        return u

    def step(self, population, cost, problem):
        NP, dim = population.shape

        op_select = np.random.randint(3, size=NP)
        u = np.zeros((NP, dim))
        op1 = np.where(op_select == 0)[0]
        op2 = np.where(op_select == 1)[0]
        op3 = np.where(op_select == 2)[0]
        u1 = self.best2bin(population[op1], population[np.argmin(cost)])
        u2 = self.rand1bin(population[op2])
        u3 = self.current2rand1bin(population[op3])
        u[op1] = u1
        u[op2] = u2
        u[op3] = u3

        new_cost = problem.func(u)

        population[new_cost < cost] = u[new_cost < cost]
        cost = np.min([new_cost, cost], 0)

        return population, cost


class L_SHADE_EpSin:
    def __init__(self, dim):
        self.uF = 0.5
        self.uCr = 0.5
        self.uFreq = 0.5
        self.H = 5
        self.NPmax = 18 * dim
        self.NPmin = 4
        self.MF = np.ones(self.H) * 0.5
        self.MCr = np.ones(self.H) * 0.5
        self.MFreq = np.ones(self.H) * 0.5

    def current2best(self, population, Fs, best):
        NP, dim = population.shape

        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where(r1 == np.arange(NP))[0]
        while duplicate.shape[0] > 0:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where(r1 == np.arange(NP))[0]

        r2 = np.random.randint(NP, size=NP)
        duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0:
            r2[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]

        x1 = population[r1]
        x2 = population[r2]
        trail = population + Fs * (best - population) + Fs * (x1 - x2)

        return trail

    def mean_wL(self, df, s):
        if np.sum(df) > 1e-8:
            w = df / np.sum(df)
            return np.sum(w * (s ** 2)) / np.sum(w * s)
        else:
            return 0.5

    def local_search(self, population, cost, problem):
        NP, dim = population.shape
        x = np.random.rand(10, dim) * 200 - 100
        cx = problem.func(x)
        fes = 10
        xb = x[np.argmin(cx)]
        for i in range(1, 250 + 1):
            sigma = np.fabs(np.log(i) / i * (x - xb))
            y = np.random.normal(xb, sigma) + (np.random.rand(10) * xb - np.random.rand(10).repeat(dim).reshape(10, dim) * x)
            cy = problem.func(y)
            x[cy < cx] = y[cy < cx]
            cx = np.minimum(cy, cx)
            xb = x[np.argmin(cx)]
            fes += 10
        x = np.concatenate((population[-10:], x), 0)
        c = np.concatenate((cost[-10:], cx), 0)
        order = np.argsort(c)
        c = c[order]
        x = x[order]
        population[-10:] = x[:10]
        cost[-10:] = c[:10]
        return population, cost, fes

    def step(self, population, cost, problem, FEs, MaxFEs, g, Gmax):
        NP, dim = population.shape

        freq = None
        if FEs <= 0.5 * MaxFEs:
            c = np.random.rand(NP)
            r = np.random.randint(self.H, size=NP)
            freq = stats.cauchy.rvs(loc=self.MFreq[r], scale=0.1, size=NP)
            F1 = 0.5 * np.sin(2 * np.pi * 0.5 * g) * (Gmax - g) / Gmax + 0.5
            F2 = 0.5 * np.sin(2 * np.pi * freq * g) * g / Gmax + 0.5
            F = np.zeros(NP)
            F[c < 0.5] = F1
            F[c >= 0.5] = F2[c >= 0.5]
            uCr = np.random.choice(self.MFreq, size=NP)
            Cr = np.random.normal(uCr, 0.1, size=NP)
        else:
            r = np.random.randint(self.H, size=NP)
            F = stats.cauchy.rvs(loc=self.MF[r], scale=0.1, size=NP)
            Cr = np.random.normal(self.MCr[r], 0.1, size=NP)
        Fs = F.repeat(dim).reshape(NP, dim)
        Crs = Cr.repeat(dim).reshape(NP, dim)
        p = np.random.rand(NP) * 0.1 * NP
        pb = np.array(np.random.rand(NP) * p, dtype=np.int64)
        best = population[pb]
        v = self.current2best(population, Fs, best)

        jrand = np.random.randint(dim, size=NP)
        u = np.where(np.random.rand(NP, dim) < Crs, v, population)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]

        new_cost = problem.func(u)

        replace = np.where(new_cost < cost)[0]
        df = cost[replace] - new_cost[replace]
        if freq is not None:
            SFreq = np.zeros(self.H)
            SFreq2 = np.zeros(self.H)
            for i in range(replace.shape[0]):
                SFreq[r[replace[i]]] += df[i] * freq[replace[i]]
                SFreq2[r[replace[i]]] += df[i] * freq[replace[i]] ** 2
            self.MFreq[SFreq > 0] = SFreq2[SFreq > 0] / SFreq[SFreq > 0]
        else:
            SF = np.zeros(self.H)
            SCr = np.zeros(self.H)
            SF2 = np.zeros(self.H)
            SCr2 = np.zeros(self.H)
            for i in range(replace.shape[0]):
                SF[r[replace[i]]] += df[i] * F[replace[i]]
                SCr[r[replace[i]]] += df[i] * Cr[replace[i]]
                SF2[r[replace[i]]] += df[i] * F[replace[i]] ** 2
                SCr2[r[replace[i]]] += df[i] * Cr[replace[i]] ** 2
            self.MF[SF > 0] = SF2[SF > 0] / SF[SF > 0]
            self.MCr[SCr > 0] = SCr2[SCr > 0] / SCr[SCr > 0]
        population[replace] = u[replace]
        cost[replace] = new_cost[replace]

        # LPSR
        nNP = int((self.NPmin - self.NPmax) / MaxFEs * FEs + self.NPmax)
        order = np.argsort(cost)
        cost = cost[order]
        population = population[order]
        population = population[:nNP]
        cost = cost[:nNP]

        fes = 0
        if nNP <= 20 and NP > 20:
            population, cost, fes = self.local_search(population, cost, problem)

        return population, cost, fes + NP


class L_SHADE_RSP:
    def __init__(self, dim):
        self.uF = 0.3
        self.uCr = 0.8
        self.H = 5
        self.MF = np.ones(self.H) * 0.3
        self.MCr = np.ones(self.H) * 0.8
        self.MF[-1] = 0.9
        self.MCr[-1] = 0.9
        self.archive = np.array([])
        self.archive_cost = np.array([])
        self.NPmin = 4
        self.NPmax = 75 * np.power(dim, 2 / 3)

    def step(self, population, cost, problem, FEs, MaxFEs, g, Gmax):
        NP, dim = population.shape

        r = np.random.randint(self.H, size=NP)
        F = stats.cauchy.rvs(loc=self.MF[r], scale=0.1, size=NP)
        Cr = np.random.normal(self.MCr[r], 0.1, size=NP)
        Cr[self.MCr[r] < 0] = 0
        if FEs < 0.25 * MaxFEs:
            Cr = np.maximum(self.MCr[r], 0.7)
        if FEs < 0.5 * MaxFEs:
            Cr = np.maximum(self.MCr[r], 0.6)
        Fs = F.repeat(dim).reshape(NP, dim)
        Crs = Cr.repeat(dim).reshape(NP, dim)

        if FEs < 0.2 * MaxFEs:
            w = 0.7
        elif 0.2 * MaxFEs <= FEs < 0.4 * MaxFEs:
            w = 0.8
        else:
            w = 1.2
        pb = 0.085 + 0.085 * FEs / MaxFEs
        PB = max(int(NP * pb), 1)
        rb = np.random.randint(PB, size=NP)
        order = np.argsort(cost)
        rank1 = np.zeros(NP)
        rank1[order] = 3 * (NP - np.arange(NP)) + 1
        r1 = np.random.choice(np.arange(NP), size=NP, p=rank1/np.sum(rank1))
        if self.archive.shape[0] > 0:
            PA = np.concatenate((population, self.archive), 0)
            PAc = np.concatenate((cost, self.archive_cost), 0)
        else:
            PA = population
            PAc = cost
        order = np.argsort(PAc)
        rank2 = np.zeros(NP + self.archive.shape[0])
        rank2[order] = 3 * (NP + self.archive.shape[0] - np.arange(NP + self.archive.shape[0])) + 1
        r2 = np.random.choice(np.arange(NP + self.archive.shape[0]), size=NP, p=rank2/np.sum(rank2))

        xb = population[rb]
        x1 = population[r1]
        x2 = PA[r2]
        v = population + Fs * w * (xb - population) + Fs * (x1 - x2)

        jrand = np.random.randint(dim, size=NP)
        u = np.where(np.random.rand(NP, dim) < Crs, v, population)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]

        new_cost = problem.func(u)

        replace = np.where(new_cost < cost)[0]

        if len(self.archive) >= NP:
            i = np.random.randint(NP, size=len(replace))
            self.archive[i] = population[replace]
            self.archive_cost[i] = cost[replace]
        else:
            self.archive = np.append(self.archive, population[replace]).reshape(-1, dim)
            self.archive_cost = np.append(self.archive_cost, cost[replace]).reshape(-1)
        if len(self.archive) > NP:
            self.archive = self.archive[:NP]
            self.archive_cost = self.archive_cost[:NP]

        df = cost[replace] - new_cost[replace]
        SF = np.zeros(self.H)
        SCr = np.zeros(self.H)
        SF2 = np.zeros(self.H)
        SCr2 = np.zeros(self.H)
        for i in range(replace.shape[0]):
            SF[r[replace[i]]] += df[i] * F[replace[i]]
            SCr[r[replace[i]]] += df[i] * Cr[replace[i]]
            SF2[r[replace[i]]] += df[i] * F[replace[i]] ** 2
            SCr2[r[replace[i]]] += df[i] * Cr[replace[i]] ** 2
        for i in range(self.H - 1):
            if SF[i] > 0:
                self.MF[i] = (self.MF[i] + SF2[i] / SF[i]) / 2
            if SCr[i] > 0:
                self.MCr[i] = (self.MCr[i] + SCr2[i] / SCr[i]) / 2

        # LPSR
        nNP = int((self.NPmin - self.NPmax) / MaxFEs * FEs + self.NPmax)
        order = np.argsort(cost)
        cost = cost[order]
        population = population[order]
        population = population[:nNP]
        cost = cost[:nNP]
        self.archive = self.archive[:int(nNP * 1.4)]
        self.archive_cost = self.archive_cost[:int(nNP * 1.4)]

        return population, cost, NP


class EDEV:
    def __init__(self, dim):
        self.algo = [JADE(), CoDE(), EPSDE()]
        self.dim = dim
        self.ng = 20
        self.NP = 60
        self.lamda = np.array([0.1, 0.1, 0.1, 0.7])

    def divide_pop(self, population, cost, k):
        pop = []
        costs = []
        p = 0
        for i in range(len(self.algo)):
            pop.append(population[p:p + int(self.lamda[i] * self.NP)])
            costs.append(cost[p:p + int(self.lamda[i] * self.NP)])
            p += int(self.lamda[i] * self.NP)
        pop[k] = np.concatenate((pop[k], population[int(-self.lamda[-1] * self.NP):]), 0)
        costs[k] = np.concatenate((costs[k], cost[int(-self.lamda[-1] * self.NP):]), 0)
        NP_divide = self.NP * self.lamda[:len(self.algo)]
        NP_divide[k] += int(self.NP * self.lamda[-1])
        return pop, costs, NP_divide

    def run(self, problem, MaxFEs, record_period):
        population = np.random.rand(self.NP, self.dim) * 200 - 100
        cost = problem.func(population)
        FEs = self.NP
        record = record_period
        fevs = []
        factor = np.min(cost)
        k = np.random.randint(len(self.algo))
        pop, costs, divide = self.divide_pop(population, cost, k)
        df = np.zeros(len(self.algo))
        g = 0
        while FEs < MaxFEs:
            g += 1
            npop = []
            ncosts = []
            for i in range(len(self.algo)):
                p, c = self.algo[i].step(pop[i], costs[i], problem)
                npop.append(p)
                ncosts.append(c)
                df[i] += np.min(costs[i]) - np.min(c)
            population = np.concatenate(npop, 0)
            cost = np.concatenate(ncosts, 0)
            if g % self.ng == 0:
                k = np.argmax(df / self.ng / divide)
                df = np.zeros(len(self.algo))
            pop, costs, divide = self.divide_pop(population, cost, k)
            FEs += self.NP
            if FEs >= record:
                fevs.append(1 - np.min(cost) / factor)
                record += record_period
            if np.min(cost) < 1e-8:
                break
        while len(fevs) < MaxFEs // record_period:
            fevs.append(1.0)
        return fevs


class EPSDE_cec:
    def __init__(self, dim):
        self.dim = dim
        self.NP = 50
        self.Cr_pool = np.array([0.1, 0.5, 0.9])
        self.F_pool = np.array([0.5, 0.9])
        self.archive = np.array([])
        self.p = 0.05

    def current2rand(self, population, F):
        NP, dim = population.shape

        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where(r1 == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where(r1 == np.arange(NP))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP, size=NP)
        duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r2[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        count = 0
        r3 = np.random.randint(NP, size=NP)
        duplicate = np.where((r3 == np.arange(NP)) + (r3 == r1) + (r3 == r2))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r3[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r3 == np.arange(NP)) + (r3 == r1) + (r3 == r2))[0]
            count += 1

        x1 = population[r1]
        x2 = population[r2]
        x3 = population[r3]
        Fs = F.repeat(dim).reshape(NP, dim)
        trail = population + Fs * (x1 - population) + Fs * (x2 - x3)

        return trail

    def jade_mu(self, population, cost, F):
        NP, dim = population.shape

        idx = np.argsort(cost)
        sorted_pop = population[idx]

        pb = max(int(NP * self.p), 1)
        rb = np.random.randint(pb, size=NP)

        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where(r1 == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where(r1 == np.arange(NP))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP + len(self.archive), size=NP)
        duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 10:
            r2[duplicate] = np.random.randint(NP + len(self.archive), size=duplicate.shape[0])
            duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        xb = sorted_pop[rb]
        x1 = population[r1]
        if self.archive.shape[0] > 0:
            x2 = np.concatenate((population, self.archive), 0)[r2]
        else:
            x2 = population[r2]
        Fs = F.repeat(dim).reshape(NP, dim)
        v = population + Fs * (xb - population) + Fs * (x1 - x2)

        return v

    def binomial(self, population, trail, Cr):
        NP, dim = population.shape
        Crs = Cr.repeat(dim).reshape(NP, dim)
        jrand = np.random.randint(dim, size=NP)
        u = np.where(np.random.rand(NP, dim) < Crs, trail, population)
        u[np.arange(NP), jrand] = trail[np.arange(NP), jrand]
        u = np.clip(u, -100, 100)
        return u

    def exponential(self, population, trail, Cr):
        NP, dim = population.shape
        Crs = Cr.repeat(dim).reshape(NP, dim)
        u = population.copy()
        L = np.random.randint(dim, size=NP).repeat(dim).reshape(NP, dim)
        L = L <= np.arange(dim)
        rvs = np.random.rand(NP, dim)
        L = np.where(rvs > Crs, L, 0)
        u = u * (1 - L) + trail * L
        u = np.clip(u, -100, 100)
        return u

    def run(self, problem, MaxFEs, record_period):
        population = np.random.rand(self.NP, self.dim) * 200 - 100
        cost = problem.func(population)
        FEs = self.NP
        record = record_period
        fevs = []
        factor = np.min(cost)
        F = np.random.choice(self.F_pool, size=self.NP)
        Cr = np.random.choice(self.Cr_pool, size=self.NP)
        mu_op = np.random.randint(2, size=self.NP)
        cr_op = np.random.randint(2, size=self.NP)
        while FEs < MaxFEs:
            v = np.zeros((self.NP, self.dim))
            v1 = self.jade_mu(population[mu_op == 0], cost[mu_op == 0], F[mu_op == 0])
            v2 = self.current2rand(population[mu_op == 1], F[mu_op == 1])
            v[mu_op == 0] = v1
            v[mu_op == 1] = v2

            u = np.zeros((self.NP, self.dim))
            u1 = self.binomial(population[cr_op == 0], v[cr_op == 0], Cr[cr_op == 0])
            u2 = self.exponential(population[cr_op == 1], v[cr_op == 1], Cr[cr_op == 1])
            u[cr_op == 0] = u1
            u[cr_op == 1] = u2

            ucost = problem.func(u)

            replaced = np.where(ucost < cost)[0]
            if len(self.archive) >= self.NP:
                self.archive[np.random.randint(self.NP, size=len(replaced))] = population[replaced]
            else:
                self.archive = np.append(self.archive, population[replaced]).reshape(-1, dim)
            if len(self.archive) > self.NP:
                self.archive = self.archive[:self.NP]
            population[replaced] = u[replaced]
            cost[replaced] = ucost[replaced]

            F[replaced] = np.random.choice(self.F_pool, size=len(replaced))
            Cr[replaced] = np.random.choice(self.Cr_pool, size=len(replaced))
            mu_op[replaced] = np.random.randint(2, size=len(replaced))
            cr_op[replaced] = np.random.randint(2, size=len(replaced))

            FEs += self.NP
            if FEs >= record:
                fevs.append(1 - np.min(cost) / factor)
                record += record_period
            if np.min(cost) < 1e-8:
                break
        while len(fevs) < MaxFEs // record_period:
            fevs.append(1.0)
        return fevs


class L_SHADE_E:
    def __init__(self, dim):
        self.dim = dim
        self.NP = dim * 18
        self.GN = int(dim * dim * 0.07143 + dim * 9.286 + 157.1)
        self.T = 0.1
        self.algo = [L_SHADE_EpSin(dim), L_SHADE_RSP(dim)]

    def run(self, problem, MaxFEs, record_period):
        population = np.random.rand(self.NP, self.dim) * 200 - 100
        cost = problem.func(population)
        Gmax = MaxFEs // self.NP
        FEs = self.NP
        record = record_period
        fevs = []
        factor = np.min(cost)
        fp = np.min(cost)
        algo = np.random.randint(2)
        g = 0
        gn = 0
        while FEs < MaxFEs:
            g += 1
            population, cost, fes = self.algo[algo].step(population, cost, problem, FEs, MaxFEs, g, Gmax)
            self.NP = population.shape[0]
            FEs += fes
            if (fp - np.min(cost)) / fp < self.T:
                gn += 1
            else:
                gn = 0
            fp = np.min(cost)
            if gn == self.GN:
                algo = 1 - algo
                gn = 0

            if FEs >= record:
                fevs.append(1 - np.min(cost) / factor)
                record += record_period
            if np.min(cost) < 1e-8:
                break
        while len(fevs) < MaxFEs // record_period:
            fevs.append(1.0)
        return fevs


if __name__ == '__main__':
    dim = 10
    MaxFEs = 200000
    period = 5000
    edev = EDEV(dim)
    epsde = EPSDE_cec(dim)
    lahadee = L_SHADE_E(dim)
    np.random.seed(1)
    np.set_printoptions(suppress=True)
    p = Schwefel(dim, np.random.rand(dim) * 160 - 80, rotate_gen(dim), 0)
    st = time.time()
    print(lahadee.run(p, MaxFEs, period))
    ed = time.time()
    print(ed - st)

