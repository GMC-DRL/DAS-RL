import numpy as np
import scipy.stats as stats


class Population:
    def __init__(self, dim):
        self.Nmax = 170                         # the upperbound of population size
        self.Nmin = 30                          # the lowerbound of population size
        self.NP = self.Nmax                     # the population size
        self.NA = int(self.NP * 2.1)            # the size of archive(collection of replaced individuals)
        self.dim = dim                          # the dimension of individuals
        self.cost = np.zeros(self.NP)           # the cost of individuals
        self.cbest = 1e15                       # the best cost in current population, initialize as 1e15
        self.cbest_id = -1                      # the index of individual with the best cost
        self.gbest = 1e15                       # the global best cost
        self.gbest_solution = np.zeros(dim)     # the individual with global best cost
        self.Xmin = np.ones(dim) * -100         # the upperbound of individual value
        self.Xmax = np.ones(dim) * 100          # the lowerbound of individual value
        self.group = self.initialize_group()    # the population
        self.archive = np.array([])             # the archive(collection of replaced individuals)
        self.MF = np.ones(dim * 20) * 0.2       # the set of step length of DE
        self.MCr = np.ones(dim * 20) * 0.2      # the set of crossover rate of DE
        self.k = 0                              # the index of updating element in MF and MCr

    # generate an initialized population with size(default self population size)
    def initialize_group(self, size=-1):
        if size < 0:
            size = self.NP
        return np.random.random((size, self.dim)) * (self.Xmax - self.Xmin) + self.Xmin

    # initialize cost
    def initialize_costs(self, problem):
        self.cost = problem.func(self.group)
        self.gbest = self.cbest = np.min(self.cost)
        self.cbest_id = np.argmin(self.cost)
        self.gbest_solution = self.group[self.cbest_id]

    # sort former 'size' population in respect to cost
    def sort(self, size, reverse=False):
        # new index after sorting
        r = -1 if reverse else 1
        ind = np.concatenate((np.argsort(r * self.cost[:size]), np.arange(self.NP)[size:]))
        self.cost = self.cost[ind]
        self.cbest = np.min(self.cost)
        self.cbest_id = np.argmin(self.cost)
        self.group = self.group[ind]
        self.F = self.F[ind]
        self.Cr = self.Cr[ind]

    # calculate new population size with non-linear population size reduction
    def cal_NP_next_gen(self, FEs, MaxFEs):
        NP = np.round(self.Nmax + (self.Nmin - self.Nmax) * np.power(FEs/MaxFEs, 1-FEs/MaxFEs))
        return NP

    # slice the population and its cost, crossover rate, etc
    def slice(self, size):
        self.NP = size
        self.group = self.group[:size]
        self.cost = self.cost[:size]
        self.F = self.F[:size]
        self.Cr = self.Cr[:size]
        if self.cbest_id >= size:
            self.cbest_id = np.argmin(self.cost)
            self.cbest = np.min(self.cost)

    # calculate wL mean
    def mean_wL(self, df, s):
        w = df / np.sum(df)
        if np.sum(w * s) > 0.000001:
            return np.sum(w * (s ** 2)) / np.sum(w * s)
        else:
            return 0.5

    # randomly choose step length nad crossover rate from MF and MCr
    def choose_F_Cr(self):
        # generate Cr can be done simutaneously
        gs = self.NP
        ind_r = np.random.randint(0, self.MF.shape[0], size=gs)  # index
        C_r = np.minimum(1, np.maximum(0, np.random.normal(loc=self.MCr[ind_r], scale=0.1, size=gs)))
        # as for F, need to generate 1 by 1
        cauchy_locs = self.MF[ind_r]
        F = stats.cauchy.rvs(loc=cauchy_locs, scale=0.1, size=gs)
        err = np.where(F < 0)[0]
        F[err] = 2 * cauchy_locs[err] - F[err]
        # F = []
        # for i in range(gs):
        #     while True:
        #         f = stats.cauchy.rvs(loc=cauchy_locs[i], scale=0.1)
        #         if f >= 0:
        #             F.append(f)
        #             break
        return C_r, np.minimum(1, F)

    # update MF and MCr, join new value into the set if there are some successful changes or set it to initial value
    def update_M_F_Cr(self, SF, SCr, df):
        if SF.shape[0] > 0:
            mean_wL = self.mean_wL(df, SF)
            self.MF[self.k] = mean_wL
            mean_wL = self.mean_wL(df, SCr)
            self.MCr[self.k] = mean_wL
            self.k = (self.k + 1) % self.MF.shape[0]
        else:
            self.MF[self.k] = 0.5
            self.MCr[self.k] = 0.5

    # non-linearly reduce population size and update it into new population
    def NLPSR(self, FEs, MaxFEs):
        self.sort(self.NP)
        N = self.cal_NP_next_gen(FEs, MaxFEs)
        A = int(max(N * 2.1, self.Nmin))
        N = int(N)
        if N < self.NP:
            self.slice(N)
        if A < self.archive.shape[0]:
            self.NA = A
            self.archive = self.archive[:A]

    # update archive, join new individual
    def update_archive(self, old_id):
        if self.archive.shape[0] < self.NA:
            self.archive = np.append(self.archive, self.group[old_id]).reshape(-1, self.dim)
        else:
            self.archive[np.random.randint(self.archive.shape[0])] = self.group[old_id]


class MadDE:
    def __init__(self, dim):
        self.dim = dim
        self.p = 0.18
        self.PqBX = 0.01
        self.F0 = 0.2
        self.Cr0 = 0.2
        self.pm = np.ones(3) / 3

    def ctb_w_arc(self, group, best, archive, Fs):
        NP, dim = group.shape
        NB = best.shape[0]
        NA = archive.shape[0]

        count = 0
        rb = np.random.randint(NB, size=NP)
        duplicate = np.where(rb == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 25:
            rb[duplicate] = np.random.randint(NB, size=duplicate.shape[0])
            duplicate = np.where(rb == np.arange(NP))[0]
            count += 1

        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP + NA, size=NP)
        duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP + NA, size=duplicate.shape[0])
            duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        xb = best[rb]
        x1 = group[r1]
        if NA > 0:
            x2 = np.concatenate((group, archive), 0)[r2]
        else:
            x2 = group[r2]
        v = group + Fs * (xb - group) + Fs * (x1 - x2)

        return v

    def ctr_w_arc(self, group, archive, Fs):
        NP, dim = group.shape
        NA = archive.shape[0]

        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where((r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r1 == np.arange(NP)))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP + NA, size=NP)
        duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP + NA, size=duplicate.shape[0])
            duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        x1 = group[r1]
        if NA > 0:
            x2 = np.concatenate((group, archive), 0)[r2]
        else:
            x2 = group[r2]
        v = group + Fs * (x1 - x2)

        return v

    def weighted_rtb(self, group, best, Fs, Fas):
        NP, dim = group.shape
        NB = best.shape[0]

        count = 0
        rb = np.random.randint(NB, size=NP)
        duplicate = np.where(rb == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 25:
            rb[duplicate] = np.random.randint(NB, size=duplicate.shape[0])
            duplicate = np.where(rb == np.arange(NP))[0]
            count += 1

        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP, size=NP)
        duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        xb = best[rb]
        x1 = group[r1]
        x2 = group[r2]
        v = Fs * x1 + Fs * Fas * (xb - x2)

        return v

    def binomial(self, x, v, Crs):
        NP, dim = x.shape
        jrand = np.random.randint(dim, size=NP)
        u = np.where(np.random.rand(NP, dim) < Crs, v, x)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        return u

    def step(self, population, problem, FEs, FEs_end, MaxFEs):
        population.sort(population.NP)
        while FEs < FEs_end and FEs < MaxFEs:
            NP, dim = population.NP, population.dim
            q = 2 * self.p - self.p * FEs / MaxFEs
            Fa = 0.5 + 0.5 * FEs / MaxFEs
            Cr, F = population.choose_F_Cr()
            mu = np.random.choice(3, size=NP, p=self.pm)
            p1 = population.group[mu == 0]
            p2 = population.group[mu == 1]
            p3 = population.group[mu == 2]
            pbest = population.group[:max(int(self.p * NP), 2)]
            qbest = population.group[:max(int(q * NP), 2)]
            Fs = F.repeat(dim).reshape(NP, dim)
            v1 = self.ctb_w_arc(p1, pbest, population.archive, Fs[mu == 0])
            v2 = self.ctr_w_arc(p2, population.archive, Fs[mu == 1])
            v3 = self.weighted_rtb(p3, qbest, Fs[mu == 2], Fa)
            v = np.zeros((NP, dim))
            v[mu == 0] = v1
            v[mu == 1] = v2
            v[mu == 2] = v3
            v[v < -100] = (population.group[v < -100] - 100) / 2
            v[v > 100] = (population.group[v > 100] + 100) / 2
            rvs = np.random.rand(NP)
            Crs = Cr.repeat(dim).reshape(NP, dim)
            u = np.zeros((NP, dim))
            if np.sum(rvs <= self.PqBX) > 0:
                qu = v[rvs <= self.PqBX]
                if population.archive.shape[0] > 0:
                    qbest = np.concatenate((population.group, population.archive), 0)[:max(int(q * (NP + population.archive.shape[0])), 2)]
                cross_qbest = qbest[np.random.randint(qbest.shape[0], size=qu.shape[0])]
                qu = self.binomial(cross_qbest, qu, Crs[rvs <= self.PqBX])
                u[rvs <= self.PqBX] = qu
            bu = v[rvs > self.PqBX]
            bu = self.binomial(population.group[rvs > self.PqBX], bu, Crs[rvs > self.PqBX])
            u[rvs > self.PqBX] = bu
            ncost = problem.func(u)
            FEs += NP
            optim = np.where(ncost < population.cost)[0]
            for i in optim:
                population.update_archive(i)
            SF = F[optim]
            SCr = Cr[optim]
            df = np.maximum(0, population.cost - ncost)
            population.update_M_F_Cr(SF, SCr, df[optim])
            count_S = np.zeros(3)
            for i in range(3):
                count_S[i] = np.mean(df[mu == i] / population.cost[mu == i])
            if np.sum(count_S) > 0:
                self.pm = np.maximum(0.1, np.minimum(0.9, count_S / np.sum(count_S)))
                self.pm /= np.sum(self.pm)
            else:
                self.pm = np.ones(3) / 3

            population.group[optim] = u[optim]
            population.cost = np.minimum(population.cost, ncost)
            population.NLPSR(FEs, MaxFEs)
            if np.min(population.cost) < population.gbest:
                population.gbest = np.min(population.cost)
                population.gbest_solution = population.group[np.argmin(population.cost)]

            if np.min(population.cost) < 1e-8:
                return population
        return population
