'''
Created on 20 juil. 2012

@author: Squirel
'''

import numpy as np
from numpy.random import uniform

from epileptor.python import pop1n
from epileptor.python import pop2n

from tqdm import tqdm

def find_threshold_indices(array, threshold):
    above_threshold_indices = []
    above_threshold = False
    for i, value in enumerate(array):
        if value > threshold:
            if not above_threshold:
                above_threshold_indices.append(i)
                above_threshold = True
        else:
            above_threshold = False
    return above_threshold_indices

def filter_indices(indices, threshold):
    filtered_indices = []
    if not indices:  # If the input list is empty, return empty list
        return filtered_indices

    first_index = indices[0]  # First kept index
    filtered_indices.append(first_index)

    for index in indices[1:]:
        if index - first_index > threshold:
            filtered_indices.append(index)
            first_index = index  # Update the first index to the current index

    return filtered_indices

class SeizureNetwork:
    def __init__(self, args):
        self.parameters = Parameters(args)
        self.initialize_arrays()

    def initialize_arrays(self):
        nbn1 = self.parameters.nbn1
        nbn2 = self.parameters.nbn2
        t_tot = self.parameters.t_tot
        dt = self.parameters.dt
        n = self.parameters.n

        self.x1_plot = np.zeros((nbn1, int(t_tot / dt / n)))
        self.x2_plot = np.zeros((nbn2, int(t_tot / dt / n)))
        self.y1_plot = np.zeros((nbn1, int(t_tot / dt / n)))
        self.y2_plot = np.zeros((nbn2, int(t_tot / dt / n)))
        self.x1bar_plot = np.zeros(int(t_tot / dt / n))
        self.x2bar_plot = np.zeros(int(t_tot / dt / n))
        self.y1bar_plot = np.zeros(int(t_tot / dt / n))
        self.y2bar_plot = np.zeros(int(t_tot / dt / n))
        self.z_plot = np.zeros((nbn1, int(t_tot / dt / n)))
        self.zbar_plot = np.zeros(int(t_tot / dt / n))
        self.KOP_plot = np.zeros((2, int(t_tot / dt / n)))
        self.evs = np.empty(shape=(1,2))

        # arrays to store means between samples
        self.x1_nsamples = np.zeros((nbn1, n))
        self.x2_nsamples = np.zeros((nbn2, n))
        self.y1_nsamples = np.zeros((nbn1, n))
        self.y2_nsamples = np.zeros((nbn2, n))
        self.z_nsamples = np.zeros((nbn1, n))
        self.x1_nsamples_i = np.zeros((nbn1))
        self.x2_nsamples_i = np.zeros((nbn2))
        self.y1_nsamples_i = np.zeros((nbn1))
        self.y2_nsamples_i = np.zeros((nbn2))
        self.z_nsamples_i = np.zeros((nbn1))
        self.x1bar_nsamples = np.zeros(n)
        self.x2bar_nsamples = np.zeros(n)
        self.y1bar_nsamples = np.zeros(n)
        self.y2bar_nsamples = np.zeros(n)
        self.zbar_nsamples = np.zeros(n)
        self.KOP_nsamples = np.zeros((2, n))


    def initialize_networks(self):
        nbn1 = self.parameters.nbn1
        nbn2 = self.parameters.nbn2
        CpES1 = self.parameters.CpES1
        CpES2 = self.parameters.CpES2
        CpCS = self.parameters.CpCS
        g_x1x1 = self.parameters.g_x1x1
        g_x2x2 = self.parameters.g_x2x2
        g_x1x2 = self.parameters.g_x1x2
        g_x2x1 = self.parameters.g_x2x1
        g_x2x1_slow = self.parameters.g_x2x1_slow
        g_x2x2_slow = self.parameters.g_x2x2_slow
        I1 = self.parameters.I1
        I2 = self.parameters.I2
        c2 = self.parameters.c2
        m = self.parameters.m
        x0 = self.parameters.x0
        r = self.parameters.r
        s = self.parameters.s
        noise1 = self.parameters.noise1
        noise2 = self.parameters.noise2
        noise3 = self.parameters.noise3

        # set random initial conditions
        x1_init = [uniform(self.parameters.x1_min, self.parameters.x1_max) for _ in range(nbn1)]
        y1_init = [uniform(self.parameters.y1_min, self.parameters.y1_max) for _ in range(nbn1)]
        z_init = [uniform(self.parameters.z_min, self.parameters.z_max) for _ in range(nbn1)]
        x2_init = [uniform(self.parameters.pop2_x2_min, self.parameters.pop2_x2_max) for _ in range(nbn2)]
        y2_init = [uniform(self.parameters.pop2_y2_min, self.parameters.pop2_y2_max) for _ in range(nbn2)]

        self.pop1 = [pop1n.pop1n(a=self.parameters.a, b=self.parameters.b, c=self.parameters.c, d=self.parameters.d, m=m,
                                 x0=x0, z0=self.parameters.z0, CpES=CpES1, CpCS=CpCS, g_x1x1=g_x1x1, g_x2x1=g_x2x1, g_x2x1_slow=g_x2x1_slow,
                                 I1=I1, r=r, s=s, noise=noise1, noise3=noise3,
                                 E_11=self.parameters.syn_x1x1_Esyn, alpha_11=self.parameters.syn_x1x1_a, beta_11=self.parameters.syn_x1x1_b, Vt_11=self.parameters.syn_x1x1_Vt, m_11=self.parameters.syn_x1x1_m,
                                 Emin_21=self.parameters.syn_x2x1_Esyn_min, Emax_21=self.parameters.syn_x2x1_Esyn_max, alpha_21=self.parameters.syn_x2x1_a, beta_21=self.parameters.syn_x2x1_b, Vt_21=self.parameters.syn_x2x1_Vt, m_21=self.parameters.syn_x2x1_m,
                                 E_21_s=self.parameters.syn_x2x1_slow_Esyn, alpha_21_s=self.parameters.syn_x2x1_slow_a, beta_21_s=self.parameters.syn_x2x1_slow_b, Vt_21_s=self.parameters.syn_x2x1_slow_Vt,
                                 r_21_s=self.parameters.syn_x2x1_slow_r, s_21_s=self.parameters.syn_x2x1_slow_s, n_21_s=self.parameters.syn_x2x1_slow_n,
                                 Kd=self.parameters.syn_x2x1_slow_Kd, K3=self.parameters.syn_x2x1_slow_K3, K4=self.parameters.syn_x2x1_slow_K4) for _ in range(nbn1)]
        for i in range(nbn1):
            self.pop1[i].x1 = x1_init[i]
            self.pop1[i].y1 = y1_init[i]
            self.pop1[i].z = z_init[i]

        self.pop2 = [pop2n.pop2n(aa=self.parameters.pop2_aa, tau=self.parameters.pop2_tau, z0=self.parameters.pop2_z0,
                                 CpES=CpES2, CpCS=CpCS, g_x2x2=g_x2x2, g_x1x2=g_x1x2, I2=I2, g_x2x2_slow=g_x2x2_slow,
                                 c2=c2, noise=noise2,
                                 E_22=self.parameters.pop2_syn_x2x2_Esyn, alpha_22=self.parameters.pop2_syn_x2x2_a, beta_22=self.parameters.pop2_syn_x2x2_b, Vt_22=self.parameters.pop2_syn_x2x2_Vt, m_22=self.parameters.pop2_syn_x2x2_m,
                                 E_12=self.parameters.pop2_syn_x1x2_Esyn, alpha_12=self.parameters.pop2_syn_x1x2_a, beta_12=self.parameters.pop2_syn_x1x2_b, Vt_12=self.parameters.pop2_syn_x1x2_Vt, m_12=self.parameters.pop2_syn_x1x2_m,
                                 E_22_s=self.parameters.pop2_syn_x2x2_slow_Esyn, alpha_22_s=self.parameters.pop2_syn_x2x2_slow_a, beta_22_s=self.parameters.pop2_syn_x2x2_slow_b, Vt_22_s=self.parameters.pop2_syn_x2x2_slow_Vt,
                                 r_22_s=self.parameters.pop2_syn_x2x2_slow_r, s_22_s=self.parameters.pop2_syn_x2x2_slow_s, n_22_s=self.parameters.pop2_syn_x2x2_slow_n,
                                 V1=self.parameters.pop2_V1, V2=self.parameters.pop2_V2, V3=self.parameters.pop2_V3, V4=self.parameters.pop2_V4, phi=self.parameters.pop2_phi, 
                                 gCa_bar=self.parameters.pop2_gCa_bar, gK_bar=self.parameters.pop2_gK_bar, gL_bar=self.parameters.pop2_gL_bar, EK=self.parameters.pop2_EK, EL=self.parameters.pop2_EL, ECa=self.parameters.pop2_ECa, Cm=self.parameters.pop2_Cm,
                                 Kd=self.parameters.pop2_syn_x2x2_slow_Kd, K3=self.parameters.pop2_syn_x2x2_slow_K3, K4=self.parameters.pop2_syn_x2x2_slow_K4) for _ in range(nbn2)]
        for j in range(nbn2):
            self.pop2[j].x2 = x2_init[j]
            self.pop2[j].y2 = y2_init[j]

        # connections between neurons
        for i in range(nbn1):
            self.pop1[i].connect_syn_pop2n(self.pop2[:])
            self.pop1[i].connect_gap(self.pop1[:])
        for j in range(nbn2):
            self.pop2[j].connect_syn_pop1n(self.pop1[:])
            self.pop2[j].connect_syn_pop2n(self.pop2[:])
            self.pop2[j].connect_gap(self.pop2[:])

        self.x1bar = np.average(x1_init)
        self.x2bar = np.average(x2_init)
        self.zbar = np.average(z_init)

    def advance_simulation(self, t_stop, x0_variable, CpES_variable):
        nbn1 = self.parameters.nbn1
        nbn2 = self.parameters.nbn2
        t_now = self.parameters.t_now
        dt = self.parameters.dt
        n = self.parameters.n

        count_samples = 0
        for ti in tqdm(np.arange(t_now / dt, t_stop / dt)):
            for i in range(nbn1):
                self.pop1[i].x0 = x0_variable[int(ti - t_now / dt )]
                self.pop1[i].CpES = CpES_variable[int(ti - t_now / dt )]
                self.x1_nsamples[i, count_samples], self.y1_nsamples[i, count_samples], self.z_nsamples[i, count_samples] = self.pop1[i].euler(dt, 0, self.x1bar, self.x2bar, self.zbar, ti)
                # for j in range(nbn2): # nbn1 == nbn2
                self.pop2[i].CpES = CpES_variable[int(ti- t_now / dt )]
                self.x2_nsamples[i, count_samples], self.y2_nsamples[i, count_samples] = self.pop2[i].euler(dt, 0, self.x1bar, self.x2bar, self.zbar, ti)

            self.x1bar = np.average(self.x1_nsamples[:, count_samples])
            self.x2bar = np.average(self.x2_nsamples[:, count_samples])
            self.zbar = np.average(self.z_nsamples[:, count_samples])

            self.x1bar_nsamples[count_samples] = self.x1bar
            self.x2bar_nsamples[count_samples] = self.x2bar
            self.y1bar_nsamples[count_samples] = np.average(self.y1_nsamples[:, count_samples])
            self.y2bar_nsamples[count_samples] = np.average(self.y2_nsamples[:, count_samples])
            self.zbar_nsamples[count_samples] = self.zbar

            count_samples += 1
            if count_samples == n:
                ti_n = int(ti / n)
                self.x1_plot[:, ti_n] = self.x1_nsamples.mean(axis=1)
                self.x2_plot[:, ti_n] = self.x2_nsamples.mean(axis=1)
                self.y1_plot[:, ti_n] = self.y1_nsamples.mean(axis=1)
                self.y2_plot[:, ti_n] = self.y2_nsamples.mean(axis=1)
                self.z_plot[:, ti_n] = self.z_nsamples.mean(axis=1)
                self.x1bar_plot[ti_n] = self.x1bar_nsamples.mean()
                self.x2bar_plot[ti_n] = self.x2bar_nsamples.mean()
                self.y1bar_plot[ti_n] = self.y1bar_nsamples.mean()
                self.y2bar_plot[ti_n] = self.y2bar_nsamples.mean()
                self.zbar_plot[ti_n] = self.zbar_nsamples.mean()
                self.KOP_plot[:, ti_n] = np.mean(self.KOP_nsamples, axis=1)
                count_samples = 0

        self.t_plot = np.arange(0, int((t_stop -t_now) / dt / n))
        self.t_plot_dt = np.arange(0, int((t_stop-t_now)), dt)
        self.parameters.t_now = t_stop

        evs_tot = None
        k = 0.7
        for nx, data in enumerate([self.x1_plot[: , int(t_now / dt /n):int(t_stop / dt /n)], self.x2_plot[:, int(t_now / dt /n):int(t_stop / dt /n)]]):
            for channel in range(data.shape[0]):
                mean = np.mean(data[channel,:])
                std = np.std(data[channel,:])
                thr = mean + k*std
                evs = find_threshold_indices(data[channel,:], thr)
                filtered_indices = filter_indices(evs, 60)
                # if filtered_indices[0] < 20:
                #     filtered_indices = filtered_indices[1:]
                t_evs = [channel + (nx*nbn1) for i in range(len(filtered_indices))]

                evs_tmp = np.concatenate([np.array(t_evs).reshape(-1,1), np.array(filtered_indices).reshape(-1,1)], axis=1)

                evs_tot = np.concatenate([evs_tot, evs_tmp], axis=0) if (channel + (nx*nbn1)) else evs_tmp

            if evs_tot is not None:
                self.evs = np.concatenate([self.evs, evs_tot], axis=0)

        return self.x1_plot[: , int(t_now / dt /n):int(t_stop / dt /n)], self.x2_plot[:, int(t_now / dt /n):int(t_stop / dt /n)], evs_tot


class Parameters:
    def __init__(self, args):
        # original parameters
        self.t_tot = int(args.get("simulation", {}).get("t_tot", None))
        self.CpES1 = float(args.get("pop1", {}).get("CpES", None))
        self.CpES2 = float(args.get("pop2", {}).get("CpES", None))
        self.CpCS = float(args.get("pop1", {}).get("CpCS", None))
        self.m = float(args.get("pop1", {}).get("m", None))
        self.r = float(args.get("pop1", {}).get("r", None))
        self.s = float(args.get("pop1", {}).get("s", None))
        self.x0 = float(args.get("pop1", {}).get("x0", None))
        self.g_x1x1 = float(args.get("pop1", {}).get("syn_x1x1.gmax", None))
        self.g_x2x2 = float(args.get("pop2", {}).get("syn_x2x2.gmax", None))
        self.g_x1x2 = float(args.get("pop2", {}).get("syn_x1x2.gmax", None))
        self.g_x2x1 = float(args.get("pop1", {}).get("syn_x2x1.gmax", None))
        self.g_x2x1_slow = float(args.get("pop1", {}).get("syn_x2x1_slow.gmax", None))
        self.g_x2x2_slow = float(args.get("pop2", {}).get("syn_x2x2_slow.gmax", None))
        self.c2 = float(args.get("pop2", {}).get("c2", None))
        self.I2 = float(args.get("pop2", {}).get("I2", None))
        self.I1 = float(args.get("pop1", {}).get("I1", None))
        self.nbn1 = int(args.get("simulation", {}).get("nbn1", None))
        self.nbn2 = int(args.get("simulation", {}).get("nbn2", None))
        self.noise2 = float(args.get("pop2", {}).get("noise", None))
        self.noise3 = float(args.get("pop1", {}).get("noise3", None))
        self.noise1 = self.noise2 * 20
        self.dt = float(args.get("simulation", {}).get("dt", None))
        self.fs = int(args.get("simulation", {}).get("fs", None))
        self.n = int(((1/self.dt) / self.fs) * 1000)
        self.t_now = 0

        # additional pop1 parameters
        self.a = float(args.get("pop1", {}).get("a", None))
        self.b = float(args.get("pop1", {}).get("b", None))
        self.c = float(args.get("pop1", {}).get("c", None))
        self.d = float(args.get("pop1", {}).get("d", None))
        self.z0 = float(args.get("pop1", {}).get("z0", None))
        self.x1_min = float(args.get("pop1", {}).get("x1_min", None))
        self.x1_max = float(args.get("pop1", {}).get("x1_max", None))
        self.y1_min = float(args.get("pop1", {}).get("y1_min", None))
        self.y1_max = float(args.get("pop1", {}).get("y1_max", None))
        self.z_min = float(args.get("pop1", {}).get("z_min", None))
        self.z_max = float(args.get("pop1", {}).get("z_max", None))
        self.syn_x1x1_gmax = float(args.get("pop1", {}).get("syn_x1x1.gmax", None))
        self.syn_x1x1_Esyn = float(args.get("pop1", {}).get("syn_x1x1.Esyn", None))
        self.syn_x1x1_a = float(args.get("pop1", {}).get("syn_x1x1.a", None))
        self.syn_x1x1_b = float(args.get("pop1", {}).get("syn_x1x1.b", None))
        self.syn_x1x1_Vt = float(args.get("pop1", {}).get("syn_x1x1.Vt", None))
        self.syn_x1x1_m = float(args.get("pop1", {}).get("syn_x1x1.m", None))
        self.syn_x2x1_gmax = float(args.get("pop1", {}).get("syn_x2x1.gmax", None))
        self.syn_x2x1_Esyn_min = float(args.get("pop1", {}).get("syn_x2x1.Esyn_min", None))
        self.syn_x2x1_Esyn_max = float(args.get("pop1", {}).get("syn_x2x1.Esyn_max", None))
        self.syn_x2x1_a = float(args.get("pop1", {}).get("syn_x2x1.a", None))
        self.syn_x2x1_b = float(args.get("pop1", {}).get("syn_x2x1.b", None))
        self.syn_x2x1_Vt = float(args.get("pop1", {}).get("syn_x2x1.Vt", None))
        self.syn_x2x1_m = float(args.get("pop1", {}).get("syn_x2x1.m", None))
        self.syn_x2x1_slow_gmax = float(args.get("pop1", {}).get("syn_x2x1_slow.gmax", None))
        self.syn_x2x1_slow_Esyn = float(args.get("pop1", {}).get("syn_x2x1_slow.Esyn", None))
        self.syn_x2x1_slow_a = float(args.get("pop1", {}).get("syn_x2x1_slow.a", None))
        self.syn_x2x1_slow_b = float(args.get("pop1", {}).get("syn_x2x1_slow.b", None))
        self.syn_x2x1_slow_Vt = float(args.get("pop1", {}).get("syn_x2x1_slow.Vt", None))
        self.syn_x2x1_slow_r = float(args.get("pop1", {}).get("syn_x2x1_slow.r", None))
        self.syn_x2x1_slow_s = float(args.get("pop1", {}).get("syn_x2x1_slow.s", None))
        self.syn_x2x1_slow_Kd = float(args.get("pop1", {}).get("syn_x2x1_slow.Kd", None))
        self.syn_x2x1_slow_K3 = float(args.get("pop1", {}).get("syn_x2x1_slow.K3", None))
        self.syn_x2x1_slow_K4 = float(args.get("pop1", {}).get("syn_x2x1_slow.K4", None))
        self.syn_x2x1_slow_n = float(args.get("pop1", {}).get("syn_x2x1_slow.n", None))
        self.pop1in = args.get("pop1", {}).get("pop1in", None)
        self.pop1in_count = int(args.get("pop1", {}).get("pop1in_count", 0))
        self.pop1in_gj = args.get("pop1", {}).get("pop1in_gj", None)
        self.pop1in_gj_count = int(args.get("pop1", {}).get("pop1in_gj_count", 0))
        self.pop2in = args.get("pop1", {}).get("pop2in", None)
        self.pop2in_count = int(args.get("pop1", {}).get("pop2in_count", 0))

        # additional pop2 parameters
        self.pop2_aa = float(args.get("pop2", {}).get("aa", None))
        self.pop2_tau = float(args.get("pop2", {}).get("tau", None))
        self.pop2_I2 = float(args.get("pop2", {}).get("I2", None))
        self.pop2_z0 = float(args.get("pop2", {}).get("z0", None))

        self.pop2_x2_min = float(args.get("pop2", {}).get("x2_min", None))
        self.pop2_x2_max = float(args.get("pop2", {}).get("x2_max", None))
        self.pop2_y2_min = float(args.get("pop2", {}).get("y2_min", None))
        self.pop2_y2_max = float(args.get("pop2", {}).get("y2_max", None))
        self.pop2_c2 = float(args.get("pop2", {}).get("c2", None))
        self.pop2_noise = float(args.get("pop2", {}).get("noise", None))

        self.pop2_CpES = float(args.get("pop2", {}).get("CpES", None))
        self.pop2_CpCS = float(args.get("pop2", {}).get("CpCS", None))

        self.pop2_V1 = float(args.get("pop2", {}).get("V1", None))
        self.pop2_V2 = float(args.get("pop2", {}).get("V2", None))
        self.pop2_V3 = float(args.get("pop2", {}).get("V3", None))
        self.pop2_V4 = float(args.get("pop2", {}).get("V4", None))
        self.pop2_phi = float(args.get("pop2", {}).get("phi", None))
        self.pop2_gCa_bar = float(args.get("pop2", {}).get("gCa_bar", None))
        self.pop2_gK_bar = float(args.get("pop2", {}).get("gK_bar", None))
        self.pop2_gL_bar = float(args.get("pop2", {}).get("gL_bar", None))
        self.pop2_ECa = float(args.get("pop2", {}).get("ECa", None))
        self.pop2_EK = float(args.get("pop2", {}).get("EK", None))
        self.pop2_EL = float(args.get("pop2", {}).get("EL", None))
        self.pop2_Cm = float(args.get("pop2", {}).get("Cm", None))

        # syn_x2x2
        self.pop2_syn_x2x2_gmax = float(args.get("pop2", {}).get("syn_x2x2.gmax", None))
        self.pop2_syn_x2x2_Esyn = float(args.get("pop2", {}).get("syn_x2x2.Esyn", None))
        self.pop2_syn_x2x2_a = float(args.get("pop2", {}).get("syn_x2x2.a  ", None))
        self.pop2_syn_x2x2_b = float(args.get("pop2", {}).get("syn_x2x2.b  ", None))
        self.pop2_syn_x2x2_Vt = float(args.get("pop2", {}).get("syn_x2x2.Vt ", None))
        self.pop2_syn_x2x2_m = float(args.get("pop2", {}).get("syn_x2x2.m  ", None))

        # syn_x1x2
        self.pop2_syn_x1x2_gmax = float(args.get("pop2", {}).get("syn_x1x2.gmax", None))
        self.pop2_syn_x1x2_Esyn = float(args.get("pop2", {}).get("syn_x1x2.Esyn", None))
        self.pop2_syn_x1x2_a = float(args.get("pop2", {}).get("syn_x1x2.a  ", None))
        self.pop2_syn_x1x2_b = float(args.get("pop2", {}).get("syn_x1x2.b  ", None))
        self.pop2_syn_x1x2_Vt = float(args.get("pop2", {}).get("syn_x1x2.Vt ", None))
        self.pop2_syn_x1x2_m = float(args.get("pop2", {}).get("syn_x1x2.m  ", None))

        # syn_x2x2_slow
        self.pop2_syn_x2x2_slow_gmax = float(args.get("pop2", {}).get("syn_x2x2_slow.gmax", None))
        self.pop2_syn_x2x2_slow_Esyn = float(args.get("pop2", {}).get("syn_x2x2_slow.Esyn", None))
        self.pop2_syn_x2x2_slow_a = float(args.get("pop2", {}).get("syn_x2x2_slow.a  ", None))
        self.pop2_syn_x2x2_slow_b = float(args.get("pop2", {}).get("syn_x2x2_slow.b  ", None))
        self.pop2_syn_x2x2_slow_Vt = float(args.get("pop2", {}).get("syn_x2x2_slow.Vt ", None))
        self.pop2_syn_x2x2_slow_r = float(args.get("pop2", {}).get("syn_x2x2_slow.r  ", None))
        self.pop2_syn_x2x2_slow_s = float(args.get("pop2", {}).get("syn_x2x2_slow.s  ", None))
        self.pop2_syn_x2x2_slow_Kd = float(args.get("pop2", {}).get("syn_x2x2_slow.Kd ", None))
        self.pop2_syn_x2x2_slow_K3 = float(args.get("pop2", {}).get("syn_x2x2_slow.K3 ", None))
        self.pop2_syn_x2x2_slow_K4 = float(args.get("pop2", {}).get("syn_x2x2_slow.K4 ", None))
        self.pop2_syn_x2x2_slow_n = float(args.get("pop2", {}).get("syn_x2x2_slow.n  ", None))

        # ingressi
        self.pop2_pop1in = args.get("pop2", {}).get("pop1in", None)
        self.pop2_pop1in_count = int(args.get("pop2", {}).get("pop1in_count", 0))
        self.pop2_pop2in = args.get("pop2", {}).get("pop2in", None)
        self.pop2_pop2in_count = int(args.get("pop2", {}).get("pop2in_count", 0))
        self.pop2_pop2in_gj = args.get("pop2", {}).get("pop2in_gj", None)
        self.pop2_pop2in_gj_count = int(args.get("pop2", {}).get("pop2in_gj_count", 0))