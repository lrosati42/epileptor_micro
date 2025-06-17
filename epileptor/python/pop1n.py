'''
Created on 20 juil. 2012

@author: Squirel
'''
import pylab as pb
from epileptor.python import fast_synapse_T as fast_synapse
from epileptor.python import slow_synapse_T as slow_synapse
from random import uniform

class pop1n(object):
    '''
    classdocs
    '''

    def __init__(self, a=1., b=3., c=1., d=5., m=0.6, s=4., x0=-1.6, z0=0., r=0.00035, I1=3.1, CpES=0.1, CpCS=0.9, g_x1x1=0., g_x2x1=0.5, g_x2x1_slow=1., noise=2, noise3=0.1,
                 E_11=0.0, alpha_11=1.1, beta_11=0.19, Vt_11=2, m_11=0.0,
                 Emin_21=-80, Emax_21=-50, alpha_21=5, beta_21=0.18, Vt_21=2, m_21=0.0,
                 E_21_s=-95, alpha_21_s=0.09, beta_21_s=0.0012, Vt_21_s=2, r_21_s=0, s_21_s=0, n_21_s=4,
                 Kd=100, K3=0.18, K4=0.034):
        '''
        Constructor
        '''
        self.parameters = a,b,c,d,m,s,x0,z0,r,I1
        self.x1=0.
        self.y1=0.
        self.z=0.
        self.x0=x0
        self.noise=noise
        self.noise3=noise3

        #synapse management syntax: syn_ pre post
        self.syn_x1x1 = fast_synapse.fast_synapse(g_x1x1,E=E_11, alpha=alpha_11, beta=beta_11, Vt=Vt_11, m=m_11) #params: gbar, E, alpha, beta, Vt, m
        self.syn_x2x1 = fast_synapse.fast_synapse(g_x2x1,uniform(Emin_21,Emax_21),alpha=alpha_21, beta=beta_21, Vt=Vt_21, m=m_21) #params: gbar, E, alpha, beta, Vt, m
        self.syn_x2x1_slow = slow_synapse.slow_synapse(g_x2x1_slow, E=E_21_s, alpha=alpha_21_s, beta=beta_21_s, Vt=Vt_21_s, m=0, r=r_21_s, s=s_21_s, n=n_21_s, Kd=Kd, K3=K3, K4=K4)  #params: gbar, E, alpha, beta, Vt, r, s, n
        self.CpES = CpES  # electrical synpases coupling
        self.CpCS = CpCS  # chemical synapses coupling

        #plotting management    !!! CAREFUL with the memory !!!
        #self.y1_plot=[]
        # self.Isyn_x2x1_plot=[]
        #self.x2barx1_plot=[]
        #self.x2barx1_mf_plot=[]
        # self.Isyn_x2x1_slow_plot=[]

        # self.Isyn_x1x1_plot=[]
        #self.x1barx1_plot=[]
        #self.x1barx1_mf_plot=[]
        # self.Igj_x1x1_plot=[]

        # input coupling management
        self.pop2in=[]
        self.pop1in=[]
        self.pop1in_gj=[]


    def euler(self, dt, Iin, x1bar, x2bar, zbar, ti):
        a,b,c,d,m,s,x0,z0,r,I1 = self.parameters

        #get average inputs
        x2in=0 #from pop2 synapses
        for n in self.pop2in:
            x2in+=n.x2
        x2in=x2in/len(self.pop2in)

        x1in=0 #from pop1 synapses
        for n in self.pop1in:
            x1in+=n.x1
        x1in=pb.array(x1in).mean()

        x1in_gj=0 #from pop1 gap junctions
        for n in self.pop1in_gj:
            x1in_gj+=n.x1
        x1in_gj=x1in_gj/len(self.pop1in_gj)

        Isyn_x2x1 = self.syn_x2x1.eq(x2in*50, self.x1*50, dt, ti*dt)
        #synapses value storage
        #self.Isyn_x2x1_plot.append(Isyn_x2x1/50)
        #self.x2barx1_plot.append(-x2bar*self.x1)
        #self.x2barx1_mf_plot.append(x2bar-self.x1)

        Isyn_x2x1_slow = self.syn_x2x1_slow.eq(x2in*50, self.x1*50, dt, ti*dt)
        #self.Isyn_x2x1_slow_plot.append(Isyn_x2x1_slow/50)

        Isyn_x1x1 = self.syn_x1x1.eq(x1in*50, self.x1*50, dt, ti*dt)   #not used because gx1x1=0
        #self.Isyn_x1x1_plot.append(Isyn_x1x1/50)
        #self.x1barx1_plot.append(x1bar*self.x1)
        #self.x1barx1_mf_plot.append(x1bar-self.x1)
        #self.Igj_x1x1_plot.append(self.CpES*(x1in_gj-self.x1))

        #if self.x1<0:
        x1next = self.x1 + (self.y1 - a * self.x1**3 + b * self.x1**2 - self.z + I1 + self.CpCS*Isyn_x1x1/50 + self.CpES*(x1in_gj-self.x1) + Isyn_x2x1/50 + Isyn_x2x1_slow/50  + uniform(-self.noise,self.noise))*dt #+ self.CpES*(x1bar-self.x1) + self.CpCS*self.syn_x1x1.eq(x1bar, self.x1, dt, ti*dt))*dt
        #else:
        #    x1next = self.x1 + (self.y1 + (m + 0.6 * (self.z-4)**2)*self.x1 - self.z + I1 + Isyn_x2x1/50 + self.CpCS*Isyn_x1x1/50 + self.CpES*(x1in_gj-self.x1) + Isyn_x2x1_slow/50 + uniform(-0.1,0.1))*dt #+ self.CpES*(x1bar-self.x1) + self.CpCS* self.syn_x1x1.eq(x1bar, self.x1, dt, ti*dt))*dt

        y1next = self.y1 + (c - d * self.x1**2 - self.y1)*dt

        znext = self.z + (r*(s*(self.x1+x2bar-self.x0) - zbar) + uniform(-self.noise3,self.noise3)) *dt
        #znext=zbar
        self.x1, self.y1, self.z = x1next, y1next, znext
        #self.y1_plot.append(y1next)
        return x1next,y1next,znext

    # def get_nullclines(self,x1span):
    #     a,b,c,d,m,s,x0,z0,r,I1 = self.parameters
    #     x1nullcline=[]
    #     y1nullcline=[]
    #     for x in x1span:
    #         x1nullcline.append(a*x**3+b*x**2+z0-I1)
    #         y1nullcline.append(c-d*x**2)
    #     return x1nullcline, y1nullcline

    def connect_syn_pop1n(self, neuron_list):
        for neuron in neuron_list:
            self.pop1in.append(neuron)

    def connect_syn_pop2n(self, neuron_list):
        for neuron in neuron_list:
            self.pop2in.append(neuron)

    def connect_gap(self, neuron_list):
        for neuron in neuron_list:
            self.pop1in_gj.append(neuron)
