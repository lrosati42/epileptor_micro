'''
Created on 13 juin 2012

@author: Squirel
'''
from numpy import *
from decimal import *

class slow_synapse(object):
    '''
    classdocs
    '''

    def __init__(self, g_bar, E, alpha, beta, Vt, m, r, s, n, Kd, K3, K4):
        '''
        Constructor
        '''
        """self.Tmax=1; self.VT=2; self.Kp=5;
        self.Vampa=0; self.a_r=1.1; self.a_d=0.19;"""
        self.gmax=g_bar
        self.Esyn=E
        self.a=alpha
        self.b=beta
        self.Cmax=0.5
        # self.threshold=0
        # self.Cdur=0.3
        # self.deadtime=1
        self.lr=-1000   #last release initialization
        self.m=m        # TODO: remove or self.m=0
        # self.m0=0
        # self.tau_r=1/(self.a*self.Cmax+self.b)
        # self.m1=0
        self.Kd=Kd
        self.K3=K3
        self.K4=K4
        self.n=n    # 4
        self.r=r    # 0
        self.s=s    # 0
        self.Vt=Vt
        # self.C_plot=[]

    def eq(self,Vpre, Vpost,dt,t):
        Cmax=1; Kp=5
        C = Cmax/(1+exp(-(Vpre-self.Vt)/Kp))
        # self.C_plot.append(C)
        try:
            self.r = self.r + (self.a*C*(1-self.r)-self.b*self.r)*dt
            self.s = self.s + (self.K3*self.r-self.K4*self.s)*dt

            I = -self.gmax*(self.s**self.n)*(Vpost-self.Esyn)/(self.Kd+self.s**self.n)
        except Exception as e:
            print( Vpre, Vpost, self.m, self.lr, e)

        return I

    def get_params(self):
        return self.gmax, self.Esyn, self.a, self.b
