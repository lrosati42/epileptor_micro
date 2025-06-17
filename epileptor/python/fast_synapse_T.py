'''
Created on 13 juin 2012

@author: Squirel
'''
from numpy import *
from decimal import *

class fast_synapse(object):
    '''
    classdocs
    '''

    def __init__(self, g_bar, E, alpha, beta, Vt, m):
        '''
        Constructor
        '''
        """self.Tmax=1; self.VT=2; self.Kp=5;
        self.Vampa=0; self.a_r=1.1; self.a_d=0.19;"""
        self.gmax=g_bar
        self.Esyn=E
        self.a=alpha
        self.b=beta
        # self.Cmax=0.5
        # self.threshold=0
        # self.Cdur=0.3
        # self.deadtime=1
        self.lr=-1000     #last release initialization
        self.m=m          # self.m=0
        # self.m0 = 0
        # self.tau_r = 1/(self.a*self.Cmax+self.b)
        # self.m1 = 0
        self.Vt = Vt

        # self.gmv=[]

        # self.C_plot=[]

    def eq(self,Vpre, Vpost,dt,t):
        Cmax=1; Kp=5;  #max transmiter concentration; half-activation value; steepness;
        C = Cmax/(1+exp(-(Vpre-self.Vt)/Kp))
        # self.C_plot.append(C)
        try:
            self.m = self.m + (self.a*C*(1-self.m)-self.b*self.m)*dt
            I = -self.gmax*self.m*(Vpost-self.Esyn)
        except Exception as e:
            print( Vpre, Vpost, self.m, self.lr, e)
        # self.gmv.append(self.gmax*self.m*Vpost)
        return I

    def get_params(self):
        return self.gmax, self.Esyn, self.a, self.b
