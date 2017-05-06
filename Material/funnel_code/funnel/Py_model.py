#!/bin/python

##############################################################
#    Yasuko Matsubara 
#    Date: October, 2013
##############################################################

import sys
import math as M

try:
    from scipy.stats import poisson
except:
    print "can not find scipy"
    print "for mac users:"
    print "sudo port install py27-scipy"
    raise

try:
    import pylab as P
    has_pylab = True
except:
    print "sudo apt-get install pylab      matplotlib "
    print "    continuing without plotting functionality"

    has_pylab = False

import array as A

##########################
# exogenous function
##########################
def exo(t, nc, Sc):
    if(t==nc):
      return Sc
    else:
      return 0
##########################
# sinusoidal periodicity 
##########################
def period(t, pfreq, prate, pshift):
    if (prate==0):
        return 1
# (1+cosine)
    val = 1 + prate*M.cos( (2*M.pi/pfreq)*(t+pshift) )
    return val



def funnel(T, N, beta0, delta, gamma, \
           Pp, Pa, Ps, thetaT, theta0, \
           Emean, Evar, E0, st, I0):

    thetaT = int(thetaT); # disease reduction, i.e., vaccine
    Emean = int(Emean); # external shock (mean)
    Evar  = int(Evar);  # external shock (variance)
    st = int(st);
    S  = [N]*T;  # susceptible
    I  = [0]*T;  # infected
    V  = [0]*T;  # vigilant
    P  = [0]*T;  # periodicity
    C  = [0]*T;  # disease-reduction
    E  = [1]*T;  # external shock

    # --- if newborn virus, default: (st=0) --- #
    if(st==0):
        S[0] = N-1;
        I[0] = 1;
        I0 = 0;
    else:
        S[0] = N;
        I[0] = 0;
    # --- if newborn virus, default: (st=0) --- #
 

    #--- periodicities ---#
    for t in range(0,T):
        P[t] = beta0*period(t, Pp, Pa, Ps);
    #--- disease reduction, i.e., vaccine ---#
    for t in range(thetaT,T):
        C[t] = theta0;
    #--- external-shock event ---#
    for t in range(Emean-Evar, Emean+Evar):
        E[t] += E0;
    #--- start (t=0,...T-1) ---#
    for t in range(0,T-1):
        ex   =  exo(t, st, I0); # default: ex=0;
        beta =  P[t+1]; # infection rate
        nu   =  C[t+1]; # reduction rate
        
        ## Funnel-RE model ##
        S[t+1] = S[t] - beta*S[t]*E[t]*(I[t]+ex) + gamma*V[t] - S[t]*nu; 
        I[t+1] = I[t] + beta*S[t]*E[t]*(I[t]+ex) - delta*I[t]; 
        V[t+1] = V[t] + delta*I[t] - gamma*V[t] + S[t]*nu;

        assert( abs(I[t+1] + S[t+1] +V[t+1] - N ) < 0.001)
    #--- return variables ---#
    return S, I, V, P


if __name__ == "__main__":



    # basic parameters
    T  = 52*10
    N = 10000
    betaN= 0.6 
    beta = betaN/N
    Emean=1
    Evar=0
    E0=0.0
    delta = 0.5 
    gamma = 0.1
    Pa = 0.40
    Ps = 0.0
    Pp = 52
    thetaT = 300;
    theta  = 0.01;
    st=0
    I0=0

    (S, I, V, Pl) = funnel(T, N, beta, delta, gamma, \
                           Pp, Pa, Ps, thetaT, theta,\
                           Emean, Evar, E0, st, I0);


    if ( has_pylab ):
        tlist = range(0, T)
        ptitle= "Funnel-RE: N=%2d betaN=%.2e delta=%.2e gamma=%.2e" %\
        (N, betaN, delta, gamma)
        P.title(ptitle)
        P.xlabel("Time")
        P.ylabel("Number of infectives @ time ")
        P.axis([1, T, 0, max(I)+100])
        P.xscale('linear')
        P.yscale('linear')
        P.plot(tlist, I)
        P.show()
    else:
        print "no pylab - do 'make demo' to see gnuplot demo"
