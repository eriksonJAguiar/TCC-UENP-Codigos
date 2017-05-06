#!/usr/bin/env python
#!/usr/bin/python
"""
   Interactive main routine, for (Funnel-RE) equation.
   Date: Feb. 2014, Yasuko Matsubara
"""
try:
    from pylab import *
    from matplotlib.widgets import Slider, Button
    #from scipy.stats import poisson
except:
    print "can not find pylab - get it from http://matplotlib.sourceforge.net/"
    raise
try:
    import Py_model 
except:
    print "can not find Py_model.py"
    raise
import sys
import getopt
import math as M
import pylab
#
try:
    import pylab as P
    has_pylab = True
except:
    print "sudo apt-get install pylab      matplotlib "
    print "    continuing without plotting functionality"
    has_pylab = False
import array as A

#---------------#
#     PLOT      #
#---------------#
LOGLOGP=0
ax = subplot(211)
subplots_adjust(bottom=0.45)
def RNFplot(t, X, Y, Z, dat, plotLog):
    if(plotLog==1):
        loglogPlot(t, X, Y, Z, dat);
    if(plotLog==-1):
        semilogPlot(t, X, Y, Z, dat);
    if(plotLog==0): 
        linlinPlot(t, X, Y, dat);
    xlabel("Time")
    ylabel("# of infectives @ time t")
def loglogPlot(t, X, Y, Z, dat):
    loglog(X,lw=2, color='red')
    loglog(Y,lw=2, color='blue')
    loglog(Z,lw=2, color='green')
    loglog(dat, '-', color='black')
    axis([1, len(X), 1, max(X)*10])
def semilogPlot(t, X, Y, Z, dat):
    semilogy(X,lw=2, color='red')
    semilogy(Y, lw=2, color='blue')
    semilogy(Z, lw=2, color='green')
    semilogy(dat, '-', color='black')
    axis([1, len(X), 1, max(X)*10])
def linlinPlot(t, X, Y, dat):
    plot(Y, lw=2, color='blue')
    plot(dat, '-', color='black')
    axis([1, len(X), 0, max(Y)*1.2])

def RSErr(T, x, y):
   if (len(y)==0):
       return 0;
   val = 0.0;
   for i in range(0, T):
       val += (x[i] - y[i])**2;
   val=((val/T)**0.5);
   return val;

#---------------#
#     init      #
#---------------#
# data setting
# load data
if(len(sys.argv)==1):
    dat = []
else:
    dat=pylab.loadtxt(sys.argv[1])
#
# basic parameters
N0 = 10000
betaN0= 0.6 
beta0 = betaN0/N0;
delta0 = 0.4 
gamma0 = 0.1
Pa0 = 0.40
Ps0 = 0.0
Pp0 = 52
thetaT0 = 300;
theta0  = 0.05; 
Em0=1
Ev0=0
E0=0.0
# default (st=0)
st0=0
I00=0

if(len(dat)==0):
    T =52*10;
else:
    # data specific parameters
    T = len(dat) 

# funnel model
(S, I, V, P) = Py_model.funnel(T, N0, beta0, delta0, gamma0, Pp0, Pa0, Ps0, \
                            thetaT0, theta0, Em0, Ev0, E0, st0, I00);
# create the horizontal axis
t = range(T)
print "duration = ", T;
t = [ float(item) for item in t]
RNFplot(t, S, I, V, dat, LOGLOGP);
err = RSErr(T, I, dat);
ptitle= "Funnel N=%2d betaN=%.2e delta=%.2e gamma=%.2e" %\
    (N0, betaN0, delta0, gamma0)
title(ptitle)
ax = subplot(212)
#cla()
plot(t,P, lw=2, color='red')
axis([1, T, min(P), max(P)])

# GUI: slide bars
axcolor = 'lightgoldenrodyellow'
hvcolor = 'darkgoldenrod'
aN      = axes([0.15, 0.35, 0.25, 0.03], axisbg=axcolor)
axbeta  = axes([0.15, 0.30, 0.25, 0.03], axisbg=axcolor)
adelta  = axes([0.15, 0.25, 0.25, 0.03],  axisbg=axcolor)
agamma  = axes([0.15, 0.20, 0.25, 0.03],  axisbg=axcolor)
aPs  = axes([0.15, 0.15, 0.25, 0.03],  axisbg=axcolor)
aPa  = axes([0.15, 0.10, 0.25, 0.03],  axisbg=axcolor)
aPp  = axes([0.15, 0.05, 0.25, 0.03],  axisbg=axcolor)
#
axthetaT  = axes([0.6, 0.35, 0.25, 0.03],  axisbg=axcolor)
axtheta   = axes([0.6, 0.30, 0.25, 0.03],  axisbg=axcolor)
axEm     = axes([0.6, 0.25, 0.25, 0.03], axisbg=axcolor)
axEv     = axes([0.6, 0.20, 0.25, 0.03], axisbg=axcolor)
axE      = axes([0.6, 0.15, 0.25, 0.03], axisbg=axcolor)
#axst     = axes([0.6, 0.10, 0.25, 0.03], axisbg=axcolor)
#axI0     = axes([0.6, 0.05, 0.25, 0.03], axisbg=axcolor)
#
sN     =  Slider(aN,      'N',     1000,  10000, valinit=N0)
sbeta  =  Slider(axbeta,  'beta0', 0.1,  1.5,    valinit=betaN0)
sdelta =  Slider(adelta,  'delta',   0,  0.5,    valinit=delta0)
sgamma =  Slider(agamma,  'gamma',   0,  0.5,    valinit=gamma0)
sPs    =  Slider(aPs,  'P_s',   1,  T/2,  valinit=Ps0)
sPa    =  Slider(aPa,  'P_a',   0,  1,    valinit=Pa0)
sPp    =  Slider(aPp,  'P_p',   0,  T,    valinit=Pp0)
#
sthetaT  =  Slider(axthetaT,  't_th',    0,  T,   valinit=thetaT0)
stheta   =  Slider(axtheta,   'th_0',    0,  0.5, valinit=theta0)
sEm      =  Slider(axEm,  'E_mean',  1,  T,   valinit=Em0)
sEv      =  Slider(axEv,  'E_var',   0,  T/5,   valinit=Ev0)
sE       =  Slider(axE,   'E0',     -1,  2,   valinit=E0)
#sst      =  Slider(axst,   'st', 0,   T,       valinit=st0)
#sI0      =  Slider(axI0,   'I0', 1,   1000,    valinit=I00)


def update(val):
    N     = sN.val
    betaN = sbeta.val
    beta  = betaN/N;
    delta = sdelta.val
    gamma = sgamma.val
    Ps    = sPs.val
    Pa    = sPa.val
    Pp    = sPp.val
#
    thetaT = sthetaT.val
    theta  = stheta.val
    Em   = sEm.val
    Ev   = sEv.val
    E    = sE.val
    #st    = sst.val
    #I0    = sI0.val

    # re-do the plot
    ax = subplot(211)
    # clear old plot
    cla()
    # model
    (S, I, V, P) = Py_model.funnel(T, N, beta, delta, gamma, Pp, Pa, Ps, \
                                thetaT, theta, Em, Ev, E, st0, I00);
    t = range(T)
    t = [ float(item) for item in t]
    RNFplot(t, S, I, V, dat, LOGLOGP)
    err = RSErr(T, I, dat);
    ptitle= "Funnel N=%2d beta=%.2e delta=%.2e gamma=%.2e" %\
        (N, beta, delta, gamma)
    title(ptitle)

    ax = subplot(212)
    cla()
    plot(t,P, lw=2, color='red')
    axis([1, T, min(P), max(P)])

    draw()

# GUI: buttons
Logax     = axes([0.6, 0.05, 0.1, 0.04])
buttonLog = Button(Logax, 'Log/Lin', color=axcolor, hovercolor=hvcolor)
resetax   = axes([0.7, 0.05, 0.1, 0.04])
button    = Button(resetax, 'Reset', color=axcolor, hovercolor=hvcolor)

def reset(event):
    sN.reset()
    sbeta.reset()
    sdelta.reset()
    sgamma.reset()
    sPs.reset()
    sPa.reset()
    sPp.reset()
#
    sthetaT.reset()
    stheta.reset()
    sEm.reset()
    sEv.reset()
    sE.reset()
    #sst.reset()
    #sI0.reset()


def logscale(event):
    global LOGLOGP
   
    if(LOGLOGP!=1):
        LOGLOGP+=1
    else:
        LOGLOGP=-1 

# event listeners
sbeta.on_changed(update)
sN.on_changed(update)
sgamma.on_changed(update)
sdelta.on_changed(update)
sPp.on_changed(update)
sPa.on_changed(update)
sPs.on_changed(update)
sthetaT.on_changed(update)
stheta.on_changed(update)
sEm.on_changed(update)
sEv.on_changed(update)
sE.on_changed(update)
#sst.on_changed(update)
#sI0.on_changed(update)


button.on_clicked(reset)
buttonLog.on_clicked(logscale)
buttonLog.on_clicked(update)

show()


