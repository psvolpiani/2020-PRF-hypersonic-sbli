#! /usr/bin/env python
# -*- coding:utf-8 -*-

# ====================================================================
# This script reads Hybrid mean field and plot key quantities
# It works only in python 2 -> python2 plot_statistics.py
# Author: P. S. Volpiani
# Date: 01/02/2018
# ====================================================================

import os, sys
import matplotlib as mpl
mpl.use('tkagg') # for Mac
import matplotlib.pyplot as plt
import numpy as np
import operator

from numpy import *
from matplotlib import *
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')
rc('lines', linewidth=1.5)
rc('font', size=16)

# Files
cases = [
#          "SBLI-M2/sbli_twtr19_95_1536x320x128-1-900files",
#          "SBLI-M2/sbli_twtr19_80_1536x320x128-1-902files",
#          "SBLI-M2/sbli_twtr19_65_1536x320x128-1-902files",
#          "SBLI-M2/sbli_twtr19_50_1536x320x128-1-902files",
#
#          "SBLI-M2/sbli_twtr10_11_1536x384x128-1-899files",
#          "SBLI-M2/sbli_twtr10_95_1536x384x128-1-901files",
#          "SBLI-M2/sbli_twtr10_80_1536x384x128-1-901files",
#          "SBLI-M2/sbli_twtr10_65_1536x384x128-1-901files",
#          "SBLI-M2/sbli_twtr10_50_1536x384x128-1-901files",
#
          "SBLI-M5/sbli_twtr08_14deg-3-302files", #400-700
#          "SBLI-M5/sbli_twtr08_10deg-2-302files",
#          "SBLI-M5/sbli_twtr08_06deg-2-302files",
          "SBLI-M5/sbli_twtr19_14deg-1-602files",
#          "SBLI-M5/sbli_twtr19_10deg-1-302files",
#          "SBLI-M5/sbli_twtr19_06deg-1-302files",
         ]

colormap = 'jet' #'RdBu'
linestyle = [":","--"]
cor = ["b", "r"]
c = 0

for case in cases:
    
    print (case)
    
    file = case+".stats"
    
    if  (case=="SBLI-M2/sbli_twtr19_95_1536x320x128-1-900files"): # Hot - 9.5°
        muref=0.000120; TW=3.658; TR=TW/1.90; p3=2.86; x0=20; xshock=35; xsp=60; deltaBL=1.613; tauwBL35=0.00136853468032;
    elif(case=="SBLI-M2/sbli_twtr19_80_1536x320x128-1-902files"): # Hot - 8.0°
        muref=0.000120; TW=3.658; TR=TW/1.90; p3=2.46; x0=20; xshock=35; xsp=60; deltaBL=1.613; tauwBL35=0.00136853468032;
    elif(case=="SBLI-M2/sbli_twtr19_65_1536x320x128-1-902files"): # Hot - 6.5°
        muref=0.000120; TW=3.658; TR=TW/1.90; p3=2.10; x0=20; xshock=35; xsp=60; deltaBL=1.613; tauwBL35=0.00136853468032;
    elif(case=="SBLI-M2/sbli_twtr19_50_1536x320x128-1-902files"): # Hot - 5.0°
        muref=0.000120; TW=3.658; TR=TW/1.90; p3=1.79; x0=20; xshock=35; xsp=60; deltaBL=1.613; tauwBL35=0.00136853468032;
    elif(case=="SBLI-M2/sbli_twtr19_35_1536x320x128-1-901files"): # Hot - 3.5°
        muref=0.000120; TW=3.658; TR=TW/1.90; p3=1.51; x0=20; xshock=35; xsp=60; deltaBL=1.613; tauwBL35=0.00136853468032;
    elif(case=="SBLI-M2/sbli_twtr10_11_1536x384x128-1-899files"): # Adi - 11.°
        muref=0.000122; TW=1.920; TR=TW/1.00; p3=3.32; x0=20; xshock=35; xsp=60; deltaBL=1.541; tauwBL35=0.00156763831991;
    elif(case=="SBLI-M2/sbli_twtr10_95_1536x384x128-1-901files"): # Adi - 9.5°
        muref=0.000122; TW=1.920; TR=TW/1.00; p3=2.86; x0=20; xshock=35; xsp=60; deltaBL=1.541; tauwBL35=0.00156763831991;
    elif(case=="SBLI-M2/sbli_twtr10_80_1536x384x128-1-901files"): # Adi - 8.0°
        muref=0.000122; TW=1.920; TR=TW/1.00; p3=2.46; x0=20; xshock=35; xsp=60; deltaBL=1.541; tauwBL35=0.00156763831991;
    elif(case=="SBLI-M2/sbli_twtr10_65_1536x384x128-1-901files"): # Adi - 6.5°
        muref=0.000122; TW=1.920; TR=TW/1.00; p3=2.10; x0=20; xshock=35; xsp=60; deltaBL=1.541; tauwBL35=0.00156763831991;
    elif(case=="SBLI-M2/sbli_twtr10_50_1536x384x128-1-901files"): # Adi - 5.0°
        muref=0.000122; TW=1.920; TR=TW/1.00; p3=1.79; x0=20; xshock=35; xsp=60; deltaBL=1.541; tauwBL35=0.00156763831991;
    elif(case=="SBLI-M2/sbli_twtr10_95_2048x490x170-1-602files"): # Adi - 9.5° (fine)
        muref=0.000122; TW=1.920; TR=TW/1.00; p3=2.86; x0=20; xshock=35; xsp=60; deltaBL=1.541; tauwBL35=0.00156763831991;
    elif(case=="SBLI-M2/sbli_twtr05_95_1800x480x256-5-900files"): # Cold - 9.5°
        muref=0.000129; TW=0.963; TR=TW/0.50; p3=2.86; x0=20; xshock=35; xsp=60; deltaBL=1.548; tauwBL35=0.00165067142201;
    elif(case=="SBLI-M2/sbli_twtr05_80_1800x480x256-5-901files"): # Cold - 8.0°
        muref=0.000129; TW=0.963; TR=TW/0.50; p3=2.46; x0=20; xshock=35; xsp=60; deltaBL=1.548; tauwBL35=0.00165067142201;
    elif(case=="SBLI-M2/sbli_twtr10_95_1800x480x256-HRe-900files"): # Adiab - 9.5° - High Re
      muref=0.00006; TW=1.920; TR=TW/1.00; p3=2.86; x0=20; xshock=35; xsp=60; deltaBL=1.46; #tauwBL35=0.;
      
    elif(case=="SBLI-M5/sbli_twtr08_14deg-3-302files"): # Cold 14deg Moderate Reynolds
      muref=0.0000180; TW=4.36; TR=TW/0.80; p3=13.62; x0=21; xshock=40; xsp=65; deltaBL =1.6342291068; star40 =0.767677064565
    elif(case=="SBLI-M5/sbli_twtr08_10deg-2-302files"): # Cold 10deg Moderate Reynolds
      muref=0.0000180; TW=4.36; TR=TW/0.80; p3=7.626; x0=21; xshock=40; xsp=65; deltaBL =1.6342291068; star40 =0.767677064565
    elif(case=="SBLI-M5/sbli_twtr08_06deg-2-302files"): # Cold 06deg Moderate Reynolds
      muref=0.0000180; TW=4.36; TR=TW/0.80; p3=3.762; x0=21; xshock=40; xsp=65; deltaBL =1.6342291068; star40 =0.767677064565
    elif(case=="SBLI-M5/sbli_twtr19_14deg-1-602files"): # Hot 14deg High Reynolds
      muref=0.0000120; TW=10.355; TR=TW/1.90; p3=13.62; x0=21; xshock=40; xsp=65; deltaBL =1.65821952886; star40 =0.945198259248
    elif(case=="SBLI-M5/sbli_twtr19_10deg-1-302files"): # Hot 10deg High Reynolds
      muref=0.0000120; TW=10.355; TR=TW/1.90; p3=7.626; x0=21; xshock=40; xsp=65; deltaBL =1.65821952886; star40 =0.945198259248
    elif(case=="SBLI-M5/sbli_twtr19_06deg-1-302files"): # Hot 06deg High Reynolds
      muref=0.0000120; TW=10.355; TR=TW/1.90; p3=3.762; x0=21; xshock=40; xsp=65; deltaBL =1.65821952886; star40 =0.945198259248
    

    with open(file, 'rb') as f:
    
        #  Read until 3 consequtive %%%
        lastThree = f.read(3)
        while ( lastThree != ['%','%','%'] ):
            next = f.read(1); lastThree = [ lastThree[1], lastThree[2], next[0] ]
        # Read new line
        next = f.read(1)
        # Read integers nx, ny, nv
        arg = np.fromfile(f, count=3, dtype='int32')
        nx = arg[0]; ny = arg[1]; nv = arg[2]; print (arg)
        # Reed coordinates
        X   = np.fromfile(f, count=nx, dtype='float64')
        Y   = np.fromfile(f, count=ny, dtype='float64')
        # Read quantities
        avg = np.fromfile(f, count=nx*ny*nv, dtype='float64')
        avg = avg.reshape([nx, ny, nv], order='F');


    # Load variables
    r   = np.zeros((nx, ny));   r[:,:] = avg[:,:,0];
    p   = np.zeros((nx, ny));   p[:,:] = avg[:,:,4];
    dila= np.zeros((nx, ny));dila[:,:] = avg[:,:,9];
    u   = np.zeros((nx, ny));   u[:,:] = avg[:,:,10];
    v   = np.zeros((nx, ny));   v[:,:] = avg[:,:,11];
    w   = np.zeros((nx, ny));   w[:,:] = avg[:,:,12];
    T   = np.zeros((nx, ny));   T[:,:] = avg[:,:,13];
    rr  = np.zeros((nx, ny));  rr[:,:] = avg[:,:,15];
    pp  = np.zeros((nx, ny));  pp[:,:] = avg[:,:,16];
    uu  = np.zeros((nx, ny));  uu[:,:] = avg[:,:,21];
    vv  = np.zeros((nx, ny));  vv[:,:] = avg[:,:,22];
    ww  = np.zeros((nx, ny));  ww[:,:] = avg[:,:,23];
    TT  = np.zeros((nx, ny));  TT[:,:] = avg[:,:,24];
    uv  = np.zeros((nx, ny));  uv[:,:] = avg[:,:,25];
    rx  = np.zeros((nx, ny));  rx[:,:] = avg[:,:,70];


    # Compute/assume global quantities
    gamma = 1.4
    Pr = 0.7
    gasR = mean(mean(p[:,:]/r[:,:]/T[:,:]))
    cp = gamma*gasR/(gamma-1.)
    
    # Compute local Mach number
    Ma = u[:,:] / sqrt(gamma * gasR * T[:,:])
    
    # Compute turbulent kinetic energy
    kk = 0.5 * ( uu[:,:] + vv[:,:]+ ww[:,:] );

    # Scaled coordinates by delta99 boundary layer at xshock
    delta0 = deltaBL;
    xstar  = np.zeros(nx); xstar[:] = (X[:]-xshock) / delta0 ;
    ystar  = np.zeros(ny); ystar[:] = Y[:] / delta0 ;
    
    # Compute wall quantities
    if (TW>0.001): Tw = TW*np.ones(nx)
    else: Tw = T[:,0]

    # Dynamic viscosity at the wall
    muw = muref * Tw[:]**0.75 ;
    # Dynamic viscosity at first cell
    mu0 = muref * T[:,0]**0.75 ;
    # Mean dynamic viscosity
    muef = ( muw[:] + mu0[:] )/2. ;
    # Wall shear stress
    tauw = muef[:] * u[:,0] / Y[0] ;
    # Heat transfer at the wall
    qw = - cp * muef[:] / Pr  * ( T[:,0] - Tw[:]) / ( Y[0] - 0.) ;
    # Density at the wall
    rhow = p[:,0] / Tw[:] / gasR ;
    # Friction velocity
    utau = sqrt( abs(tauw[:]) / rhow[:] ) ;
    # Viscous length scale
    lv = muw[:] / rhow[:] / utau[:] ;

    # Find index of reference location
    min_index, min_value = min(enumerate(abs(X[:]-x0)), key=operator.itemgetter(1)); I = min_index;

    # Find edge of boundary layer, erring on the side of a bit too much outside
    f=r[I,:] * ww[I,:]/abs(tauw[I]);
    max_index, max_value = max(enumerate(f), key=operator.itemgetter(1)); J = max_index;
    while (J<ny) and (f[J]>0.02): J=J+1;
    U  = np.zeros(ny);  U[:] = u[I,:] / u[I,J] ;
    RU = np.zeros(ny); RU[:] = r[I,:] * u[I,:] / r[I,J] / u[I,J] ;

    # Find delta0 = delta99(I)
    J=0;
    while ( U[J]<0.99 ): J=J+1;
    J=J-1;
    delta0 = Y[J] + ( Y[J+1] - Y[J] ) / ( U[J+1] - U[J] ) * ( 0.99 - U[J] ) ;
    print "delta0     = ", delta0

    # Find delta*
    f[:] = 1. - RU[:] ;  fw = 1. ;
    dstar0 = (fw+f[0])/2. * Y[0] ;
    J=0;
    while (Y[J]<delta0):
        J=J+1;
        dstar0 = dstar0 + mean(f[J-1:J]) * (Y[J]-Y[J-1]) ;
    print "deltaStar  = ", dstar0

    # Find deltaTheta
    f[:] = RU[:] * (1.-U[:]) ;  fw = 0 ;
    theta0 = (fw+f[0])/2. * Y[0] ;
    J=0;
    while (Y[J]<delta0):
        J=J+1;
        theta0 = theta0 + mean(f[J-1:J]) * (Y[J]-Y[J-1]) ;
    print "deltaTheta = ", theta0
    
    # Scale coordinates by delta99 boundary layer at xshock
    delta0 = deltaBL;
    xstar = (X[:]-xshock) / delta0 ;
    ystar = Y[:] / delta0 ;
    
    # ======== #
    # Figure 1 #
    # ======== #
 
    if (c==0): fig1, axes1 = plt.subplots(figsize=(6,4))
    rhoedge=1; Ue=1;
    axes1.plot(xstar,2*tauw/rhoedge/Ue**2, linestyle = linestyle[c], color = cor[c]);
    axes1.set_xlabel( r"$(x - x_{imp})/\delta_{imp}$"); axes1.set_ylabel(r"$C_f$");
    axes1.set_xlim([-15, 15]); plt.minorticks_on()
    fig1.tight_layout()
    
    # ======== #
    # Figure 2 #
    # ======== #
    
    if (c==0): fig2, axes2 = plt.subplots(figsize=(6,4))
    axes2.plot(xstar,p[:,0]/gasR, linestyle = linestyle[c], color = cor[c]);
    axes2.plot((-15, 0), (1, 1), 'k-'); axes2.plot((0, 15), (p3, p3), 'k-'); axes2.plot((0, 0), (1, p3), 'k-')
    axes2.set_xlabel( r"$(x - x_{imp})/\delta_{imp}$"); axes2.set_ylabel(r"$p_w/p_\infty$")
    axes2.set_xlim([-15, 15]); plt.minorticks_on()
    fig2.tight_layout()
    
    # ======== #
    # Figure 3 #
    # ======== #

    if (c==0): fig3, axes3 = plt.subplots(figsize=(6,4))
    axes3.plot(xstar,sqrt(pp[:,0])/gasR, linestyle = linestyle[c], color = cor[c]);
    axes3.set_xlabel( r"$(x - x_{imp})/\delta_{imp}$"); axes3.set_ylabel(r"$p_{rms}/p_\infty$")
    axes3.set_xlim([-15, 15]); plt.minorticks_on()
    fig3.tight_layout()
    
    # ======== #
    # Figure 4 #
    # ======== #

    if (c==0): fig4, axes4 = plt.subplots(figsize=(6,4))
    St = qw[:] / ( rhoedge * Ue * cp * ( Tw[:] - TR ) )
    axes4.plot(xstar,St, linestyle = linestyle[c], color = cor[c]);
    axes4.set_xlabel( r"$(x - x_{imp})/\delta_{imp}$"); axes4.set_ylabel(r"$S_{t}$");
    axes4.set_xlim([-15, 15]); plt.minorticks_on()
    fig4.tight_layout()
    
    # ======== #
    # Figure 5 #
    # ======== #
    
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,2.5));
    axes.set_title(r"$u$")
    cax=axes.pcolor(xstar, ystar, u.transpose(), cmap=colormap)
    axes.contour(xstar, ystar, u.transpose(), levels=[0], colors='white')
    axes.contour(xstar, ystar, dila.transpose(), levels=[-0.15])
    axes.contour(xstar, ystar, Ma.transpose(), levels=[1], colors='black')
    axes.set_xlabel(r"$(x - x_{imp})/\delta_{imp}$"); axes.set_ylabel(r"$y/\delta_{imp}$");
    plt.axis([-10., 10., 0, 5])
    
    
    c = c + 1
plt.show()
