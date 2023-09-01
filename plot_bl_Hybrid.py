#! /usr/bin/env python
# -*- coding:utf-8 -*-

# ====================================================================
# This script reads Hybrid stats files and plot key quantities
# It works only in python 2 -> python2 plot_bl_Hybrid.py
# Author: P. S. Volpiani
# Date: 07/07/2022
# ====================================================================

import os, sys
import matplotlib as mpl
mpl.use('tkagg') # for Mac
import matplotlib.mathtext as mathtext
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import operator

from math import *
from pylab import *
from numpy import *
from matplotlib import *
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')
rc('lines', linewidth=1.5)
rc("font", size=16)
plt.rc('legend',**{'fontsize':14})
matplotlib.rcParams.update({'axes.labelsize': 20})

colormap  = 'jet' #'RdBu'
linestyle = ["-","--","-.",":", "--"]
cor       = ["b", "r", "g", "r", "g"]
print_figure = "no"
c = 0

# Files
cases = [
         "./twtr08/bl_twtr08-6-102files",    # 0.8 - High Re
        #"./twtr08/bl_twtr08-5-135files",    # 0.8 - Very High Re
         "./twtr19/bl_twtr19-5-142files",    # 1.9
         ]

# EXP
fileXP = "./EXP/Schlatter2010_Retheta1000.txt"
y1     = numpy.loadtxt(fileXP,skiprows=14, usecols=(0,))
yp1    = numpy.loadtxt(fileXP,skiprows=14, usecols=(1,))
up1    = numpy.loadtxt(fileXP,skiprows=14, usecols=(2,))
urmsp1 = numpy.loadtxt(fileXP,skiprows=14, usecols=(3,))
vrmsp1 = numpy.loadtxt(fileXP,skiprows=14, usecols=(4,))
wrmsp1 = numpy.loadtxt(fileXP,skiprows=14, usecols=(5,))
uvp1   = numpy.loadtxt(fileXP,skiprows=14, usecols=(6,))
markers1 = np.arange(1, len(yp1), 10)

fileXP = "./EXP/Schlatter2010_Retheta1410.txt"
y2     = numpy.loadtxt(fileXP,skiprows=14, usecols=(0,))
yp2    = numpy.loadtxt(fileXP,skiprows=14, usecols=(1,))
up2    = numpy.loadtxt(fileXP,skiprows=14, usecols=(2,))
urmsp2 = numpy.loadtxt(fileXP,skiprows=14, usecols=(3,))
vrmsp2 = numpy.loadtxt(fileXP,skiprows=14, usecols=(4,))
wrmsp2 = numpy.loadtxt(fileXP,skiprows=14, usecols=(5,))
uvp2   = numpy.loadtxt(fileXP,skiprows=14, usecols=(6,))
markers2 = np.arange(1, len(yp2), 10)

# Schulein (1996) results at section 6 - 356mm (Shock impinges at 350, so xplot should be 40)
yp3 = [ 12.,  24.,  36.,  48.,  60.,  72.,  84.,  108.,  132.,  168.,  192.,  227.,  263.,  311.,  371.,  431.,  491.,  551.,  611.,  670.,  730.,  850. ]
up3 = [ 9.37, 13.00, 14.00, 14.85, 15.44, 15.58, 15.99, 16.33, 17.03, 17.87, 18.37, 18.76, 19.10, 19.95, 20.61, 21.24, 21.68, 21.99, 22.21, 22.31, 22.32, 22.31 ]


for case in cases:
    
    print case
    
    file = case+".stats"
    
    if  (case=="./TwTr08/bl_twtr08-2-152files"):    # 0.8 - High Re
      muref=0.0000180 ; TW=4.36; Tr=TW/0.8; xplot=40.; xsp=65; dz=5./200.;
    elif  (case=="./twtr08/bl_twtr08-5-135files"):  # 0.8 - Very high Re 
      muref=0.0000090 ; TW=4.36; Tr=TW/0.8; xplot=40.; xsp=65; dz=5./340.;
    elif  (case=="./twtr08/bl_twtr08-6-102files"):  # 0.8 - Between high and very high Re test
      muref=0.0000120 ; TW=4.36; Tr=TW/0.8; xplot=40.; xsp=65; dz=5./300.;
    elif  (case=="./twtr19/bl_twtr19-5-142files"):  # 1.9 - Very high Re
      muref=0.0000120 ; TW=10.355; Tr=TW/1.9; xplot=40.; xsp=65; dz=5./140.;
    elif  (case=="../../SBLI-new/m228_twtr100_bl/bl_1536x384x128-1-242files"):    # Adi
      muref=0.000122 ; TW=1.920; Tr=TW/1.0; xplot=35.; xsp=60; dz=5./128;
    elif  (case=="../../SBLI-new/m228_twtr190_bl/case2_1536x288x128-5-600files"): # Hot
      muref=0.000120 ; TW=3.658; Tr=TW/1.9; xplot=35.; xsp=60; dz=5./128;
    elif  (case=="../../SBLI-new/m228_twtr050_bl/bl1_1800x480x256-1-240files"):   # Cold
      muref=0.000129 ; TW=0.963; Tr=TW/0.5; xplot=35.; xsp=60; dz=5./256;
    elif  (case=="../../SBLI-new/m228_twtr100_bl_hRe/bl_1800x480x256-1-241files"):# Adi - High Reynolds
      muref=0.000060 ; TW=1.920; Tr=TW/1.0; xplot=35.; xsp=60; dz=5./256;

    with open(file, 'rb') as f:
    
        #  Read until 3 consequtive %%%
        lastThree = f.read(3)
        while ( lastThree != ['%','%','%'] ):
            next = f.read(1); lastThree = [ lastThree[1], lastThree[2], next[0] ]
        # Read new line
        next = f.read(1)
        # Read integers nx, ny, nv
        arg = np.fromfile(f, count=3, dtype='int32')
        nx = arg[0]; ny = arg[1]; nv = arg[2];
        # Reed coordinates
        X   = np.fromfile(f, count=nx, dtype='float64')
        Y   = np.fromfile(f, count=ny, dtype='float64')
        # Read quantities
        avg = np.fromfile(f, count=nx*ny*nv, dtype='float64')
        avg = avg.reshape([nx, ny, nv], order='F');


    # Load variables
    r   = np.zeros((nx, ny));   r[:,:] = avg[:,:,0];
    p   = np.zeros((nx, ny));   p[:,:] = avg[:,:,4];
    u   = np.zeros((nx, ny));   u[:,:] = avg[:,:,10];
    v   = np.zeros((nx, ny));   v[:,:] = avg[:,:,11];
    w   = np.zeros((nx, ny));   w[:,:] = avg[:,:,12];
    T   = np.zeros((nx, ny));   T[:,:] = avg[:,:,13];
    uu  = np.zeros((nx, ny));  uu[:,:] = avg[:,:,21];
    vv  = np.zeros((nx, ny));  vv[:,:] = avg[:,:,22];
    ww  = np.zeros((nx, ny));  ww[:,:] = avg[:,:,23];
    TT  = np.zeros((nx, ny));  TT[:,:] = avg[:,:,24];
    uv  = np.zeros((nx, ny));  uv[:,:] = avg[:,:,25];
    uT  = np.zeros((nx, ny));  uT[:,:] = avg[:,:,28];
    vT  = np.zeros((nx, ny));  vT[:,:] = avg[:,:,29];


    # Compute/assume global quantities
    gamma = 1.4
    Pr = 0.7
    gasR = mean(mean(p[:,:]/r[:,:]/T[:,:]))
    cp = gamma*gasR/(gamma-1.)


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


    # Find reference location
    min_index, min_value = min(enumerate(abs(X[:]-xplot)), key=operator.itemgetter(1));
    iplot = min_index;


    # Compute integral quantities
    delta  = np.zeros(nx);
    dstar  = np.zeros(nx);
    theta  = np.zeros(nx);
    uvd    = np.zeros((nx, ny));
    ystar  = np.zeros((nx, ny));
    ustar  = np.zeros((nx, ny));
    for I in range(0, nx):


        # Find edge of boundary layer, erring on the side of a bit too much outside
        f = np.zeros(ny); f[:]=r[I,:] * ww[I,:]/abs(tauw[I]);
        max_index, max_value = max(enumerate(f), key=operator.itemgetter(1)); J = max_index;
        while (J<ny) and (f[J]>0.02): J=J+1;
        U  = np.zeros(ny);  U[:] = u[I,:] / u[I,J] ;
        RU = np.zeros(ny); RU[:] = r[I,:] * u[I,:] / r[I,J] / u[I,J] ;


        # Find delta99(I)
        J=0;
        while ( U[J]<0.99 ): J=J+1;
        J=J-1;
        delta[I] = Y[J] + ( Y[J+1] - Y[J] ) / ( U[J+1] - U[J] ) * ( 0.99 - U[J] ) ;


        # Find delta*(I)
        f[:] = 1. - RU[:] ;  fw = 1. ;
        dstar[I] = (fw+f[0])/2. * Y[0] ;
        J=0;
        while (Y[J]<delta[I]):
            J=J+1;
            dstar[I] = dstar[I] + mean(f[J-1:J]) * (Y[J]-Y[J-1]) ;


        # Find deltaTheta
        f[:] = RU[:] * (1.-U[:]) ;  fw = 0 ;
        theta[I] = (fw+f[0])/2. * Y[0] ;
        J=0;
        while (Y[J]<delta[I]):
            J=J+1;
            theta[I] = theta[I] + mean(f[J-1:J]) * (Y[J]-Y[J-1]) ;


        # Find Van Driest transformed velocities
        f[:] = 1/utau[I] * sqrt(r[I,:]/rhow[I]) ;  fw = 1/utau[I] ;
        uvd[I,0] = ( fw + f[0] )/2. * ( u[I,0] - 0. );
        for J in range(1, ny):
            uvd[I,J] = uvd[I,J-1] + mean(f[J-1:J]) * (u[I,J]-u[I,J-1]) ;


        # Find Trettel transformed velocities
        Te=1;
        mu = muref * (T[I,:]/Te)**0.75 ;
        ystar[I,:] = sqrt( r[I,:]*abs(tauw[I]) ) / mu[:] * Y[:] ;
        f = mu/abs(tauw[I]) ;  fw = muw[I]/abs(tauw[I]) ;
        ustar[I,0] = (fw+f[0])/2. * (ystar[I,0]-0.)/(Y[0]-0.) * u[I,0] ;
        for J in range(1, ny):
            dystardy = ( ystar[I,J] - ystar[I,J-1] ) / ( Y[J] - Y[J-1] ) ;
            ustar[I,J] = ustar[I,J-1] + mean(f[J-1:J]) * dystardy * (u[I,J]-u[I,J-1]) ;


    # Reynolds numbers
    Ue=1; rhoedge=1;
    Re_delta  = rhoedge * Ue * delta[:] / muref ;
    Re_dstar  = rhoedge * Ue * dstar[:] / muref ;
    Re_theta  = rhoedge * Ue * theta[:] / muref ;
    Re_delta2 = rhoedge * Ue * theta[:] / muw[:] ;
    Re_tau    = delta[:] / lv[:] ;
    Re_star   = delta[:] * sqrt(rhoedge * abs(tauw[:])) / muref ;


    # Compute dx+, dy+, dz+ at xplot
    dx  = ( X[iplot+1] - X[iplot] ) / lv[iplot]
    dy1 = ( Y[0] - 0. ) / lv[iplot]
    dyN = ( Y[-1]-Y[-2] ) / lv[iplot]
    dz  = dz / lv[iplot]

    # Print on screen
    print "nx        = ", arg[0]
    print "ny        = ", arg[1]
    print "delta99   = ", delta[iplot]
    print "delta*    = ", dstar[iplot]
    print "Re_delta  = ", Re_delta[iplot]
    print "Re_theta  = ", Re_theta[iplot]
    print "Re_delta2 = ", Re_delta2[iplot]
    print "Re_tau    = ", Re_tau[iplot]
    print "Re_star   = ", Re_star[iplot]
    print "dx+       = ", dx
    print "dy1+      = ", dy1
    print "dyN+      = ", dyN
    print "dz+       = ", dz
    print "tauw      = ", tauw[iplot]
    print "lv        = ", lv[iplot]

    # ======== #
    # Figure 1 #
    # ======== #

    if (c==0): fig1, axes1 = plt.subplots(figsize=(6,4))
    axes1.plot(X,delta, linestyle = '-' , color = cor[c]);
    axes1.plot(X,dstar, linestyle = '--', color = cor[c]);
    axes1.plot(X,theta, linestyle = ':' , color = cor[c]);
    
    axes1.set_xlabel( r"$x$");
    axes1.set_ylabel(r"$\delta_{99}, \; \delta^{*}, \; \delta_{\theta}$");
    fig1.tight_layout()
    grid(b=True, which='major', color='gray', linestyle=':')
    plt.minorticks_on()

    # ======== #
    # Figure 2 #
    # ======== #

    if (c==0): fig2, axes2 = plt.subplots(figsize=(6,4))
    axes2.plot(X,Re_theta, linestyle = '-' , color = cor[c]);
    axes2.plot(X,Re_tau,   linestyle = '--', color = cor[c]);
    axes2.plot(X,Re_delta2,  linestyle = ':' , color = cor[c]);

    axes2.set_xlabel( r"$x$");
    axes2.set_ylabel(r"$Re_{\theta}, \; Re_{\tau}, \; Re_{\delta_2}$");
    fig2.tight_layout()
    grid(b=True, which='major', color='gray', linestyle=':')
    plt.minorticks_on()

    # ======== #
    # Figure 3 #
    # ======== #

    if (c==0): fig3, axes3 = plt.subplots(figsize=(6,4))
    rhoedge=1; Ue=1;
    axes3.plot(X,2*tauw/rhoedge/Ue**2, linestyle = linestyle[c], color = cor[c]);

    axes3.set_xlabel( r"$x$");
    axes3.set_ylabel(r"$C_f$");
    fig3.tight_layout()
    grid(b=True, which='major', color='gray', linestyle=':')
    plt.minorticks_on()
    fig_name = "./fig_bl/bl_Cf.pdf"
    if (print_figure == "yes"): fig3.savefig(fig_name, dpi=300)

    # ======== #
    # Figure 4 #
    # ======== #

    if (c==0): fig4, axes4 = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True);

    axes4[0].plot(u[iplot,:], Y[:]/delta[iplot])
    axes4[0].set_ylim([0., 1.5])
    axes4[0].set_xlabel(r"$u$");
    axes4[0].set_ylabel(r"$y/\delta$");
    axes4[0].minorticks_on();
    axes4[0].xaxis.grid(b=True, which='major', color='gray', linestyle=':')
    axes4[0].yaxis.grid(b=True, which='major', color='gray', linestyle=':')

    axes4[1].plot(T[iplot,:], Y[:]/delta[iplot])
    axes4[1].set_xlabel(r"$T$");
    axes4[1].minorticks_on();
    axes4[1].xaxis.grid(b=True, which='major', color='gray', linestyle=':')
    axes4[1].yaxis.grid(b=True, which='major', color='gray', linestyle=':')
    fig4.tight_layout()
    
    # ======== #
    # Figure 5 #
    # ======== #

    if (c==0): fig5, axes5 = plt.subplots(figsize=(6, 4))
    t = np.arange(1, 10000, 1)
    tlin = np.arange(0.5, 12, 0.1)
    tlog = np.zeros(size(t));
    tlog[:] = 5.2+2.44*log(t[:]);

    #axes5.scatter(yp1[markers1], up1[markers1], color='r', marker='^', facecolors='white', linewidth='1')
    axes5.scatter(yp2[markers2], up2[markers2], color='k', marker='^', facecolors='white', linewidth='1')
    #axes5.scatter(yp3[:], up3[:], color='g', marker='s', facecolors='white', linewidth='1')

    axes5.semilogx(Y[:]/lv[iplot],uvd[iplot,:], linestyle = linestyle[c], color = cor[c]);
    axes5.semilogx(ystar[iplot,:],ustar[iplot,:]+5., linestyle = linestyle[c], color = cor[c]);

    axes5.semilogx(tlin, tlin, 'k:')
    axes5.semilogx(t[10:-1], tlog[10:-1], 'k:')

    axes5.semilogx(tlin, tlin+5., 'k:')
    axes5.semilogx(t[10:-1], tlog[10:-1]+5., 'k:')

    axes5.set_xlabel( r"$y^+ , \; y_{TL}^+$");
    axes5.set_ylabel(r"$u^+, \; u^+_{TL}$");
    axes5.set_xlim([0.5, 10000])
    fig5.tight_layout()
    grid(b=True, which='major', color='gray', linestyle=':')
    plt.minorticks_on()
    fig_name = "./fig_bl/bl_uvd_3.pdf"
    if (print_figure == "yes"): fig5.savefig(fig_name, dpi=300)

    # ======== #
    # Figure 6 #
    # ======== #

    if (c==0): fig6, axes6 = plt.subplots(figsize=(6, 4))
    x_min=0.; x_max=1.2;

    axes6.scatter(y2[markers2], urmsp2[markers2]**2, color='c', marker='o', facecolors='white', linewidth='1')
    axes6.scatter(y2[markers2], vrmsp2[markers2]**2, color='c', marker='o', facecolors='white', linewidth='1')
    axes6.scatter(y2[markers2], wrmsp2[markers2]**2, color='c', marker='o', facecolors='white', linewidth='1')
    axes6.scatter(y2[markers2], - uvp2[markers2]**2, color='c', marker='o', facecolors='white', linewidth='1')

    axes6.plot(Y[:]/delta[iplot],r[iplot,:]*uu[iplot,:]/tauw[iplot], linestyle = '-' , color = cor[c]);
    axes6.plot(Y[:]/delta[iplot],r[iplot,:]*vv[iplot,:]/tauw[iplot], linestyle = '--', color = cor[c]);
    axes6.plot(Y[:]/delta[iplot],r[iplot,:]*ww[iplot,:]/tauw[iplot], linestyle = '-.', color = cor[c]);
    axes6.plot(Y[:]/delta[iplot],r[iplot,:]*uv[iplot,:]/tauw[iplot], linestyle = ':' , color = cor[c]);

    axes6.set_xlabel( r"$y/\delta$");
    axes6.set_ylabel(r"$\overline{\rho} \widetilde{u''_i u''_j} / \tau_w$");
    axes6.set_xlim([x_min, x_max])
    axes6.xaxis.set_ticks(np.linspace(x_min, x_max, 7))
    axes6.yaxis.set_ticks(np.linspace(-2, 12, 8))
    fig6.tight_layout()
    grid(b=True, which='major', color='gray', linestyle=':')
    plt.minorticks_on()
    fig_name = "./fig_bl/bl_Rij_3.pdf"
    if (print_figure == "yes"): fig6.savefig(fig_name, dpi=300)

    # ======== #
    # Figure 7 #
    # ======== #

    if (c==0): fig7, axes7 = plt.subplots(figsize=(6,4))
    Te=1.; Ue=1;
    T_Wals = Tw[iplot]/Te + (Tr-Tw[iplot])/Te * u[iplot,:]/Ue + (Te-Tr)/Te * (u[iplot,:]/Ue) **2
    markers1 = np.arange(1, len(T_Wals), 10)
    axes7.scatter(u[iplot,markers1]/Ue,T_Wals[markers1], marker='o', facecolors='white', linewidth='1', color = cor[c]);

    Cf = 2.*tauw[iplot]/rhoedge/Ue**2
    Ch = qw[iplot] / ( rhoedge * Ue * cp * ( Tw[iplot] - Tr ) )
    s  = 2.*Ch/Cf
    if (Tw[iplot] == Tr):
        f_Zhang = ( u[iplot,:]/Ue )**2
    else:
        f_Zhang = ( 1 - s * Pr ) * ( u[iplot,:]/Ue )**2 + s * Pr * ( u[iplot,:]/Ue )
    T_Zhang = Tw[iplot]/Te + (Tr-Tw[iplot])/Te * f_Zhang + (Te-Tr)/Te * (u[iplot,:]/Ue) **2
    markers2 = np.arange(1, len(T_Zhang), 10)
    axes7.scatter(u[iplot,markers2]/Ue,T_Zhang[markers2], marker='*', facecolors='white', linewidth='1', color = cor[c]);

    axes7.plot(u[iplot,:]/Ue,T[iplot,:]/Te, linestyle = linestyle[c] , color = cor[c]);

    axes7.set_xlabel( r"$\overline{u}/u_\infty$");
    axes7.set_ylabel(r"$\overline{T}/T_\infty$");
    fig7.tight_layout()
    grid(b=True, which='major', color='gray', linestyle=':')
    plt.minorticks_on()
    fig_name = "./fig_bl/bl_Walz_Zhang_x40.pdf"
    if (print_figure == "yes"): fig7.savefig(fig_name, dpi=300)

    Bq = qw[iplot] / ( rhow[iplot] * utau[iplot] * cp * Tw[iplot] )
    print "Bq = ", Bq

    # ======== #
    # Figure 8 #
    # ======== #

    if (c==0): fig8, axes8 = plt.subplots(figsize=(6,4));
    Cf = 2.*tauw[:]/rhoedge/Ue**2
    St = qw[:] / ( rhoedge * Ue * cp * ( Tw[:] - Tr ) )
    s  = 2.*St/Cf
    axes8.plot(X,s, linestyle = linestyle[c] , color = cor[c]);
    axes8.minorticks_on();
    axes8.set_xlabel(r"$X$");
    axes8.set_ylabel(r"$2St/Cf$");
    axes8.xaxis.grid(b=True, which='major', color='gray', linestyle=':')
    axes8.yaxis.grid(b=True, which='major', color='gray', linestyle=':')
    fig8.tight_layout()

    # ======== #
    # Figure 9 #
    # ======== #

    if (c==0): fig9, axes9 = plt.subplots(figsize=(6,4));

    axes9.plot(Y[:]/delta[iplot], u[iplot,:], linestyle = linestyle[c], color = cor[c])
    axes9.set_xlim([0., 1.25])
    axes9.set_xlabel(r"$y/\delta$");
    axes9.set_ylabel(r"$u/u_\infty$");
    axes9.minorticks_on();
    axes9.xaxis.grid(b=True, which='major', color='gray', linestyle=':')
    axes9.yaxis.grid(b=True, which='major', color='gray', linestyle=':')
    fig9.tight_layout()
    fig_name = "./bl_uprofile.pdf"
    if (print_figure == "yes"): fig9.savefig(fig_name, dpi=300)

    # ========= #
    # Figure 10 # - viscosity
    # ========= #

    if (c==0): fig10, axes10 = plt.subplots(figsize=(6,4));

    mu  = np.zeros(ny);  mu[:] = muref * T[iplot,:]**0.75 ;
    ll  = mu / r[iplot,:] / utau[iplot]
    axes10.plot(Y[:]/delta[iplot], mu/muref, linestyle = linestyle[c], color = cor[c])
    axes10.set_xlim([0., 1.25])
    axes10.set_xlabel(r"$y/\delta$");
    axes10.set_ylabel(r"$\mu/\mu_{ref}$");
    axes10.minorticks_on();
    axes10.xaxis.grid(b=True, which='major', color='gray', linestyle=':')
    axes10.yaxis.grid(b=True, which='major', color='gray', linestyle=':')
    fig10.tight_layout()

    # ========= #
    # Figure 11 # - rho
    # ========= #

    if (c==0): fig11, axes11 = plt.subplots(figsize=(6,4));

    mu  = np.zeros(ny);  mu[:] = muref * T[iplot,:]**0.75 ;
    ll  = mu / r[iplot,:] / utau[iplot]
    axes11.plot(Y[:]/delta[iplot], r[iplot,:], linestyle = linestyle[c], color = cor[c])
    axes11.set_xlim([0., 1.25])
    axes11.set_xlabel(r"$y/\delta$");
    axes11.set_ylabel(r"$\rho/\rho_{\infty}$");
    axes11.minorticks_on();
    axes11.xaxis.grid(b=True, which='major', color='gray', linestyle=':')
    axes11.yaxis.grid(b=True, which='major', color='gray', linestyle=':')
    fig11.tight_layout()

    # ========= #
    # Figure 12 # - lv
    # ========= #

    if (c==0): fig12, axes12 = plt.subplots(figsize=(6,4));

    mu  = np.zeros(ny);  mu[:] = muref * T[iplot,:]**0.75 ;
    ll  = mu / r[iplot,:] / sqrt( abs(tauw[iplot]) / r[iplot,:] )
    axes12.plot(Y[:]/delta[iplot], ll/ll[-50], linestyle = linestyle[c], color = cor[c])
    axes12.set_xlim([0., 1.25])
    axes10.set_ylim([0., 5.])
    axes12.set_xlabel(r"$y/\delta$");
    axes12.set_ylabel(r"$l_v/l_{v,\infty}$");
    axes12.minorticks_on();
    axes12.xaxis.grid(b=True, which='major', color='gray', linestyle=':')
    axes12.yaxis.grid(b=True, which='major', color='gray', linestyle=':')
    fig12.tight_layout()
    print "lv/lv_infty = ", lv[iplot]/ll[-50]

    # ========= #
    # Figure 13 #
    # ========= #
    
    nc=len(cases);
    fig13, axes13 = plt.subplots(figsize=(9,3))
    cax=axes13.pcolor(X, Y, u.transpose(), cmap=colormap)
    axes13.contour(X, Y, u.transpose())
    #cax=axes13.pcolor(X, Y, p.transpose(), cmap=colormap)
    #axes13.contour(X, Y, u.transpose(), levels=[0], colors='white', linewidths = '1.')
    #axes13.contour(X, Y, Ma.transpose(), levels=[1], colors='black', linewidths = '1.')
    axes13.set_xlabel(r"$X$");
    axes13.set_ylabel(r"$Y$");
    cbar = fig13.colorbar(cax)
    plt.axis([0, 50, 0, 2])
    fig13.tight_layout()
    fig_name = "./bl_pavg.png" # pdf is much more expensive
    if (print_figure == "yes"): fig13.savefig(fig_name, dpi=400)


    c = c + 1

plt.show()

