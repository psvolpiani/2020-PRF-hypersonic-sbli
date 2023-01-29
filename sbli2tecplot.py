#! /usr/bin/env python
# -*- coding:utf-8 -*-

# ====================================================================
# This script reads Hybrid mean field and writes it in tecplot format
# It works only in python 2 -> python2 sbli2tecplot.py
# Author: P. S. Volpiani
# Date: 01/02/2018
# ====================================================================

import os, sys
import numpy as np
from numpy import *

# Files
cases = [
          "SBLI-M2/sbli_twtr19_95_1536x320x128-1-900files",
          "SBLI-M2/sbli_twtr19_80_1536x320x128-1-902files",
          "SBLI-M2/sbli_twtr19_65_1536x320x128-1-902files",
          "SBLI-M2/sbli_twtr19_50_1536x320x128-1-902files",
#
          "SBLI-M2/sbli_twtr10_11_1536x384x128-1-899files",
          "SBLI-M2/sbli_twtr10_95_1536x384x128-1-901files",
          "SBLI-M2/sbli_twtr10_80_1536x384x128-1-901files",
          "SBLI-M2/sbli_twtr10_65_1536x384x128-1-901files",
          "SBLI-M2/sbli_twtr10_50_1536x384x128-1-901files",
#
          "SBLI-M5/sbli_twtr08_14deg-3-302files", #400-700
          "SBLI-M5/sbli_twtr08_10deg-2-302files",
          "SBLI-M5/sbli_twtr08_06deg-2-302files",
          "SBLI-M5/sbli_twtr19_14deg-1-602files",
          "SBLI-M5/sbli_twtr19_10deg-1-302files",
          "SBLI-M5/sbli_twtr19_06deg-1-302files",
         ]

c = 1
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
      muref=0.0000180; TW=4.36; TR=TW/0.80; p3=13.62; x0=30; xshock=40; xsp=65; deltaBL =1.6342291068; star40 =0.767677064565
    elif(case=="SBLI-M5/sbli_twtr08_10deg-2-302files"): # Cold 10deg Moderate Reynolds
      muref=0.0000180; TW=4.36; TR=TW/0.80; p3=7.626; x0=30; xshock=40; xsp=65; deltaBL =1.6342291068; star40 =0.767677064565
    elif(case=="SBLI-M5/sbli_twtr08_06deg-2-302files"): # Cold 06deg Moderate Reynolds
      muref=0.0000180; TW=4.36; TR=TW/0.80; p3=3.762; x0=30; xshock=40; xsp=65; deltaBL =1.6342291068; star40 =0.767677064565
    elif(case=="SBLI-M5/sbli_twtr19_14deg-1-602files"): # Hot 14deg High Reynolds
      muref=0.0000120; TW=10.355; TR=TW/1.90; p3=13.62; x0=22; xshock=40; xsp=65; deltaBL =1.65821952886; star40 =0.945198259248
    elif(case=="SBLI-M5/sbli_twtr19_10deg-1-302files"): # Hot 10deg High Reynolds
      muref=0.0000120; TW=10.355; TR=TW/1.90; p3=7.626; x0=24; xshock=40; xsp=65; deltaBL =1.65821952886; star40 =0.945198259248
    elif(case=="SBLI-M5/sbli_twtr19_06deg-1-302files"): # Hot 06deg High Reynolds
      muref=0.0000120; TW=10.355; TR=TW/1.90; p3=3.762; x0=24; xshock=40; xsp=65; deltaBL =1.65821952886; star40 =0.945198259248
    

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

    name = case.replace('/', "-")
    print (name)
    # Write tecplot file
    tecplot_file = open("./tecplot_sol_"+name+".dat","w")

    if ('SBLI-M2' in case):
      tecplot_file.write('# Reference : "Volpiani, P. S., Bernardini, M., & Larsson, J. (2018). \n')
      tecplot_file.write('#              Effects of a nonadiabatic wall on supersonic shock/boundary-layer interactions. \n')
      tecplot_file.write('#              Physical Review Fluids, 3(8), 083401."\n')
    else:
      tecplot_file.write('# Reference : "Volpiani, P. S., Bernardini, M., & Larsson, J. (2020). \n')
      tecplot_file.write('#              Effects of a nonadiabatic wall on hypersonic shock/boundary-layer interactions. \n')
      tecplot_file.write('#              Physical Review Fluids, 5(1), 014602."\n')

    tecplot_file.write('TITLE="DNS solution for SBLI"\n')
    tecplot_file.write('VARIABLES ="X","Y","RHO","U","V","P","T","MU"\n')
    tecplot_file.write('Zone I=' + '{:>10}'.format(str(nx)) + ',J=' + '{:>10}'.format(str(ny)) )
    tecplot_file.write(', F=POINT\n')
    for j in xrange(len(ystar)):
        for i in xrange(len(xstar)):
            tecplot_file.write('{:>20}'.format( str(X[i])       ) )
            tecplot_file.write('{:>20}'.format( str(Y[j])       ) )
            tecplot_file.write('{:>20}'.format( str(r[i,j])     ) )
            tecplot_file.write('{:>20}'.format( str(u[i,j])     ) )
            tecplot_file.write('{:>20}'.format( str(v[i,j])     ) )
            tecplot_file.write('{:>20}'.format( str(p[i,j])     ) )
            tecplot_file.write('{:>20}'.format( str(T[i,j])     ) )
            tecplot_file.write('{:>20}'.format( str(muref * T[i,j]**0.75 ) ) )
            tecplot_file.write( '\n' )
    tecplot_file.close()

    c = c + 1
