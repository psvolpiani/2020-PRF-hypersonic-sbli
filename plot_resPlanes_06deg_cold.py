#! /usr/bin/env python
# -*- coding:utf-8 -*-

# ====================================================================
# This script reads Hybrid restart files and plot flow quantities
# It works in python 3 -> python plot_resPlanes.py
# Author: P. S. Volpiani
# Date: 07/07/2022
# ====================================================================

import os, sys
import matplotlib as mpl
mpl.use('tkagg') # for Mac
import matplotlib.pyplot as plt
import numpy as np
import operator
import glob

from numpy import *
from matplotlib import *
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')
rc('lines', linewidth=1.5)
rc('font', size=16)

colormap = 'jet' #'RdBu'
c = 0

# File
cases = glob.glob("./*.res")

# Plot variable : rho, u, Cf, qw
plot_variable = "rho"

# Case dependent
deltaBL=1.6342291068;
TW=4.36;
TR=TW/0.8;
Rgas=0.02857;
muref=0.0000180;
xsepStar=-1.34824017279;
xreaStar=-0.949479274485;
xshock=40;
gamma=1.4;
Pr=0.7;
cp = gamma*Rgas/(gamma-1.);

for case in cases:
    
    print case
    
    file = case
    
    # Read Instanteneous.res!

    with open(file, 'rb') as f:
    
        # Read integers: precision, version, 0
        arg     = np.fromfile(f, count=3, dtype='int32')
        prec    = arg[0]; print "precision  = ", prec
        # Read integers: nx, ny, nz
        arg     = np.fromfile(f, count=3, dtype='int32')
        nx      = arg[0]; print "nx         = ", nx
        ny      = arg[1]; print "ny         = ", ny
        nz      = arg[2]; print "nz         = ", nz
        # Read integers: nvars, variables
        arg     = np.fromfile(f, count=1, dtype='int32')
        nvar    = arg[0]; print "nv         = ", nvar
        arg     = np.fromfile(f, count=nvar, dtype='int32')
        
        # Read floats: time and coordinates
        time    = np.fromfile(f, count=1, dtype='float64')
        X       = np.fromfile(f, count=nx, dtype='float64')
        Y       = np.fromfile(f, count=ny, dtype='float64')
        Z       = np.fromfile(f, count=nz, dtype='float64')
        #print Z
    
        # Quantities are stored using float32 format
        data    = np.fromfile(f, count=nx*ny*nz*nvar, dtype='float32')
        data    = data.reshape([nz, ny, nx, nvar], order='F');
        rho     = data[:,:,:,0]
        rhou    = data[:,:,:,1]
        rhov    = data[:,:,:,2]
        rhow    = data[:,:,:,3]
        rhoe    = data[:,:,:,4]
        u       = np.divide(rhou, rho)
        v       = np.divide(rhov, rho)
        w       = np.divide(rhow, rho)
        p       = (gamma-1.)*(rhoe-rho*(u**2+v**2+w**2)/2.); #print type(p)
        T       = p/rho/Rgas

        # Wall quantities
        Tw      = TW*np.ones((nz,nx));
        muw     = muref * Tw[:,:]**0.75 ;
        mu0     = muref * T[:,0,:]**0.75 ;
        muef    = ( muw[:,:] + mu0[:,:] )/2. ;
        tauw    = muef[:,:] * u[:,0,:] / Y[0] ;
        qw      = - cp * muef[:,:] / Pr  * ( T[:,0,:] - Tw[:,:]) / ( Y[0] - 0.) ;
        
        # Scaled coordinates by delta99 boundary layer at xshock
        delta0 = deltaBL;
        xstar  = np.zeros(nx); xstar[:] = (X[:]-xshock) / delta0 ;
        ystar  = np.zeros(ny); ystar[:] = Y[:] / delta0 ;
        zstar  = np.zeros(nz); zstar[:] = Z[:] / delta0 ;

        # Plot variable : rho, u, v, w, dila, vorz, p, Cf, qw, rhoy
        if (plot_variable=="rho"):
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,2.5));
            #axes.set_title(r"$\rho$")
            cax=axes.pcolor(xstar, ystar, rho[0,:,:], cmap=colormap)
            axes.set_xlabel(r"$(x - x_{imp})/\delta_{imp}$");
            axes.set_ylabel(r"$y/\delta_{imp}$");
            fig.tight_layout()
            plt.axis([-10., 10., 0, 5])
            #fig_name = case+"_rho.png" # pdf is much more expensive
            #fig.savefig(fig_name, dpi=400)
            #cbar = fig.colorbar(cax)
            plt.show()
    
        if (plot_variable=="u"):
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,2.5));
            #axes.set_title(r"$u$")
            cax=axes.pcolor(xstar, ystar, u[0,:,:], cmap=colormap)
            axes.set_xlabel(r"$(x - x_{imp})/\delta_{imp}$");
            axes.set_ylabel(r"$y/\delta_{imp}$");
            plt.axis([-10., 10., 0, 5])
            #axes.minorticks_on();
            fig.tight_layout()
            #fig_name = case+"_u.png" # pdf is much more expensive
            #fig.savefig(fig_name, dpi=400)
            #cbar = fig.colorbar(cax)
            plt.show()
        
        if (plot_variable=="Cf"):
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,2.5));
            #axes.set_title(r"$Cf$")
            rhoedge=1; Ue=1;
            cax=axes.pcolor(xstar, zstar, 2*tauw[:,:]/rhoedge/Ue**2, cmap=colormap)
            axes.axvline(xsepStar, color='k', linestyle='--')
            axes.axvline(xreaStar, color='k', linestyle='--')
            axes.set_xlabel(r"$(x - x_{imp})/\delta_{imp}$");
            axes.set_ylabel(r"$z/\delta_{imp}$");
            plt.axis([-10, 10, 0, zstar.max()])
            #axes.minorticks_on();
            fig.tight_layout()
            #fig_name = case+"_Cf.png" # pdf is much more expensive
            #fig.savefig(fig_name, dpi=400)
            #cbar = fig.colorbar(cax)
            plt.show()
        
        if (plot_variable=="qw"):
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,2.5));
            #axes.set_title(r"$q_w/(\rho_\infty u_\infty C_p T_r )$")
            rhoedge=1; Ue=1;
            cax=axes.pcolor(xstar, zstar, qw / ( rhoedge * Ue * cp * TR ), cmap=colormap)
            axes.axvline(xsepStar, color='k', linestyle='--')
            axes.axvline(xreaStar, color='k', linestyle='--')
            axes.set_xlabel(r"$(x - x_{imp})/\delta_{imp}$");
            axes.set_ylabel(r"$z/\delta_{imp}$");
            plt.axis([-10, 10, 0, zstar.max()])
            #axes.minorticks_on();
            fig.tight_layout()
            #fig_name = case+"_qw.png" # pdf is much more expensive
            #fig.savefig(fig_name, dpi=400)
            #cbar = fig.colorbar(cax)
            plt.show()
        
        c = c + 1
