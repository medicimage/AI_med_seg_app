#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nib
import os, sys
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import zoom
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import transforms
#import SimpleITK as sitk
from matplotlib.widgets import Slider, Button, RadioButtons


# read input image data including kidney ground-truth masks
# def readData4(patientName,subjectInfo,reconMethod,genBoundBox):
def readData4(img, reconMethod, genBoundBox, targetOrgan):

    seqNum = 1

    # reconstruction Method
    # scanner ---> 'SCAN'
    # grasp   ---> 'GRASP'


    if seqNum==1:
        im=img.get_data()
        im2 = np.zeros((im.shape[0],im.shape[1],im.shape[2],6))
        
        for ix in range(0,5):
            im2[:,:,:,ix]=im[:,:,:]

        if targetOrgan == 'Psoas':
            maskAddressL = './models/1qfr4i2cwCoGRrC2RDIKRu_LabelLeft__extract.nii'
            maskAddressR = './models/1qfr4i2cwCoGRrC2RDIKRu_LabelRight__extract.nii'
        elif targetOrgan == 'Pancreas':
            maskAddressL = ''
            maskAddressR = ''
        else:
            maskAddressL = ''
            maskAddressR = ''

        if seqNum==1:
            
            am = 1 
            
            if os.path.isfile(maskAddressL):
                lkm1= nib.load(maskAddressL);lkm=2*lkm1.get_data();
                lkm[lkm>2]=2;
            else:
                lkm=np.zeros(np.shape(im));

            if os.path.isfile(maskAddressR):
                rkm1= nib.load(maskAddressR);rkm=rkm1.get_data();
                rkm[rkm>1]=1;
            else:
                rkm=np.zeros(np.shape(im));
        else:
            lkm=0;rkm=0;am=0;

    else:
        lkm=0;rkm=0;am=0;


    boxes=[];
    if genBoundBox:
        aL=np.nonzero(lkm==2);
        aR=np.nonzero(rkm==1);

        if aL[0].size!=0:
            boxL=np.array([int((min(aL[0])+max(aL[0]))/2),int((min(aL[1])+max(aL[1]))/2),int((min(aL[2])+max(aL[2]))/2),\
              (max(aL[0])-min(aL[0])),(max(aL[1])-min(aL[1])),(max(aL[2])-min(aL[2]))])
        else:
            boxL=np.zeros((6,));
            
        if aR[0].size!=0:
            boxR=np.array([int((min(aR[0])+max(aR[0]))/2),int((min(aR[1])+max(aR[1]))/2),int((min(aR[2])+max(aR[2]))/2),\
              (max(aR[0])-min(aR[0])),(max(aR[1])-min(aR[1])),(max(aR[2])-min(aR[2]))])
        else:
            boxR=np.zeros((6,));
        
        boxes=np.vstack([np.array(boxR),np.array(boxL)]);
        
    oriKM = rkm+lkm;
    im2=(im2/np.amax(im2))*100;
    
    return im2, oriKM, boxes, rkm,lkm

# read nii file and return image volume
def readVolume4(img):

    im = img.get_data()

    im2 = np.zeros((im.shape[0], im.shape[1], im.shape[2], 6))

    for ix in range(0, 5):
        im2[:, :, :, ix] = im[:, :, :]

    im2 = (im2 / np.amax(im2)) * 100;

    return im2

def plotMask(fig, ax, img, mask, slice_i, view, organTarget):

    if view == 'AX':
        tm90 = mask[:, :, slice_i]
        tm90[tm90 >= 1] = 1
        masked = np.ma.masked_where(tm90 == 0, tm90)
        # colour map for ground-truth (red)
        if organTarget == 'Liver':
            cmapm = matplotlib.colors.ListedColormap(["red", "red", "red"], name='from_list', N=None)
            ax.imshow(masked, cmap=cmapm, interpolation='none', alpha=0.3)
            ax.contour(tm90, colors='red', linewidths=1.0)

        if organTarget == 'Kidneys':
            cmapm = matplotlib.colors.ListedColormap(["blue", "blue", "blue"], name='from_list', N=None)
            ax.imshow(masked, cmap=cmapm, interpolation='none', alpha=0.3)
            ax.contour(tm90, colors='blue', linewidths=1.0)

        if organTarget == 'Pancreas':
            cmapm = matplotlib.colors.ListedColormap(["green", "green", "green"], name='from_list', N=None)
            ax.imshow(masked, cmap=cmapm, interpolation='none', alpha=0.3)
            ax.contour(tm90, colors='green', linewidths=1.0)

        if organTarget == 'Psoas':
            cmapm = matplotlib.colors.ListedColormap(["yellow", "yellow", "yellow"], name='from_list', N=None)
            ax.imshow(masked, cmap=cmapm, interpolation='none', alpha=0.3)
            ax.contour(tm90, colors='yellow', linewidths=1.0)

    if view == 'CR':
        tm90 = mask[slice_i, :, :]
        tm90[tm90 >= 1] = 1
        masked = np.ma.masked_where(tm90 == 0, tm90)
        # colour map for ground-truth (red)
        if organTarget == 'Liver':
            cmapm = matplotlib.colors.ListedColormap(["red", "red", "red"], name='from_list', N=None)
            rotMasked = list(reversed(list(zip(*masked))))
            ax.imshow(rotMasked, cmap=cmapm, interpolation='none', alpha=0.3)
            rot_tm90 = list(reversed(list(zip(*tm90))))
            ax.contour(rot_tm90, colors='red', linewidths=1.0)

        if organTarget == 'Kidneys':
            cmapm = matplotlib.colors.ListedColormap(["blue", "blue", "blue"], name='from_list', N=None)
            rotMasked = list(reversed(list(zip(*masked))))
            ax.imshow(rotMasked, cmap=cmapm, interpolation='none', alpha=0.3)
            rot_tm90 = list(reversed(list(zip(*tm90))))
            ax.contour(rot_tm90, colors='blue', linewidths=1.0)

        if organTarget == 'Pancreas':
            cmapm = matplotlib.colors.ListedColormap(["green", "green", "green"], name='from_list', N=None)
            rotMasked = list(reversed(list(zip(*masked))))
            ax.imshow(rotMasked, cmap=cmapm, interpolation='none', alpha=0.3)
            rot_tm90 = list(reversed(list(zip(*tm90))))
            ax.contour(rot_tm90, colors='green', linewidths=1.0)

        if organTarget == 'Psoas':
            cmapm = matplotlib.colors.ListedColormap(["yellow", "yellow", "yellow"], name='from_list', N=None)
            rotMasked = list(reversed(list(zip(*masked))))
            ax.imshow(rotMasked, cmap=cmapm, interpolation='none', alpha=0.3)
            rot_tm90 = list(reversed(list(zip(*tm90))))
            ax.contour(rot_tm90, colors='yellow', linewidths=1.0)
    #
    if view == 'SG':
        tm90 = mask[:, slice_i, :]
        tm90[tm90 >= 1] = 1
        masked = np.ma.masked_where(tm90 == 0, tm90)
        # colour map for ground-truth (red)
        if organTarget == 'Liver':
            cmapm = matplotlib.colors.ListedColormap(["red", "red", "red"], name='from_list', N=None)
            rotMasked = list(reversed(list(zip(*masked))))
            ax.imshow(rotMasked, cmap=cmapm, interpolation='none', alpha=0.3)
            rot_tm90 = list(reversed(list(zip(*tm90))))
            ax.contour(rot_tm90, colors='red', linewidths=1.0)

        if organTarget == 'Kidneys':
            cmapm = matplotlib.colors.ListedColormap(["blue", "blue", "blue"], name='from_list', N=None)
            rotMasked = list(reversed(list(zip(*masked))))
            ax.imshow(rotMasked, cmap=cmapm, interpolation='none', alpha=0.3)
            rot_tm90 = list(reversed(list(zip(*tm90))))
            ax.contour(rot_tm90, colors='blue', linewidths=1.0)

        if organTarget == 'Pancreas':
            cmapm = matplotlib.colors.ListedColormap(["green", "green", "green"], name='from_list', N=None)
            rotMasked = list(reversed(list(zip(*masked))))
            ax.imshow(rotMasked, cmap=cmapm, interpolation='none', alpha=0.3)
            rot_tm90 = list(reversed(list(zip(*tm90))))
            ax.contour(rot_tm90, colors='green', linewidths=1.0)

        if organTarget == 'Psoas':
            cmapm = matplotlib.colors.ListedColormap(["yellow", "yellow", "yellow"], name='from_list', N=None)
            rotMasked = list(reversed(list(zip(*masked))))
            ax.imshow(rotMasked, cmap=cmapm, interpolation='none', alpha=0.3)
            rot_tm90 = list(reversed(list(zip(*tm90))))
            ax.contour(rot_tm90, colors='yellow', linewidths=1.0)

    return fig, ax

# plot image volume
def plotImage(img_vol, slice_i):
    selected_slice = img_vol[:, :, slice_i,1]
    fig, ax = plt.subplots()
    ax.imshow(selected_slice, 'gray', interpolation='none')
    return fig

