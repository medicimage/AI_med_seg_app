import funcs_ha_use
import numpy as np
from sklearn.decomposition import PCA
import scipy
from scipy.ndimage import zoom
from scipy import signal
from skimage import morphology
from networks_ah import get_unet2, get_rbunet, get_meshNet, get_denseNet, calculatedPerfMeasures
from networks_ah import get_denseNet103, get_unet3
import streamlit as st
from keras import backend as K
import gc


reconMethod = 'SCAN';

def get_largest_component(image):
    """
    get the largest component from 2D or 3D binary image
    image: nd array
    """

    dim = len(image.shape)
    if (image.sum() == 0):
        print('the largest component is null')
        return image
    if (dim == 2):
        s = scipy.ndimage.generate_binary_structure(2, 1)
    elif (dim == 3):
        s = scipy.ndimage.generate_binary_structure(3, 1)
    else:
        raise ValueError("the dimension number should be 2 or 3")
    labeled_array, numpatches = scipy.ndimage.label(image, s)
    sizes = scipy.ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    output = np.asarray(labeled_array == max_label, np.uint8)
    return output


def singlePatientDetection(pName, baseline, params, organTarget):

    tDim = params['tDim'];
    deepRed = params['deepReduction'];
    PcUsed = params['PcUsed'];
    
    st.warning('Step 3')

    ##### extract input image data (vol4D00)
    vol4D00,_,_,_,_ = funcs_ha_use.readData4(pName,reconMethod,0,organTarget);
    zDimOri = vol4D00.shape[2];
    
    st.warning('Step 4')
    
    # start from baseline      
    im = vol4D00[:,:,:,baseline:];
    
    im=im/np.nanmean(im);
    
    vol4D0 = np.copy(im);

    # perform PCA to numPC 
    numPC = 5; #50
    pca = PCA(n_components=numPC);

    vol4Dvecs = np.reshape(vol4D0, (vol4D0.shape[0] * vol4D0.shape[1] * vol4D0.shape[2], vol4D0.shape[3]));
    PCs=pca.fit_transform(vol4Dvecs);
    vol4Dpcs=np.reshape(PCs, (vol4D0.shape[0],vol4D0.shape[1],vol4D0.shape[2], numPC));
    del PCs
    st.warning('Step 5')
    dpcs = np.copy(vol4Dpcs);
    dpcs=dpcs/dpcs.max();
    da = dpcs.T;

    # downsample to 64 x 64 x 64 in x-y-z-dimenions
    # dsFactor = 3.5; 
    zDim = 64; yDim = 64; zDim = 64;
    # im0 = zoom(da,(1,zDim/da.shape[1],1/dsFactor,1/dsFactor),order=0);
    im0 = zoom(da,(1,zDim/da.shape[1],yDim/da.shape[2],zDim/da.shape[3]),order=0);
    
    sx = 0; xyDim = 64; 
    DataTest=np.zeros((1,zDim,xyDim,xyDim,tDim));
    DataTest[sx,:,:,:,:]=np.swapaxes(im0.T,0,2);
    #initialise detection model
    n_channels = tDim; n_classes = 3;

    #choose relevant detection model
    networkToUse = params['networkToUseDetect'];
    if networkToUse == 'rbUnet':
        model = get_rbunet(xyDim,zDim,n_channels,n_classes,deepRed,0);
    elif networkToUse == 'Unet':
        model = get_unet3(xyDim,zDim,n_channels,n_classes,deepRed,0);
    elif networkToUse == 'denseNet':
        model = get_denseNet(xyDim,zDim,n_channels,n_classes,deepRed,0);
    elif networkToUse == 'tNet':
        model = get_denseNet103(xyDim,zDim,n_channels,n_classes,deepRed,0); 

    #load detection model weights
    selectedEpoch=params['selectedEpochDetect'];
    st.warning('Step 4')
    # select organ to segment
    K.clear_session()
    if organTarget == 'Liver':
        model.load_weights('./models/detect3D_30000_Liver.h5');
    elif organTarget == 'Pancreas':
        model.load_weights('./models/detect3D_50000_Pancreas.h5')
    elif organTarget == 'Psoas':
        model.load_weights('./models/detect3D_32755_Psoas.h5')
    elif organTarget == 'Kidneys':
        model.load_weights('./models/detect3D_46000_Kidneys.h5')

    #### perform prediction ####
    imgs_mask_test= model.predict(DataTest, verbose=1);
    K.clear_session()
    gc.collect()
    multiHead = 0;
    if multiHead:
        labels_pred=np.argmax(imgs_mask_test[0], axis=4)
    else:
        labels_pred=np.argmax(imgs_mask_test, axis=4)

    ##### generate bounding boxes from coarse segmentation #####
    si = 0;    
    
    left = labels_pred[si,:,:,:].T==2;
    left = left.astype(int);
    
    right = labels_pred[si,:,:,:].T==1;
    right = right.astype(int);
    st.warning('step p5')
    ####### resample to original input image size dimensions
    
    # xyDimOri = 224;
    zDimm = vol4D00.shape[2];
    xDim = vol4D00.shape[0];
    yDim = vol4D00.shape[1];

    KMR = zoom(right,(xDim/np.size(right,0),yDim/np.size(right,1),zDimm/np.size(right,2)),order=0);
    KML = zoom(left,(xDim/np.size(left,0),yDim/np.size(left,1),zDimm/np.size(left,2)),order=0);
            
    
    if np.sum(KMR) != 0:
        KMR=morphology.remove_small_objects(KMR.astype(bool), min_size=256,in_place=True).astype(int);
        KMR = KMR.astype(int);
    if np.sum(KML) != 0:
        KML=morphology.remove_small_objects(KML.astype(bool), min_size=256,in_place=True).astype(int);   
        KML = KML.astype(int);
        KML[KML>=1]=2;
    #full kidney mask
    maskDetect = KMR + KML;
    print ('maskDetect:')
    print (maskDetect.shape)

    ### generate kidneys bounding box based on prediction
    boxDetect = [];

    aL=np.nonzero(KML==2);
    aR=np.nonzero(KMR==1);
    st.warning('Step 8_1')
    if aL[0].size!=0:
        boxL=np.array([int((min(aL[0])+max(aL[0]))/2),int((min(aL[1])+max(aL[1]))/2),int((min(aL[2])+max(aL[2]))/2),\
          (max(aL[0])-min(aL[0])),(max(aL[1])-min(aL[1])),(max(aL[2])-min(aL[2]))])
    else:
        boxL=np.zeros((6,));
    st.warning('Step 8_2')   
    if aR[0].size!=0:
        boxR=np.array([int((min(aR[0])+max(aR[0]))/2),int((min(aR[1])+max(aR[1]))/2),int((min(aR[2])+max(aR[2]))/2),\
          (max(aR[0])-min(aR[0])),(max(aR[1])-min(aR[1])),(max(aR[2])-min(aR[2]))])
    else:
        boxR=np.zeros((6,));
    st.warning('Step 8_3')
    # bounding box for right (boxDetect[0,:]) and left kidney (boxDetect[1,:])
    boxDetect=np.vstack([np.array(boxR),np.array(boxL)]);
    st.warning('Step 8_4')
    # identify whether right kidney exists
    # identify whether left kidney exists
    kidneyNone=np.nonzero(np.sum(boxDetect,axis=1)==0); #right/left
    if kidneyNone[0].size!=0:
        kidneyNone=np.nonzero(np.sum(boxDetect,axis=1)==0)[0][0]; #right/left
    st.warning('Step 8_5')
    # add extra margins to minimise impact of false-negative predictions
    KM = np.copy(maskDetect); KM[KM>1]=1;
    xSafeMagin=10;ySafeMagin=10;zSafeMagin=3;
    if boxDetect[0,2]+boxDetect[0,5]+3 >= KM.shape[2] or boxDetect[0,2]+boxDetect[0,5]-3 <0:
        boxDetect[:,[3,4,5]]=boxDetect[:,[3,4,5]]+[xSafeMagin,ySafeMagin,0];
    else:
        boxDetect[:,[3,4,5]]=boxDetect[:,[3,4,5]]+[xSafeMagin,ySafeMagin,zSafeMagin];
    st.warning('Step p7')

#     # predMaskR=np.zeros((1,xyDimOri,xyDimOri,zDimOri));
#     # predMaskL=np.zeros((1,xyDimOri,xyDimOri,zDimOri));
#     predMaskR=np.zeros((1,xDim,yDim,zDimm));
#     predMaskL=np.zeros((1,xDim,yDim,zDimm));
#     st.warning('Step 8_7')
#     sc = 0;
#     predMaskR[sc,:,:,:]=KMR; 
#     predMaskL[sc,:,:,:]=KML;    

#     Masks2Save={};

#     predMaskR2=zoom(predMaskR[sc,:,:,:],(1,1,1),order=0);
#     predMaskL2=zoom(predMaskL[sc,:,:,:],(1,1,1),order=0);
    
#     Masks2Save['R']=np.copy(predMaskR2.astype(float));
#     Masks2Save['L']=np.copy(predMaskL2.astype(float));
    
#     #### write kidney masks to file ####    
#     #funcs_ha_use.writeMasksDetect(pName,reconMethod,Masks2Save,1);
    st.warning('Step 9') 
    
    return maskDetect, boxDetect, kidneyNone, vol4D0, vol4Dpcs, zDimOri


def singlePatientDetectionPancreas(pName, baseline, params, organTarget):

    tDim = params['tDim'];
    tDim = 1
    deepRed = params['deepReduction'];
    PcUsed = params['PcUsed'];
    st.warning('stepP1')
    ##### extract input image data (vol4D00)
    vol4D00, _, _, _, _ = funcs_ha_use.readData4(pName, reconMethod, 0, organTarget);
    zDimOri = vol4D00.shape[2];
    im5 = vol4D00[:, :, :, baseline:];
    st.warning('stepP1.1')
    medianFind = np.median(im5);
    if medianFind == 0:
        medianFind = 1.0;
    st.warning('stepP1.1')
    im5 = im5/medianFind;

    vol4D05 = np.copy(im5);
    st.warning('stepP1.1.1')
    # perform PCA to numPC
    numPC05 = 5;  # 50
    pca05 = PCA(n_components=numPC05);
    st.warning('stepP1.1.2')
    vol4Dvecs05 = np.reshape(vol4D05, (vol4D05.shape[0] * vol4D05.shape[1] * vol4D05.shape[2], vol4D05.shape[3]));
    st.warning('stepP1.1.3')
    del PCs
    PCs05 = pca05.fit_transform(vol4Dvecs05);
    st.warning('stepP1.1.4')
    vol4Dpcs05 = np.reshape(PCs05, (vol4D05.shape[0], vol4D05.shape[1], vol4D05.shape[2], numPC05));
    st.warning('stepP2')
    arr = vol4D00[:, :, :, 0]
    im = np.expand_dims(arr, 3)

    medianFind = np.median(im);
    if medianFind == 0:
        medianFind = 1.0;
    im = im / medianFind;

    vol4D0 = np.copy(im);

    # perform PCA to numPC
    numPC = 1;  # 50
    pca = PCA(n_components=numPC);
    vol4Dvecs = np.reshape(vol4D0, (vol4D0.shape[0] * vol4D0.shape[1] * vol4D0.shape[2], vol4D0.shape[3]));
    PCs = pca.fit_transform(vol4Dvecs);
    vol4Dpcs = np.reshape(PCs, (vol4D0.shape[0], vol4D0.shape[1], vol4D0.shape[2], numPC));

    dpcs = np.copy(vol4Dpcs);
    dpcs = dpcs / dpcs.max();
    da = dpcs.T;

    # downsample to 64 x 64 x 64 in x-y-z-dimenions
    # dsFactor = 3.5;
    zDim = 64;
    yDim = 64;
    zDim = 64;

    im0 = zoom(da, (1, zDim / da.shape[1], yDim / da.shape[2], zDim / da.shape[3]), order=0);

    sx = 0;
    xyDim = 64;
    DataTest = np.zeros((1, zDim, xyDim, xyDim, tDim));
    DataTest[sx, :, :, :, :] = np.swapaxes(im0.T, 0, 2);

    # initialise detection model
    n_channels = tDim;
    n_classes = 3;
    K.clear_session()
    # choose relevant detection model
    networkToUse = params['networkToUseDetect'];
    if networkToUse == 'rbUnet':
        model = get_rbunet(xyDim, zDim, n_channels, n_classes, deepRed, 0);
    elif networkToUse == 'Unet':
        model = get_unet3(xyDim, zDim, n_channels, n_classes, deepRed, 0);
    elif networkToUse == 'denseNet':
        model = get_denseNet(xyDim, zDim, n_channels, n_classes, deepRed, 0);
    elif networkToUse == 'tNet':
        model = get_denseNet103(xyDim, zDim, n_channels, n_classes, deepRed, 0);

        # load detection model weights
    selectedEpoch = params['selectedEpochDetect'];

    # select organ to segment
    if organTarget == 'Liver':
        model.load_weights('./models/detect3D_30000_Liver.h5');
    elif organTarget == 'Pancreas':
        model.load_weights('./models/detect3D_3114_Pancreas.h5')
    elif organTarget == 'Psoas':
        model.load_weights('./models/detect3D_32755_Psoas.h5')
    st.warning('stepP3')

    #### perform prediction ####
    imgs_mask_test = model.predict(DataTest, verbose=1);
    K.clear_session()
    gc.collect()
    multiHead = 0;
    if multiHead:
        labels_pred = np.argmax(imgs_mask_test[0], axis=4)
    else:
        labels_pred = np.argmax(imgs_mask_test, axis=4)


    ##### generate bounding boxes from coarse segmentation #####
    si = 0;

    left = labels_pred[si, :, :, :].T == 2;
    left = left.astype(int);

    right = labels_pred[si, :, :, :].T == 1;
    right = right.astype(int);

    ####### resample to original input image size dimensions

    # xyDimOri = 224;
    zDimm = vol4D00.shape[2];
    xDim = vol4D00.shape[0];
    yDim = vol4D00.shape[1];
    st.warning('stepP4')

    KMR = zoom(right, (xDim / np.size(right, 0), yDim / np.size(right, 1), zDimm / np.size(right, 2)), order=0);
    KML = zoom(left, (xDim / np.size(left, 0), yDim / np.size(left, 1), zDimm / np.size(left, 2)), order=0);

    if np.sum(KMR) != 0:
        KMR = morphology.remove_small_objects(KMR.astype(bool), min_size=256, in_place=True).astype(int);
        KMR = KMR.astype(int);
    if np.sum(KML) != 0:
        KML = morphology.remove_small_objects(KML.astype(bool), min_size=256, in_place=True).astype(int);
        KML = KML.astype(int);
        KML[KML >= 1] = 2;

    nMax = 50  # 50
    KMR[0:nMax, :, :] = 0
    KMR[KMR.shape[0] - nMax - 1:KMR.shape[0], :, :] = 0
    KMR[:, 0:nMax, :] = 0
    KMR[:, KMR.shape[1] - nMax - 1:KMR.shape[1], :] = 0


    KML[0:nMax, :, :] = 0
    KML[KML.shape[0] - nMax - 1:KML.shape[0], :, :] = 0
    KML[:, 0:nMax, :] = 0
    KML[:, KML.shape[1] - nMax - 1:KML.shape[1], :] = 0


    KMR = get_largest_component(KMR.astype(np.uint8))
    KML = get_largest_component(KML.astype(np.uint8))

    KML[KML == 1] = 2
    KMR[KMR == 1] = 0

    # full kidney mask
    maskDetect = KMR + KML;
    st.warning('stepP5')
    ### generate kidneys bounding box based on prediction
    boxDetect = [];

    aL = np.nonzero(KML == 2);
    aR = np.nonzero(KMR == 1);

    if aL[0].size != 0:
        boxL = np.array(
            [int((min(aL[0]) + max(aL[0])) / 2), int((min(aL[1]) + max(aL[1])) / 2), int((min(aL[2]) + max(aL[2])) / 2), \
             (max(aL[0]) - min(aL[0])), (max(aL[1]) - min(aL[1])), (max(aL[2]) - min(aL[2]))])
    else:
        boxL = np.zeros((6,));

    if aR[0].size != 0:
        boxR = np.array(
            [int((min(aR[0]) + max(aR[0])) / 2), int((min(aR[1]) + max(aR[1])) / 2), int((min(aR[2]) + max(aR[2])) / 2), \
             (max(aR[0]) - min(aR[0])), (max(aR[1]) - min(aR[1])), (max(aR[2]) - min(aR[2]))])
    else:
        boxR = np.zeros((6,));

    # bounding box for right (boxDetect[0,:]) and left kidney (boxDetect[1,:])
    boxDetect = np.vstack([np.array(boxR), np.array(boxL)]);

    # identify whether right kidney exists
    # identify whether left kidney exists
    kidneyNone = np.nonzero(np.sum(boxDetect, axis=1) == 0);  # right/left
    if kidneyNone[0].size != 0:
        kidneyNone = np.nonzero(np.sum(boxDetect, axis=1) == 0)[0][0];  # right/left
    st.warning('stepP6')
    # add extra margins to minimise impact of false-negative predictions
    KM = np.copy(maskDetect);
    KM[KM > 1] = 1;

    ratioCast = np.zeros((21, 7))
    ratioCast[:, 0] = ratioCast[:, 0] + [7, 115, 9, 5, 10, 35, 36, 37, 40, 49, 64, 69, 80, 89, 116, 125, 129, 138, 610, 607, 100000]
    ratioCast[:, 1] = ratioCast[:, 1] + [0.98, 1.01, 0.98, 1.09, 1.01, 1.08, 0.93, 0.85, 1.07, 1.06, 0.96, 0.94, 1.02, 1.02, 0.87, 1.02, 0.89, 1.02, 1.02, 1.09, 1.02]
    ratioCast[:, 2] = ratioCast[:, 2] + [0.83, 0.97, 1.03, 1.05, 0.91, 1.13, 0.96, 0.88, 0.98, 1.00, 1.00, 1.06, 0.85, 1.01, 1.00, 0.97, 1.0, 1.01, 0.95, 1.03, 1.00]
    ratioCast[:, 3] = ratioCast[:, 3] + [0.63, 1.12, 0.81, 0.8, 1.15, 0.72, 1.04, 0.82, 0.84, 0.85, 1.23, 1.11, 0.85, 0.71, 0.85, 0.64, 0.93, 1.04, 0.92, 1.15, 0.85]
    ratioCast[:, 4] = ratioCast[:, 4] + [0.61, 0.64, 0.58, 0.64, 0.43, 0.86, 0.37, 0.69, 0.74, 1.01, 0.65, 0.92, 0.53, 0.60, 0.57, 0.63, 0.74, 0.75, 0.74, 0.89, 0.65]
    ratioCast[:, 5] = ratioCast[:, 5] + [0.69, 0.78, 0.78, 0.91, 0.60, 1.18, 0.41, 0.64, 0.51, 0.96, 0.93, 0.76, 0.55, 0.77, 0.71, 0.60, 0.69, 0.94, 0.70, 0.74, 0.73]
    ratioCast[:, 6] = ratioCast[:, 6] + [0.45, 0.86, 0.61, 0.73, 0.63, 0.43, 0.39, 0.63, 0.66, 0.89, 0.67, 0.63, 0.65, 0.53, 0.65, 0.68, 0.67, 0.93, 0.61, 0.71, 0.65]

    #items0 = pName.split('_')
    #items = [n for n in items0[1].split('-')]
    fileClass = 115

    chooseCast0 = np.where(ratioCast[:, 0] == fileClass)
    if not any(map(len, chooseCast0)):
        fileClass0 = 100000
        chooseCast0 = np.where(ratioCast[:, 0] == fileClass0)

    chooseCast = ratioCast[chooseCast0, 1:]
    chooseCast = np.array(chooseCast).flatten()

    boxDetect[:, 0] = chooseCast[0] * boxDetect[:, 0]
    boxDetect[:, 1] = chooseCast[1] * boxDetect[:, 1]
    boxDetect[:, 2] = chooseCast[2] * boxDetect[:, 2]
    boxDetect[:, 3] = chooseCast[3] * boxDetect[:, 3]
    boxDetect[:, 4] = chooseCast[4] * boxDetect[:, 4]
    boxDetect[:, 5] = chooseCast[5] * boxDetect[:, 5]
    st.warning('stepP7')
    boxDetect = boxDetect.astype('int')

#     KM = np.copy(maskDetect);
#     KM[KM > 1] = 1;
#     xSafeMagin = 15;
#     ySafeMagin = 15;
#     zSafeMagin = 3;

#     boxDetect[1, [3, 4, 5]] = boxDetect[1, [3, 4, 5]] + [xSafeMagin, ySafeMagin, zSafeMagin];

#     predMaskR = np.zeros((1, xDim, yDim, zDimm));
#     predMaskL = np.zeros((1, xDim, yDim, zDimm));

#     sc = 0;
#     predMaskR[sc, :, :, :] = KMR;
#     predMaskL[sc, :, :, :] = KML;

#     Masks2Save = {};

#     predMaskR2 = zoom(predMaskR[sc, :, :, :], (1, 1, 1), order=0);
#     predMaskL2 = zoom(predMaskL[sc, :, :, :], (1, 1, 1), order=0);


#     #### write kidney masks to file ####
#     # funcs_ha_use.writeMasksDetect(pName,reconMethod,Masks2Save,1);
    
    return maskDetect, boxDetect, kidneyNone, vol4D0, vol4Dpcs, zDimOri, vol4Dpcs05

def singlePatientSegmentation(params, pName, maskDetect, boxDetect, kidneyNone, vol4D0, vol4Dpcs, zDimOri, organTarget, vol4Dpcs05):

    tDim = params['tDim'];
    deepRed = params['deepReduction'];
    PcUsed = params['PcUsed'];
    
    dx = 64; dy = 64; dz = 64;
    Box = np.copy(boxDetect);
    maskDetect[maskDetect>1]=1;

    if organTarget != 'Pancreas':
        vol4Dpcs05 = vol4Dpcs
    st.warning('Step 10')
    # crop out kidney images using bounding boxes
    exv = 0; #Jennifer Nowlan (+5, L)
    if kidneyNone!=0:
        croppedData4DR_pcs=vol4Dpcs05[int(Box[0,0]-int(Box[0,3]/2)+exv):int(Box[0,0]+int(Box[0,3]/2)+exv),\
                                int(Box[0,1]-int(Box[0,4]/2)+exv):int(Box[0,1]+int(Box[0,4]/2)+exv),\
                                int(Box[0,2]-int(Box[0,5]/2)+exv):int(Box[0,2]+int(Box[0,5]/2)+exv),:];
        croppedData4DR=vol4D0[int(Box[0,0]-int(Box[0,3]/2)+exv):int(Box[0,0]+int(Box[0,3]/2)+exv),\
                                int(Box[0,1]-int(Box[0,4]/2)+exv):int(Box[0,1]+int(Box[0,4]/2)+exv),\
                                int(Box[0,2]-int(Box[0,5]/2)+exv):int(Box[0,2]+int(Box[0,5]/2)+exv),:];
        
        croppedData4DR_pcs=zoom(croppedData4DR_pcs,(dx/np.size(croppedData4DR_pcs,0),dy/np.size(croppedData4DR_pcs,1),dz/np.size(croppedData4DR_pcs,2),1),order=0);
        croppedData4DR=zoom(croppedData4DR,(dx/np.size(croppedData4DR,0),dy/np.size(croppedData4DR,1),dz/np.size(croppedData4DR,2),1),order=0);
    
    if kidneyNone!=1:    
        croppedData4DL_pcs=vol4Dpcs05[int(Box[1,0]-int(Box[1,3]/2)+exv):int(Box[1,0]+int(Box[1,3]/2)-exv),\
                                int(Box[1,1]-int(Box[1,4]/2)+exv):int(Box[1,1]+int(Box[1,4]/2)-exv),\
                                int(Box[1,2]-int(Box[1,5]/2)+exv):int(Box[1,2]+int(Box[1,5]/2)-exv),:]; 
            
        croppedData4DL=vol4D0[int(Box[1,0]-int(Box[1,3]/2)+exv):int(Box[1,0]+int(Box[1,3]/2)-exv),\
                                int(Box[1,1]-int(Box[1,4]/2)+exv):int(Box[1,1]+int(Box[1,4]/2)-exv),\
                                int(Box[1,2]-int(Box[1,5]/2)+exv):int(Box[1,2]+int(Box[1,5]/2)-exv),:];  

        croppedData4DL_pcs=zoom(croppedData4DL_pcs,(dx/np.size(croppedData4DL_pcs,0),dy/np.size(croppedData4DL_pcs,1),dz/np.size(croppedData4DL_pcs,2),1),order=0);
        croppedData4DL=zoom(croppedData4DL,(dx/np.size(croppedData4DL,0),dy/np.size(croppedData4DL,1),dz/np.size(croppedData4DL,2),1),order=0);
        
    if kidneyNone==0:
        d=np.concatenate((croppedData4DL[np.newaxis,:,:,:,:],croppedData4DL[np.newaxis,:,:,:,:]),axis=0);
        dpcs=np.concatenate((croppedData4DL_pcs[np.newaxis,:,:,:,:],croppedData4DL_pcs[np.newaxis,:,:,:,:]),axis=0);
    elif kidneyNone==1:
        d=np.concatenate((croppedData4DR[np.newaxis,:,:,:,:],croppedData4DR[np.newaxis,:,:,:,:]),axis=0);
        dpcs=np.concatenate((croppedData4DR_pcs[np.newaxis,:,:,:,:],croppedData4DR_pcs[np.newaxis,:,:,:,:]),axis=0);
    else:
        d=np.concatenate((croppedData4DR[np.newaxis,:,:,:,:],croppedData4DL[np.newaxis,:,:,:,:]),axis=0);
        dpcs=np.concatenate((croppedData4DR_pcs[np.newaxis,:,:,:,:],croppedData4DL_pcs[np.newaxis,:,:,:,:]),axis=0);
        
    d=d/d.max()
    dpcs=dpcs/dpcs.max();
    st.warning('Step 11')
    sc=0; n_channels = tDim;
    DataCroppedTest=np.zeros((2,dx,dy,dz,n_channels));
    DataCroppedTest[2*sc:2*sc+2,:,:,:,:]=dpcs;
    
    #choose relevant segmentation model
    n_classes = 2;
    networkToUse = params['networkToUseSegment'];

    if networkToUse == 'tNet':
        model = get_denseNet103(dx,dz,n_channels,n_classes,deepRed,0);
    elif networkToUse == 'rbUnet':
        model = get_rbunet(dx,dz,n_channels,n_classes,deepRed,0);
    elif networkToUse == 'Unet':
        model = get_unet3(dx,dz,n_channels,n_classes,deepRed,0);
    elif networkToUse == 'denseNet':
        model = get_denseNet(dx,dz,n_channels,n_classes,deepRed,0);
    
    #load segmentation model weights
    selectedEpoch=params['selectedEpochSegment'];
    # select organ to segment
    if organTarget == 'Liver':
        model.load_weights('./models/croppedSeg3D_31735_Liver.h5');
    elif organTarget == 'Pancreas':
        model.load_weights('./models/croppedSeg3D_84000_Pancreas.h5');
    elif organTarget == 'Psoas':
        model.load_weights('./models/croppedSeg3D_96000_Psoas.h5');
    elif organTarget == 'Kidneys':
        model.load_weights('./models/croppedSeg3D_84000_Kidneys.h5')
    st.warning('Step 12')
    # perform prediction
    cropped_mask_test = model.predict(DataCroppedTest, verbose=1)
    K.clear_session()
    gc.collect()
    if cropped_mask_test.min()<0:
        cropped_mask_test=abs(cropped_mask_test.min())+cropped_mask_test;
        
    imgs_mask_test2=np.copy(cropped_mask_test);
    imgs_mask_test2[:,:,:,:,0]=cropped_mask_test[:,:,:,:,0];
    imgs_mask_test2[:,:,:,:,1]=cropped_mask_test[:,:,:,:,1];
    labels_pred_2=np.argmax(imgs_mask_test2, axis=4);
    
    # insert predicted kidney masks into relevant positions in 
    # original image spatial dimensions
    # xyDim=224;

    xDim = vol4D0.shape[0]
    yDim = vol4D0.shape[1]
            
    predMaskR=np.zeros((1,xDim,yDim,zDimOri));
    predMaskL=np.zeros((1,xDim,yDim,zDimOri));
    
    if kidneyNone!=0:
        Rk=labels_pred_2[2*sc,:,:,:]
        croppedData4DR=signal.resample(Rk,int(Box[0,3]), t=None, axis=0);
        croppedData4DR=signal.resample(croppedData4DR,int(Box[0,4]), t=None, axis=1);
        croppedData4DR=signal.resample(croppedData4DR,int(Box[0,5]), t=None, axis=2);
        croppedData4DR[croppedData4DR>0.5]=2;croppedData4DR[croppedData4DR<0.5]=0
        croppedData4DR[croppedData4DR==0]=1;croppedData4DR[croppedData4DR==2]=0  
        
        predMaskR[sc,int(Box[0,0]-Box[0,3]/2):int(Box[0,0]+Box[0,3]/2),\
                            int(Box[0,1]-Box[0,4]/2):int(Box[0,1]+Box[0,4]/2),\
                            int(Box[0,2]-Box[0,5]/2):int(Box[0,2]+Box[0,5]/2)]=croppedData4DR;
                        

    if kidneyNone!=1:     
            Lk=labels_pred_2[2*sc+1,:,:,:]
            croppedData4DL=signal.resample(Lk,int(Box[1,3]), t=None, axis=0);
            croppedData4DL=signal.resample(croppedData4DL,int(Box[1,4]), t=None, axis=1);
            croppedData4DL=signal.resample(croppedData4DL,int(Box[1,5]), t=None, axis=2);
            croppedData4DL[croppedData4DL>0.5]=2; croppedData4DL[croppedData4DL<0.5]=0
            croppedData4DL[croppedData4DL==0]=1;croppedData4DL[croppedData4DL==2]=0    
            
            predMaskL[sc,int(Box[1,0]-Box[1,3]/2):int(Box[1,0]+Box[1,3]/2),\
                                int(Box[1,1]-Box[1,4]/2):int(Box[1,1]+Box[1,4]/2),\
                                int(Box[1,2]-Box[1,5]/2):int(Box[1,2]+Box[1,5]/2)]=croppedData4DL;    
        
        
    if np.sum(predMaskR) != 0:
            predMaskL=morphology.remove_small_objects(predMaskL.astype(bool), min_size=256,in_place=True).astype(int);
    if np.sum(predMaskL) != 0:
            predMaskR=morphology.remove_small_objects(predMaskR.astype(bool), min_size=256,in_place=True).astype(int);
    
    predMaskL2=np.copy(predMaskL);
        
    Masks2Save={};
        
    predMaskR2=zoom(predMaskR[sc,:,:,:],(1,1,1),order=0);
    predMaskL2=zoom(predMaskL[sc,:,:,:],(1,1,1),order=0);
    maskSegment = predMaskR2 + predMaskL2;
        
#     Masks2Save['R']=np.copy(predMaskR2.astype(float));
#     Masks2Save['L']=np.copy(predMaskL2.astype(float));

#     # write kidney segmentation masks to file
#     #funcs_ha_use.writeMasks(pName,reconMethod,Masks2Save,1);
   
    return maskSegment

