import detectCroppedSeg3DKerasDR_predict_ha
import streamlit as st
import funcs_ha_use
import numpy as np
from streamlit import caching


def runDeepSegmentationModel(organTarget, img):

    # model parameters
    params = {};
    params['TestSetNum'] = 1;
    params['tpUsed'] = 50;
    params['tDim'] = params['tpUsed'];
    params['PcUsed'] = 1;
    # params['visualizeResults']= 0;
    # params['visSlider']= 0;
    params['deepReduction'] = 0;

    params['networkToUseDetect'] = 'rbUnet'  # 'denseNet'; #'tNet'; #'Unet' #meshNet
    params['networkToUseSegment'] = 'tNet'  # 'denseNet'; #'rbUnet' # 'Unet' #meshNet

    if params['PcUsed'] == 1:
        tDim = 5;
        params['tDim'] = tDim;

    baseline = '1';
    reconMethod = 'SCAN';
    if organTarget == 'Liver':
        
        params['selectedEpochDetect'] = '30000';
        params['selectedEpochSegment'] = '31735';
        st.warning('step 1')
        # call the model to detect and segment and return the mask
        maskDetect, boxDetect, kidneyNone, vol4D0, vol4Dpcs, zDimOri = detectCroppedSeg3DKerasDR_predict_ha.singlePatientDetection(img, int(baseline),
                                                                                              params, 'Liver');
        st.warning('step 2')
#         if (np.sum(maskDetect)==0):
#             st.warning('No organ detected')
#             maskSegment = None
#             plotMask = N2ne
#         else:
        plotMask = detectCroppedSeg3DKerasDR_predict_ha.singlePatientSegmentation(params, img, maskDetect, boxDetect, kidneyNone, vol4D0, vol4Dpcs,
                                                zDimOri, 'Liver', None);

    if organTarget == 'Kidneys':
        params['selectedEpochDetect'] = '46000';
        params['selectedEpochSegment'] = '84000';
        # call the model to detect and segment and return the mask
        maskDetect, boxDetect, kidneyNone, vol4D0, vol4Dpcs, zDimOri = detectCroppedSeg3DKerasDR_predict_ha.singlePatientDetection(img, int(baseline),
                                                                                              params, 'Kidneys');
        if (np.sum(maskDetect)==0):
            st.warning('No organ detected')
            maskSegment = None
            plotMask = None
        else:
            plotMask = detectCroppedSeg3DKerasDR_predict_ha.singlePatientSegmentation(params, img, maskDetect, boxDetect, kidneyNone, vol4D0, vol4Dpcs,
                                                zDimOri, 'Kidneys', None);

    if organTarget == 'Pancreas':
     
        params['selectedEpochDetect'] = '3114';
        params['selectedEpochSegment'] = '84000';

        # call the model to detect and segment and return the mask
        maskDetect, boxDetect, kidneyNone, vol4D0, vol4Dpcs, zDimOri, vol4Dpcs05 = detectCroppedSeg3DKerasDR_predict_ha.singlePatientDetectionPancreas(img, int(baseline), params, 'Pancreas');
#         if (np.sum(maskDetect)==0):
#             st.warning('No organ detected')
#             maskSegment = None
#             plotMask = None
#         else:
        plotMask = detectCroppedSeg3DKerasDR_predict_ha.singlePatientSegmentation(params, img, maskDetect,
                                                                                               boxDetect, kidneyNone,
                                                                                               vol4D0, vol4Dpcs,
                                                                                               zDimOri, 'Pancreas', vol4Dpcs05);

    if organTarget == 'Psoas':

        params['selectedEpochDetect'] = '32755';
        params['selectedEpochSegment'] = '96000';
        vol4D00, oriKM, boxDetect0, _, _ = funcs_ha_use.readData4(img, reconMethod, 1, 'Psoas');

        maskDetect0 = oriKM.copy()
        kidneyNone0 = np.nonzero(np.sum(boxDetect0, axis=1) == 0);  # right/left
        if kidneyNone0[0].size != 0:
            kidneyNone0 = np.nonzero(np.sum(boxDetect0, axis=1) == 0)[0][0];  # right/left

        # call the model to detect and segment and return the mask
        maskDetect, boxDetect, kidneyNone, vol4D0, vol4Dpcs, zDimOri = detectCroppedSeg3DKerasDR_predict_ha.singlePatientDetection(img, int(baseline),
                                                                                              params, 'Psoas');
        # if (np.sum(maskDetect)==0):
        #     st.warning('No organ detected')
        #     maskSegment = None
        #     plotMask = None
        # else:
        plotMask = detectCroppedSeg3DKerasDR_predict_ha.singlePatientSegmentation(params, img, maskDetect0, boxDetect0, kidneyNone0, vol4D0, vol4Dpcs,
                                                zDimOri, 'Psoas', None);

    return plotMask
