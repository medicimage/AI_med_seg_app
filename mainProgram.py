# This is the entry point for the application

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import nibabel as nib
import modelDeployment
import funcs_ha_use
from nibabel import FileHolder, Nifti1Image, load
from io import BytesIO
from skimage import measure
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import Dicom2Nifti
import trimesh
import base64

import time
timestr = time.strftime("%Y%m%d-%H%M%S")



mask = None
# upload file
@st.cache
def loadData(dataAddress):
    img_vol = funcs_ha_use.readVolume4(dataAddress)
    return img_vol

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.sidebar.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

def dir_selector(folder_path='.'):
    dirnames = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    names = ['-']
    for i in dirnames:
        names.append(i)
    selected_folder = st.sidebar.selectbox('Select a DICOM folder', names)
    if selected_folder is None:
        return None
    return os.path.join(folder_path, selected_folder)

def plotImage(img_vol, slice_i):
    selected_slice = img_vol[:, :, slice_i, 1]

    ax.imshow(selected_slice, 'gray', interpolation='none')
    return fig

def plotImageSag(img_vol, slice_i):
    selected_slice3 = img_vol[:, slice_i, :, 1]

    rotateIm = list(reversed(list(zip(*selected_slice3))))
    ax2.imshow(rotateIm, 'gray', interpolation='none')
    # print(selected_slice2.shape)

    return fig2

def plotImageCor(img_vol, slice_i):
    selected_slice2 = img_vol[slice_i, :, :, 1]
    rotateIm = list(reversed(list(zip(*selected_slice2))))
    ax1.imshow(rotateIm, 'gray', interpolation='none')
    #print(selected_slice2.shape)
    return fig1


class FileDownloader(object):

    def __init__(self, data, filename='morph_file', file_ext='txt'):
        super(FileDownloader, self).__init__()
        self.data = data
        self.filename = filename
        self.file_ext = file_ext

    def download(self):
        b64 = base64.b64encode(self.data.encode()).decode()
        new_filename = "{}_{}_.{}".format(self.filename, timestr, self.file_ext)
        st.markdown("#### Download File ###")
        href = f'<a href="data:file/{self.file_ext};base64,{b64}" download="{new_filename}">Click Here to Download</a>'
        st.markdown(href, unsafe_allow_html=True)

## adding image

image = Image.open('./Images/logo.png')
st.sidebar.image(image)

## main menu

choice = st.sidebar.radio('', ['Project Overview', 'Prototype', 'Publications and Code', 'About'],index=0)
if choice == 'Project Overview':
    st.header('AI-driven Organ Reconstruction and Morphological Features Extraction from Medical Images')

    "The research project aims to develop a framework for the automatic 3D reconstruction of organs and the automatic " \
    "extraction of morphological features from radiological scans. The framework will be driven by an artificial intelligent" \
    " approach and it will have an open-source nature in order to foster clinical studies, collaboration and integrations across" \
    " discipline and modalities."
    _left, mid, _right = st.columns([1, 3, 1])
    with mid:
        image_Overview = Image.open('./Images/Overview.png')
        st.image(image_Overview, width=500, use_column_width='always')

    "The following objectives and goals have been set out :"
    " "
    "- to provide novel algorithms for the automatic segmentation of abdominal organs from MR images. Emerging AI-techniques" \
    " based on Convolutional Neural Networks are developed. The aim is to reach a high level of precision and accuracy" \
    " in order for this to be used in clinical contexts. "
    " "
    "- to visualise the data and provide a 3D-interactive representation of the organs."
    " "
    "- to provide an automatic measurement of the most important biomarkers for the abdominal organs to enable early "\
    " diagnosis or to provide a stratification of subjects according to organ morphology and test the genetic "\
    "and environmental factors that may contribute to organ function."
    " "
    "- to provide a novel open-source imaging framework for the automatic feature extraction and segmentation of the"\
    "organs in the abdomen from medical images, using the newly developed algorithms. "
    " "
    "- to elaborate the concept of publicly available cloud-based image-processing services as one-stop-shop services"\
    " that clinical experts, GPs, Life Sciences and well-being researchers can seamlessly access through webpages, "\
    "without needing knowledge of the underpinning technology."
    " "
    "the aim is to provide the scientific community with a framework that will be easy to use, and it will not "\
    "require the interaction of an expert operator. The framework could be used by the wider community and the "\
    "developed algorithms will be open-source and easily accessible, showing a high accuracy in the reconstruction "\
    "of patient-specific anatomical structures in order to improve the analysis and detection of diseases and treatment "\
    "planning performed by radiologist, clinicians and medical health services. The proposed framework will also "\
    "support the stratification of subjects according to organ morphology."

if choice == 'Publications and Code':
    st.header('Publications and Code')
    "The algorithms implemented are based on the following research articles:"
    " "
    "- B. Villarini, H. Asaturyan, S. Kurugol, O. Afacan, J. D. Bell and E. L. Thomas, "\
    "3D Deep Learning for Anatomical Structure Segmentation in Multiple Imaging Modalities,"\
    "2021 IEEE 34th International Symposium on Computer-Based Medical Systems (CBMS),"\
    "2021, pp. 166-171, doi: 10.1109/CBMS52027.2021.00066"
    " "
    "- H. Asaturyan, B. Villarini, K. Sarao, S.K. Chow, O. Afacan, and S. Kurugol,"\
    " Improving Automatic Renal Segmentation in Clinically Normal and Abnormal Paediatric DCE-MRI via "\
    "Contrast Maximisation and Convolutional Networks for Computing Markers of Kidney Function, "\
    "Sensors Journal – MDPI – Special Issue in Advances in Image Segmentation: Theory and Applications (In press)"
    " "
    "The source code for training and testing the proposed deep learning models is available in GitHub at the following link:"
    "https://github.com/med-seg/kidney-mc"
    " "
    "The code for the web-application is available in GitHub at the following link:"
    "Add link...."
if choice == 'About':
    st.header("About the project")
    "The project was part of a project funded by the Royal Academy of Engineering - Leverhume Trust Research Fellowship "\
    "awarded to Dr. Barbara Villarini (LTRF1920\\16\\26)."
    "The project started on 1st September 2020 - 15 months duration"
    "https://www.raeng.org.uk/news/news-releases/2020/august/academy-awards-seven-new-leverhulme-trust-research"
    _left, mid, _right = st.columns([1, 3, 1])
    with mid:
        image_Overview = Image.open('./Images/About.png')
        st.image(image_Overview)
    "Information regarding the funding scheme:"
    "https://www.raeng.org.uk/grants-prizes/grants/support-for-research/leverhulme-research-fellowship"

if choice == "Prototype":
    #sample
    sample = st.sidebar.checkbox('Load Sample - MRI Liver')
    ## adding dir selector for dicom folder
    dirname = dir_selector()

    ## adding uploader bar
    uploaded_nii_file = st.sidebar.file_uploader(" OR Select a NIfTI file:", type=['nii', 'img'], accept_multiple_files=False)

    if sample or uploaded_nii_file or os.path.isdir(dirname):
        if sample:
            uploaded_nii_file = nib.load('./Data/Liver_extract.nii')
            # voxel size
            sx, sy, sz = uploaded_nii_file.header.get_zooms()
            nii_data = uploaded_nii_file.get_data()
            img_1 = np.asarray(nii_data)
            img = Nifti1Image(img_1, affine=uploaded_nii_file.affine)
        elif os.path.isdir(dirname):
            with st.spinner('Wait for it. Loading DICOM files...'):
                try:
                    img = Dicom2Nifti.dicom2nii(dirname)
                    sx, sy, sz = img.header.get_zooms()
                except RuntimeError:
                    st.text('This does not look like a DICOM folder!')
        else:
            rr = uploaded_nii_file.read()
            bb = BytesIO(rr)
            fh = FileHolder(fileobj=bb)
            img = Nifti1Image.from_file_map({'header': fh, 'image': fh})
            sx, sy, sz = img.header.get_zooms()

        img_vol = loadData(img)
        ## set title main window
        st.title("Medical Images Visualisation")

        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        # plot the slider
        n_slices1 = img_vol.shape[2]
        slice_i1 = col1.slider('Slice - Axial', 0, n_slices1, int(n_slices1 / 2))

        n_slices2 = img_vol.shape[0]
        slice_i2 = col2.slider('Slice - Coronal', 0, n_slices2, int(n_slices2 / 2))

        n_slices3 = img_vol.shape[1]
        slice_i3 = col3.slider('Slice - Sagittal', 0, n_slices3, int(n_slices3 / 2))

    # plot volume
        fig, ax = plt.subplots()
        plt.axis('off')

        fig = plotImage(img_vol, slice_i1)

        #plot coronal view
        fig1, ax1 = plt.subplots()
        plt.axis('off')


        fig1 = plotImageCor(img_vol, slice_i2)

        # plot sagittal view
        fig2, ax2 = plt.subplots()
        plt.axis('off')

        fig2 = plotImageSag(img_vol, slice_i3)

        # write meta information from file
        if st.sidebar.checkbox('Show Metadata'):
            # header = uploaded_nii_file.header
            st.subheader('Metadata')
            hdr = img.header

            header = [(_key) for _key in hdr]
            headers_data = []
            headers_data.append([hdr[_key] for _key in hdr])
            headers_dataT = list(zip(*headers_data))
            headers_dataTcc=[]
            for cc in headers_dataT:
                headers_dataTcc.append(str(cc[0]))

            data_all = {'key': header, 'value': headers_dataTcc}
            df = pd.DataFrame(data_all)
            st.dataframe(df.astype(str))

        if sample:
            optionS = st.sidebar.radio('Select Organ to segment', ['None', 'Liver'],
                                      index=0)
            if optionS == 'Liver':
                with st.spinner('Wait for it...'):
                    # load segmentation model
                    # perform segmentation
                    mask = modelDeployment.runDeepSegmentationModel('Liver', img)
                    # plot segmentation mask

                    fig, ax = funcs_ha_use.plotMask(fig, ax, img, mask, slice_i1, 'AX', 'Liver')
                    fig1, ax1 = funcs_ha_use.plotMask(fig1, ax1, img, mask, slice_i2, 'CR', 'Liver')
                    fig2, ax2 = funcs_ha_use.plotMask(fig2, ax2, img, mask, slice_i3, 'SG', 'Liver')

        else:
            option = st.sidebar.radio('Select Organ to segment', ['None', 'Liver', 'Pancreas', 'Psoas', 'Kidneys'], index=0)
            if option == 'Liver':
                ## start spinner
                with st.spinner('Wait for it...'):
                    # load segmentation model
                    # perform segmentation
                    mask = modelDeployment.runDeepSegmentationModel('Liver', img)
                    # plot segmentation mask

                    fig, ax = funcs_ha_use.plotMask(fig, ax, img, mask, slice_i1, 'AX', 'Liver')
                    fig1, ax1 = funcs_ha_use.plotMask(fig1, ax1, img, mask, slice_i2, 'CR', 'Liver')
                    fig2, ax2 = funcs_ha_use.plotMask(fig2, ax2, img, mask, slice_i3, 'SG', 'Liver')


            if option == 'Pancreas':
                ## start spinner
                with st.spinner('Wait for it...'):
                    # load segmentation model
                    # perform segmentation
                    mask = modelDeployment.runDeepSegmentationModel('Pancreas', img)
                    # plot segmentation mask
                    fig, ax = funcs_ha_use.plotMask(fig, ax, img, mask, slice_i1, 'AX', 'Pancreas')
                    fig1, ax1 = funcs_ha_use.plotMask(fig1, ax1, img, mask, slice_i2, 'CR', 'Pancreas')
                    fig2, ax2 = funcs_ha_use.plotMask(fig2, ax2, img, mask, slice_i3, 'SG', 'Pancreas')

            if option == 'Psoas':
                ## start spinner
                with st.spinner('Wait for it...'):
                    # load segmentation model
                    # perform segmentation
                    mask = modelDeployment.runDeepSegmentationModel('Psoas', img)

                    # plot segmentation mask
                    fig, ax = funcs_ha_use.plotMask(fig, ax, img, mask, slice_i1, 'AX', 'Psoas')
                    fig1, ax1 = funcs_ha_use.plotMask(fig1, ax1, img, mask, slice_i2, 'CR', 'Psoas')
                    fig2, ax2 = funcs_ha_use.plotMask(fig2, ax2, img, mask, slice_i3, 'SG', 'Psoas')


            if option == 'Kidneys':
                ## start spinner
                with st.spinner('Wait for it...'):
                    # load segmentation model
                    # perform segmentation
                    mask = modelDeployment.runDeepSegmentationModel('Kidneys', img)
                    if (maskSegment != None):
                        # plot segmentation mask
                        fig, ax = funcs_ha_use.plotMask(fig, ax, img, mask, slice_i1, 'AX', 'Kidneys')
                        fig1, ax1 = funcs_ha_use.plotMask(fig1, ax1, img, mask, slice_i2, 'CR', 'Kidneys')
                        fig2, ax2 = funcs_ha_use.plotMask(fig2, ax2, img, mask, slice_i3, 'SG', 'Kidneys')


    # plot volume
        plot = col1.pyplot(fig)
        plot = col2.pyplot(fig1)
        plot = col3.pyplot(fig2)

        visual3D = st.sidebar.checkbox('3D Visualisation')

        with st.spinner('Wait for 3D visualisation...'):

            if visual3D:
                if mask is not None:
                    verts, faces, normals, values = measure.marching_cubes_lewiner(mask, 0.0, allow_degenerate=False)

                    fig4 = go.Figure(data=[go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2], i=faces[:,0], j=faces[:,1], k=faces[:,2],
                                                opacity=0.6,
                                                autocolorscale=True)])
                    col4.plotly_chart(fig4)
                else:
                    st.warning ('No segmentation - Select Organ to segment to see the 3D visualisation')

            morphF = st.sidebar.checkbox('Morphological features')
            if morphF:
                if mask is not None and visual3D:
                    ## compute volume
                    # voxel size
                    vox_volume = sx * sy * sz
                    volume = np.count_nonzero(mask) * vox_volume * pow(10, -3)

                    with st.spinner('Wait for it...'):
                        ## compute curvature
                        mesh = trimesh.Trimesh(verts, faces)
                        curvatures = trimesh.curvature.discrete_mean_curvature_measure(mesh, verts, 10.0 / (4 * np.pi))
                        meanCurv = abs(np.mean(curvatures))
                    # show in a table
                        d = {'Volume': [volume], 'Curvature': [meanCurv]}
                        #print (VC)
                        df_VC = pd.DataFrame(d)
                        st.subheader('Morphological measures')
                        st.table(df_VC)
                        download = FileDownloader(df_VC.to_csv(), file_ext='csv').download()


                else:
                    st.warning('No segmentation - Select Organ to segment and perform 3D visualisation to get morphological measures')




