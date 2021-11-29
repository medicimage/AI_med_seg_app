Welcome to the AI Organ Segmentation and Visualisation Streamilt app. 
The app uses some deep learning models to perform automatic segmentation of the organ of interest from MRI images and CT scans. 
Published manuscripts:
-	B. Villarini, H. Asaturyan, S. Kurugol, O. Afacan, J. D. Bell and E. L. Thomas, "3D Deep Learning for Anatomical Structure Segmentation in Multiple Imaging Modalities," 2021 IEEE 34th International Symposium on Computer-Based Medical Systems (CBMS), 2021, pp. 166-171, doi: 10.1109/CBMS52027.2021.00066
-	Asaturyan, H., Villarini, B., Sarao, K., Chow S. K., Afacan, O. and Kurugol, S. Improving Automatic Renal Segmentation in Clinically Normal and Abnormal Paediatric DCE-MRI via Contrast Maximisation and Convolutional Networks for Computing Markers of Kidney Function, Sensor – Special Issue on in Advances in Image Segmentation: Theory and Applications (In Press)

Video demonstration at https://www.youtube.com/watch?v=dlgi_ZboygE

The cloud-hosted website (no code, downloading, or installation needed) is available at https://share.streamlit.io/medicimage/ai_med_seg_app/main/mainProgram.py

## INSTALLATION - For running locally:
1.	Install Anaconda from instructions [here](https://docs.anaconda.com/anaconda/install/). 
- If requested, add Anaconda to PATH variable
- To reduce download and install size, you could install Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html)

2.	Download github repository locally
- Manual download: go to the GitHub page https://github.com/medicimage/AI_med_seg_app --> select the green 'Code' button dropdown --> Download Zip
- Or clone the repo locally using git or other SVN manager.

3.	Open a terminal in the master folder

4.	Create the a conda environment by running the command `conda env create -f environment.yml`

5.	Install Streamlit following the instruction at this link: [here](https://docs.streamlit.io/library/get-started/installation) 

6.	Make sure you have installed all the libraries that you can find listed in the [requirements.txt](https://github.com/medicimage/AI_med_seg_app/blob/main/requirements.txt) file.
- If you want to install a single library you can do it via `pip install <name of the library>`
- Otherwise using the command `pip install -r requirements.txt`

7.	Start streamlit app in same terminal window (do this every time to start the app)
- Run the command `streamlit run mainProgram.py`
- Streamlit will start from this command terminal and open up automatically. 


