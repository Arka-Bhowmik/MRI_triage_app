"""
Created on Sat June 15 2024

@author: Arka Bhowmik, MSKCC
"""
# load in-built packages
import os, sys, base64, logging
logging.disable(logging.WARNING) 
import streamlit            as st
import numpy                as np
import matplotlib.pyplot    as plt
from streamlit_option_menu  import option_menu
from collections            import Counter
from sklearn.metrics        import confusion_matrix, roc_curve
from scipy                  import stats
#
# Automated path selection
@st.cache_data
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
#
# Load the default path (where the present code)
default_path = os.path.dirname(resource_path(__file__))
sys.path.append(os.path.dirname(default_path))
# load custom python file
from support_function import progress_bar, redirect, compute_auc
from testing import inference_triage, inference_segment
#
# Loading the background image as text inside CSS
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()
#
# Load the images used for background
side_background = get_img_as_base64(os.path.join(default_path,'app_file/sidebar.png'))
logo = get_img_as_base64(os.path.join(default_path,'app_file/logo.png'))
triage_img = get_img_as_base64(os.path.join(default_path,'app_file/Triage_pipeline.png'))
compute_step_1 = get_img_as_base64(os.path.join(default_path,'app_file/compute_steps_1.png'))
compute_step_2 = get_img_as_base64(os.path.join(default_path,'app_file/compute_steps_2.png'))
# CSS style for page background color
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-color: #e4e2f5;
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}

[data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image/png;base64,{side_background}");
    background-position: center; 
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

[data-testid="stHeader"] {{
    background: rgba(39, 114, 199, 0.8);
    }}

</style>
"""
# Insert a logo to the page
def insert_logo_page_navigation(file):
    st.sidebar.markdown(f"<img src='data:image/png;base64,{file}' style='max-width: 100%'>"
        , unsafe_allow_html=True)
#
# Insert Navigation Bar
def navigation_bar():
    with st.sidebar.container():
        selected = option_menu(
            menu_title="Navigation Menu",
            options=["Home", "Triage", "Segmentation", 'Contact', 'License'],
            icons=['house', 'alt', 'mask', 'envelope', 'eyeglasses'],
            menu_icon="list",
            orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "#0968c3"},
                "icon": {"color": "orange", "font-size": "25px"}, 
                "nav-link": {"font-size": "25px", "text-align": "left", "margin":"20px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "green"},
            }
            )
        #
    #
    return selected
#
# Insert Navigation Bar
def compute_options():
    with st.container():
        selected = option_menu(
            menu_title=None,
            options=["CPU", "GPU"],
            icons=['cpu', 'gpu-card'],
            menu_icon=None,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#e4e2f5"},
                "icon": {"color": "black", "font-size": "20px"}, 
                "nav-link": {"font-size": "10px", "text-align": "left", "margin":"18px", "--hover-color": "#e4e2f5"},
                "nav-link-selected": {"background-color": "orange"},
            }
            )
        #
    #
    return selected
#
# Function to compute performance
def output_performance(output_dataframe, variable):
    #
    true_labels = output_dataframe.Ground_labels.tolist()
    pred_labels = output_dataframe.Predicted_labels.tolist()
    Probability = output_dataframe.Ensemble_probability.tolist()
    #
    counter_predict  = Counter(pred_labels)
    counter_ground   = Counter(true_labels)
    positive_labels = counter_predict[1]
    negative_labels = counter_predict[0]
    GTP = counter_ground[1]
    GTN = counter_ground[0]
    #
    if (GTP!=0) and (GTN!=0):
        TN, FP, FN, TP = confusion_matrix(true_labels, pred_labels).ravel()    # DETERMINE THE CONFUSION MATRIX
        Sensitivity = TP/(TP+FN),              # RECALL OR SENSITIVITY (POSITIVE CASES CORRECTLY IDENTIFIED AS POSITIVE)
        Specificity = TN/(TN+FP),              # SPECIFICITY (NEGATIVE CASES CORRECTLY IDENTIFIED AS NEGATIVE)
        print("(a) No. of completely normal exams triaged for threshold value of τ = ", variable, " is ", negative_labels, "\n(b) Sensitivity:", (100*Sensitivity),"%" " \n(c) Specificity:", (100*Specificity),"%")
        #
    else:
        print("No. of completely normal exams triaged for threshold value \nof (τ = ", variable, ") is ", negative_labels)
    #
    return GTP, GTN, true_labels, Probability
#
# Function to plot ROC Curve for model threshold varied from 0.00 to 1.00
def plot_roc(GTP, GTN, true_labels, Probability, output_path):
    #
    if (GTP!=0) and (GTN!=0) and (GTP>=15) and (GTN>=15):
        # Predicts the false positive rate and true positive rate
        fpr, tpr, _ = roc_curve(np.array(true_labels), np.array(Probability), pos_label=1)
        # Predicts the auc with 95% CI
        alpha = .95
        auc, auc_cov = compute_auc.delong_roc_variance(np.array(true_labels), np.array(Probability))
        auc_std = np.sqrt(auc_cov)
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
        ci = stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)
        ci[ci > 1] = 1
        # Plot the ROC curve
        plt.figure(figsize=(5, 5))
        plt.plot(fpr,tpr,linestyle='-',color = 'black');
        plt.legend(["{} {:.3f} {} {} {:.4f}, {:.4f} {}".format('AUC:', auc, '\n95% CI:', '[', ci[0], ci[1], ']')], loc='best')
        plt.title('ROC Curve', size = 12);
        plt.xlabel('False Positive Rate', size = 12);
        plt.ylabel('True Positive Rate', size = 12);
        plt.savefig(os.path.join(output_path, 'roc.png'))
    #
#
def run():
    # Home page background setting
    st.markdown(page_bg_img, unsafe_allow_html=True)
    # Insert a logo in sidebar
    insert_logo_page_navigation(logo)
    # Insert blank space
    st.sidebar.text("") # Blank space
    st.sidebar.text("") # Blank space
    st.sidebar.text("") # Blank space
    st.sidebar.text("") # Blank space
    # Insert navigation bar
    selected = navigation_bar()
    # Page 1: Home page
    if selected == "Home":
        #
        st.header('MRI Triage and Image Segmentation App')
        html_text1="""
        __This app perform__ __:orange[triage task]__ and __:green[segmentation task]__\n\n
        ##### :orange[__a) Triage task:__] Completely normal/Low Suspicion breast MRI to an abbreviated list\n
        """
        #
        html_text2="""
        ##### :green[__b) Segmentation task:__] Segment whole breast from thorax\n
        """
        #
        st.markdown(html_text1)
        st.markdown(f"<img src='data:image/png;base64,{triage_img}' style='max-width: 100%'>", unsafe_allow_html=True)
        st.text("") # Blank space
        st.markdown(html_text2)
    # Page 2: DL triage page
    if selected == "Triage":
        # 
        st.header('MRI Triage Task')
        st.markdown(f"<img src='data:image/png;base64,{compute_step_1}' style='max-width: 100%'>", unsafe_allow_html=True)
        st.text("") # Blank space
        st.markdown("""__:red[Note]: This ensemble model predicts output probability of each exam for being suspicious and completely normal. The triage algorithm's objective is to triage abbreviated normal patient without missing CA. AI abbreviated normal patient can be stopped for further scan, but for those with AI being suspicious the scan shall continue (i.e., full protocol) prior to radiologist's reading. The AI directed abbrevaited protocol will reduce the scan time and increase scanning capacity/day.__""")
        html_text1="""
        __Step 1: 🖥️ Select the compute resource__\n
        """
        #
        st.markdown(html_text1)
        compute = compute_options()
        if compute == "CPU":
            file_type = st.selectbox('__Step 2: 🛣️ Options for path__', ('List of path', 'Single Path'))
            if file_type == 'Single Path':
                batch = 1
                uploaded_file = st.file_uploader("__Step 3: 🏞️ Choose an Image (file extension .nii.gz)__", type = ['gz'])
                st.markdown(f"__Step 4: 🗄️ Batch size :blue[(Default)]:__ {batch}")
                threshold = st.text_input("__Step 5:  👩🏻‍💻 User defined prediction threshold (range: 0-1) :blue[(default/best calibration = 0.33)]__", 0.33)
            else:
                uploaded_file = st.file_uploader("__Step 3: 📋 Choose List of Images (file extension .xlsx or .csv with path to '.nii.gz', :blue[_see_] sample csv file in input folder)__", type = ['xlsx', 'csv'])
                col1, col2 = st.columns(2)
                with col1:
                    batch = st.slider("__Step 4(a): 🗄️ Select batch size:__",1, 8)
                #
                with col2:
                    st.markdown(f"__Step 4(b): 🗄️ Selected batch size: {batch}__ ")
                    st.markdown(" _:blue[(batch size > 2-4 needs more CPU cores to load 3D '.nii.gz' files, choose smaller size to avoid error)]_ ")
                #
                threshold = st.text_input("__Step 5: 👩🏻‍💻 User defined prediction threshold (range: 0-1) :blue[(default/best calibration = 0.33)]__", 0.33)
            #
            if uploaded_file is not None:
                #
                if (os.path.splitext(uploaded_file.name)[-1] == '.xlsx'):
                    prediction = st.button("Predict", type='primary')
                elif (os.path.splitext(uploaded_file.name)[-1] == '.csv'):
                    prediction = st.button("Predict", type='primary')
                    
                elif (os.path.splitext(uploaded_file.name)[-1] == '.gz'):
                    prediction = st.button("Predict", type='primary')
                #
                if (prediction != None) and (prediction != False):
                    #
                    if float(threshold) > 1.0:
                        st.write("__⚠️ Fail Notice:__ Algorithm threshold should be in range (0 - 1) :blue[(see instruction in Step 5)]")
                    else:
                        #
                        # Writes the uploaded file to a temp folder
                        file_path = os.path.join(os.path.dirname(default_path), "tempDir")
                        file_name = uploaded_file.name
                        temp_path = os.path.join(file_path, file_name)
                        with open(temp_path,"wb") as f:
                            f.write(uploaded_file.getbuffer())
                        #
                        st.info('Prediction operation of AI is in progress........', icon="ℹ️")
                        progress_text = "⏳ Elapsed total time: "
                        my_bar = progress_bar.Progress(number_of_functions=1)
                        with redirect.stdout(to=st.empty()):
                            output_dataframe = my_bar.go(progress_text, inference_triage.ai_predict_fn, int(batch), float(threshold), os.path.dirname(default_path), 
                                                         file_path, file_name)
                        #
                        st.success('Completed!', icon="✅")
                        st.info('Results 💾: Probability scores and plots are saved in output folder.', icon="ℹ️")
                        with redirect.stdout(to=st.empty()):
                            if ((file_name.endswith('.csv')) or (file_name.endswith('.xlsx'))):
                                GTP, GTN, true_labels, Probability = output_performance(output_dataframe, float(threshold)) # Prints the performance
                                plot_flag = 1
                            else:
                                plot_flag = 0
                                pred_labels = output_dataframe.Predicted_labels.tolist()
                                counter_predict  = Counter(pred_labels)
                                negative_labels = counter_predict[0]
                                print("No. of completely normal exams triaged for threshold value \nof (τ =", float(threshold), ") is ", negative_labels)
                            #
                        #
                        if plot_flag == 1:
                            plot_roc(GTP, GTN, true_labels, Probability, os.path.join(os.path.dirname(default_path), "output"))
                        #
                        os.remove(temp_path)
                    #
                #
            #
        #
        if compute == "GPU":
            file_type = st.selectbox('__Step 2: 🛣️ Options for path__', ('List of path', 'Single Path'))
            if file_type == 'Single Path':
                batch = 1
                uploaded_file = st.file_uploader("__Step 3: 🏞️ Choose an Image (file extension .nii.gz)__", type = ['gz'])
                st.markdown(f"__Step 4: 🗄️ Batch size :blue[(Default)]:__ {batch}")
                threshold = st.text_input("__Step 5:  👩🏻‍💻 User defined prediction threshold (range: 0-1) :blue[(default/best calibration = 0.33)]__", 0.33)
            else:
                uploaded_file = st.file_uploader("__Step 3: 📋 Choose List of Images (file extension .xlsx or .csv with path to '.nii.gz', :blue[_see_] sample csv file in input folder)__", type = ['xlsx', 'csv'])
                col1, col2 = st.columns(2)
                with col1:
                    batch = st.slider("__Step 4(a): 🗄️ Select batch size:__",1, 32)
                #
                with col2:
                    st.markdown(f"__Step 4(b): 🗄️ Selected batch size: {batch}__ ")
                    st.markdown(" _:blue[(batch size > 4 needs high GPU resource to load 3D '.nii.gz' files, choose smaller size to avoid error)]_ ")
                #
                threshold = st.text_input("__Step 5: 👩🏻‍💻 User defined prediction threshold (range: 0-1) :blue[(default/best calibration = 0.33)]__", 0.33)
            #
            if uploaded_file is not None:
                #
                if (os.path.splitext(uploaded_file.name)[-1] == '.xlsx'):
                    prediction = st.button("Predict", type='primary')
                elif (os.path.splitext(uploaded_file.name)[-1] == '.csv'):
                    prediction = st.button("Predict", type='primary')
                    
                elif (os.path.splitext(uploaded_file.name)[-1] == '.gz'):
                    prediction = st.button("Predict", type='primary')
                #
                if (prediction != None) and (prediction != False):
                    #
                    if float(threshold) > 1.0:
                        st.write("__⚠️ Fail Notice:__ Algorithm threshold should be in range (0 - 1) :blue[(see instruction in Step 5)]")
                    else:
                        #
                        # Writes the uploaded file to a temp folder
                        file_path = os.path.join(os.path.dirname(default_path), "tempDir")
                        file_name = uploaded_file.name
                        temp_path = os.path.join(file_path, file_name)
                        with open(temp_path,"wb") as f:
                            f.write(uploaded_file.getbuffer())
                        #
                        st.info('Prediction operation of AI is in progress........', icon="ℹ️")
                        progress_text = "⏳ Elapsed total time: "
                        my_bar = progress_bar.Progress(number_of_functions=1)
                        with redirect.stdout(to=st.empty()):
                            output_dataframe = my_bar.go(progress_text, inference_triage.ai_predict_fn, int(batch), float(threshold), os.path.dirname(default_path), 
                                                         file_path, file_name)
                        #
                        st.success('Completed!', icon="✅")
                        st.info('Results 💾: Probability scores and plots are saved in output folder.', icon="ℹ️")
                        with redirect.stdout(to=st.empty()):
                            if ((file_name.endswith('.csv')) or (file_name.endswith('.xlsx'))):
                                GTP, GTN, true_labels, Probability = output_performance(output_dataframe, float(threshold)) # Prints the performance
                                plot_flag = 1
                            else:
                                plot_flag = 0
                                pred_labels = output_dataframe.Predicted_labels.tolist()
                                counter_predict  = Counter(pred_labels)
                                negative_labels = counter_predict[0]
                                print("No. of completely normal exams triaged for threshold value \nof (τ =", float(threshold), ") is ", negative_labels)
                            #
                        #
                        if plot_flag == 1:
                            plot_roc(GTP, GTN, true_labels, Probability, os.path.join(os.path.dirname(default_path), "output"))
                        #
                        os.remove(temp_path)
                    #
                #
            #
        #
    # Page 3: DL Segmentation page
    if selected == "Segmentation":
        # 
        st.header('MRI Segmentation Task')
        st.markdown(f"<img src='data:image/png;base64,{compute_step_2}' style='max-width: 100%'>", unsafe_allow_html=True)
        st.text("") # Blank space
        st.markdown("""__:red[Note]: The output of each exam are MIP image and its segmented mask, that are saved in same path as source image. This U-net model only performs 2D image segmentation of axial breast mip from thorax.__""")
        html_text1="""
        __Step 1: 🖥️ Select the compute resource__\n
        """
        #
        st.markdown(html_text1)
        compute = compute_options()
        if compute == "CPU":
            #
            uploaded_file = st.file_uploader("__Step 2: 📋 Choose List of Images (file extension .xlsx or .csv with path to '.nii.gz', :blue[_see_] sample csv file in input folder)__", type = ['xlsx', 'csv'])
            col1, col2 = st.columns(2)
            with col1:
                batch = st.slider("__Step 3(a): 🗄️ Select batch size:__",1, 8)
            #
            with col2:
                st.markdown(f"__Step 3(b): 🗄️ Selected batch size: {batch}__ ")
                st.markdown(" _:blue[(batch size > 2-4 needs more CPU cores to load 3D '.nii.gz' files, choose smaller size to avoid error)]_ ")
            #
            img_type = st.selectbox('__Step 4: 👩🏻‍💻 User defined output choice__', ('MIP & mask', 'Sub-MIP1-3 & mask', 'All'))
            #
            if uploaded_file is not None:
                #
                if (os.path.splitext(uploaded_file.name)[-1] == '.xlsx'):
                    prediction = st.button("Predict", type='primary')
                elif (os.path.splitext(uploaded_file.name)[-1] == '.csv'):
                    prediction = st.button("Predict", type='primary')
                #
                if (prediction != None) and (prediction != False):
                    #
                    # Writes the uploaded file to a temp folder
                    file_path = os.path.join(os.path.dirname(default_path), "tempDir")
                    file_name = uploaded_file.name
                    temp_path = os.path.join(file_path, file_name)
                    with open(temp_path,"wb") as f:
                        f.write(uploaded_file.getbuffer())
                    #
                    st.info('Prediction operation of AI is in progress........', icon="ℹ️")
                    progress_text = "⏳ Elapsed total time: "
                    my_bar = progress_bar.Progress(number_of_functions=1)
                    with redirect.stdout(to=st.empty()):
                        output_dataframe = my_bar.go(progress_text, inference_segment.ai_predict_fn, int(batch), os.path.dirname(default_path), 
                                                     file_path, file_name, str(img_type))
                    #
                    st.success('Completed!', icon="✅")
                    st.info('Results 💾: Images and masks are saved into the input image folder.', icon="ℹ️")
                    #
                    os.remove(temp_path)
                    #
                #
            #
        #
        if compute == "GPU":
            #
            uploaded_file = st.file_uploader("__Step 2: 📋 Choose List of Images (file extension .xlsx or .csv with path to '.nii.gz', :blue[_see_] sample csv file in input folder)__", type = ['xlsx', 'csv'])
            col1, col2 = st.columns(2)
            with col1:
                batch = st.slider("__Step 3(a): 🗄️ Select batch size:__",1, 32)
            #
            with col2:
                st.markdown(f"__Step 3(b): 🗄️ Selected batch size: {batch}__ ")
                st.markdown(" _:blue[(batch size > 4 needs high GPU resource to load 3D '.nii.gz' files, choose smaller size to avoid error)]_ ")
            #
            img_type = st.selectbox('__Step 4: 👩🏻‍💻 User defined output choice__', ('MIP & mask', 'Sub-MIP1-3 & mask', 'All'))
            #
            if uploaded_file is not None:
                #
                if (os.path.splitext(uploaded_file.name)[-1] == '.xlsx'):
                    prediction = st.button("Predict", type='primary')
                elif (os.path.splitext(uploaded_file.name)[-1] == '.csv'):
                    prediction = st.button("Predict", type='primary')
                #
                if (prediction != None) and (prediction != False):
                    #
                    # Writes the uploaded file to a temp folder
                    file_path = os.path.join(os.path.dirname(default_path), "tempDir")
                    file_name = uploaded_file.name
                    temp_path = os.path.join(file_path, file_name)
                    with open(temp_path,"wb") as f:
                        f.write(uploaded_file.getbuffer())
                    #
                    st.info('Prediction operation of AI is in progress........', icon="ℹ️")
                    progress_text = "⏳ Elapsed total time: "
                    my_bar = progress_bar.Progress(number_of_functions=1)
                    with redirect.stdout(to=st.empty()):
                        output_dataframe = my_bar.go(progress_text, inference_segment.ai_predict_fn, int(batch), os.path.dirname(default_path), 
                                                     file_path, file_name, str(img_type))
                    #
                    st.success('Completed!', icon="✅")
                    st.info('Results 💾: Images and masks are saved into the input image folder.', icon="ℹ️")
                    #
                    os.remove(temp_path)
                    #
                #
            #
        #
    # Page 4: Contact page
    if selected == "Contact":
        # 
        st.header('📱  Contact Developers')
        html_text1="""
        __User can reach out below contact for help__\n\n
        ###### :orange[__Email__] ✉︎: arkabhowmik@yahoo.co.uk \n
        ###### :orange[__Raw Code__] ⚙️: [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/Arka-Bhowmik/mri_triage_normal.git)
        """
        #
        st.markdown(html_text1)
    # Page 5: License page
    if selected == "License":
        # 
        st.header('𓍝  License Statement')
        html_text1="""
        
        MIT License \n\n

        Copyright ©️ 2023 Arka Bhowmik
        
        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:
        
        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.
        
        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        """
        #
        st.markdown(html_text1)
    #
#
if __name__=='__main__':
    run()