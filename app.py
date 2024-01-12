import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns 
import pickle 

#import model 
xgboost = pickle.load(open('xgb.pkl','rb'))

#load dataset
data = pd.read_csv('Stroke_Dataset_New.csv')
#data = data.drop(data.columns[0],axis=1)

st.title('Aplikasi Stroke')

html_layout1 = """
<br>
<div style="background-color:red ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Stroke Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['Xgboost','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('Data Pasien')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset Stroke</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDA'):
    pr =ProfileReport(data,explorative=True)
    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)

#train test split
X = data.drop('stroke',axis=1)
y = data['stroke']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    gender = st.sidebar.selectbox('Jenis Kelamin',['Laki-laki','Perempuan'])
    if gender == 'Laki-laki':
        gender = 1
    else:
        gender = 0

    ever_married = st.sidebar.selectbox('Sudah Menikah',['Tidak','Ya'])
    if ever_married == 'Tidak':
        ever_married = 0
    else:
        ever_married = 1

    work_type = st.sidebar.selectbox('Pekerjaan',['Private','Self-employed','children','never_worked'])
    if work_type == 'Private':
        work_type = 0
    elif work_type == 'Self-employed':
        work_type = 1
    elif work_type == 'children':
        work_type = 2
    else:
        work_type = 3


    Residence_type = st.sidebar.selectbox('Tipe Domisili',['Urban','Rural'])
    if Residence_type == 'Urban':
        Residence_type = 0
    else:
        Residence_type = 1

    smoking_status = st.sidebar.selectbox('Apakah anda merokok',['Tidak','Ya'])
    if smoking_status == 'Tidak':
        smoking_status = 0
    else:
        smoking_status = 1

    age = st.sidebar.slider('Umur',20,100,10)

    hypertension = st.sidebar.selectbox('Apakah anda Hipertensi',['Tidak','Ya'])
    if hypertension == 'Tidak':
        hypertension = 0
    else:
        hypertension = 1

    heart_disease = st.sidebar.selectbox('Apakah anda Penyakit Jantung',['Tidak','Ya'])
    if heart_disease == 'Tidak':
        heart_disease = 0
    else:
        heart_disease = 1

    avg_glucose_level = st.sidebar.slider('Level Glukosa',0,200,10)

    bmi = st.sidebar.slider('BMI',0,80,10)


    
    user_report_data = {
        'gender':gender,
        'ever_married':ever_married,
        'work_type':work_type,
        'Residence_type': Residence_type,
        'smoking_status':smoking_status,
        'age':age,
        'hypertension':hypertension,
        'heart_disease':heart_disease,
        'avg_glucose_level':avg_glucose_level,
        'bmi':bmi,
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasien
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)
user_result = xgboost.predict(user_data)
xgboost_score = accuracy_score(y_test,xgboost.predict(X_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Kamu Aman'
else:
    output ='Kamu terkena stroke'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(xgboost_score*100)+'%')


