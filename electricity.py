import pandas as pd
import streamlit as st 
import pickle 
import plotly.express as px
import numpy as np

tab_1,tab_2 = st.tabs(['VIEW PREDICTION','DATAFRAME AND DOWNLOAD'])

option = st.sidebar.selectbox("Choose the type of prediction to perform",["Single","Multiple"])

model_lr = pickle.load(open("model_lr.sav", 'rb'))
model_gb = pickle.load(open("model_gb.sav", 'rb'))

if option.lower() == "single" :
    st.sidebar.title("Data Input")
    gap = st.sidebar.number_input("Input the Global Active Power",0.0000,12.0000,step=1e-6,format="%.5f")
    grp = st.sidebar.number_input("Input the Global Reactive Power",0.0000,2.0000,step=1e-6,format="%.5f")
    volt = st.sidebar.number_input("Input the Voltage",0.0000,280.0000,step=1e-6,format="%.5f")
    gi = st.sidebar.number_input("Input the Global Intensity",0.0000,50.0000,step=1e-6,format="%.5f")
    dy = st.sidebar.selectbox("Select the Day",["Sunday","Monday","Tuesday","Wednesday","Thursday",
                                                   "Friday","Saturday"])
    if dy == "Sunday" :
        day = '0'
    elif dy == "Monday" :
        day = '1'
    elif dy == "Tuesday" :
        day = '2'
    elif dy == "Wednesday" :
        day = '3'
    elif dy == "Thursday" :
        day = '4'
    elif dy == "Friday" :
        day = '5'
    elif dy == "Saturday" :
        day = '6'
    hour = st.sidebar.selectbox("Select the hour",['0','1','2','3','4','5','6','7','8','9','10','11',
                                                   '12','13','14','15','16','17','18','19','20','21','22','23'])

    df = pd.DataFrame()

    df['Global_active_power'] = [gap]
    df['Global_reactive_power'] = [grp]
    df['Voltage'] = [volt]
    df['Global_intensity'] = [gi]
    df["Day"] = [day]
    df['Hour'] = [hour]
    
    data = df.copy()
    cols_to_cat = ['Hour','Day']
    new_df = data.drop(cols_to_cat,axis=1)
    cat_df = pd.get_dummies(data[cols_to_cat])
    for i in cat_df.columns :
        cat_df[i] = cat_df[i].astype('int8')
    encoded_cols = ['Hour_0', 'Hour_1', 'Hour_2',
       'Hour_3', 'Hour_4', 'Hour_5', 'Hour_6', 'Hour_7', 'Hour_8', 'Hour_9',
       'Hour_10', 'Hour_11', 'Hour_12', 'Hour_13', 'Hour_14', 'Hour_15',
       'Hour_16', 'Hour_17', 'Hour_18', 'Hour_19', 'Hour_20', 'Hour_21',
       'Hour_22', 'Hour_23', 'Day_0', 'Day_1', 'Day_2', 'Day_3', 'Day_4',
       'Day_5', 'Day_6']
    df_encode = pd.DataFrame()
    for i in (encoded_cols) : 
        if i in (cat_df.columns) :
            df_encode[i] = [1]
        else :
            df_encode[i] = [0]
    data = new_df.join(df_encode)
    pred_lr = model_lr.predict(data)
    pred_gb = model_gb.predict(data)
    
    tab_1.success("Linear Regression Prediction")
    df['Predict_lr'] = [pred_lr]
    tab_1.write(f"""When the Day is {dy}, the hour is {hour}, the Global Active Power is {gap},
                the Global Reactive power is {grp}, the Global Intensity is {gi}, and the voltage is {volt}.
                 The Linear regression algorithm predicts electricity consumption as {pred_lr}""")
    
    tab_1.success("Gradient Boosting Prediction")
    df['Predict_gb'] = [pred_gb]
    tab_1.write(f"""When the Day is {dy}, the hour is {hour}, the Global Active Power is {gap},
                the Global Reactive power is {grp}, the Global Intensity is {gi}, and the voltage is {volt}.
                 The Gradient boosting regressor algorithm predicts electricity consumption as {pred_gb}""")
    
    tab_2.write(df)
    @st.cache_data 

    def convert_df(df): 
        return df.to_csv().encode('utf-8')
    csv = convert_df(df)
    tab_2.success("Print Result as CSV file")
    tab_2.download_button("Download",csv,"Prediction.csv",'text/csv')

else :
    file = st.sidebar.file_uploader("Input File")
    if file == None :
        st.write("A file should be uploaded")
    else :
        @st.cache_data
        def load(data) :
            df = pd.read_csv(data)
            return df
        
        df = load(file)

        convert = ['Hour','Day']
        for i in convert :
            df[i] = df[i].astype('category')

        data = df.copy()

        cols = ['Hour',"Day"]
        new_df = pd.get_dummies(data[cols])
        data = data.drop(cols,axis=1)
        data = data.join(new_df)

        cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
       'Global_intensity', 'Hour_0', 'Hour_1', 'Hour_2',
       'Hour_3', 'Hour_4', 'Hour_5', 'Hour_6', 'Hour_7', 'Hour_8', 'Hour_9',
       'Hour_10', 'Hour_11', 'Hour_12', 'Hour_13', 'Hour_14', 'Hour_15',
       'Hour_16', 'Hour_17', 'Hour_18', 'Hour_19', 'Hour_20', 'Hour_21',
       'Hour_22', 'Hour_23', 'Day_0', 'Day_1', 'Day_2', 'Day_3', 'Day_4',
       'Day_5', 'Day_6']
        
        for i in cols :
            if i not in data.columns :
                data[i] = np.zeros(len(df))
        #st.write(data.columns)

        pred_lr = model_lr.predict(data)
        pred_gb = model_gb.predict(data)

        lists = []
        for i in pred_lr :

            if i < 0 :
                lists.append(0)
            else :
                lists.append(i)
        
        df['prediction Lr'] = lists
        df['prediction Gb'] = pred_gb

        with tab_1 :
            st.write("The bar plot for daily electricity")
            fig = px.bar(df, x= 'Day', y= ['prediction Lr', 'prediction Gb'],barmode= 'group')
            st.write(fig)

            st.write("The bar plot for electricity per hour")
            fig_1 = px.bar(df, x= 'Hour', y= ['prediction Lr', 'prediction Gb'],barmode= 'group')
            st.write(fig_1)

        tab_2.dataframe(df)
        @st.cache_data 

        def convert_df(df): 
            
            return df.to_csv().encode('utf-8')
        csv = convert_df(df)
        tab_2.success("Print Result as CSV file")
        tab_2.download_button("Download",csv,"Prediction.csv",'text/csv')
        
