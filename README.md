# prediction_app_pycharm
Prediction application using streamlit in pycharm
==================================================
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
from sklearn.preprocessing import LabelEncoder

# Title of the app
st.sidebar.header('**Crime Prediction Application**')

def user_caract_entree():
    # Input fields for age group
    st.sidebar.header('Age Group')
    age_group = st.sidebar.selectbox('Age Group', ['18-24', '25-44', '45-64', '65+', '<18'])

    # Input fields for perpetrator sex
    st.sidebar.header('Sex')
    perp_sex_f = st.sidebar.checkbox('Female')
    perp_sex_m = st.sidebar.checkbox('Male')

    # Input fields for perpetrator race
    st.sidebar.header('Race')
    perp_race = st.sidebar.selectbox('Perpetrator Race', ['Black', 'White', 'White Hispanic',
                                                          'Black Hispanic', 'Unknown',
                                                          'Asian / Pacific Islander',
                                                          'American Indian/Alaskan Native',
                                                          'Others'])

    # Input fields for latitude and longitude
    st.sidebar.header('Location')
    latitude = st.sidebar.slider('Latitude', 40.50110098000005, 40.91027045300007, 40.50110098000005)
    longitude = st.sidebar.slider('Longitude', -74.249754952, -73.70659676099996, -74.249754952)

    # Collect the user inputs
    user_inputs = {
        'AGE_GROUP_18-24': [1 if age_group == '18-24' else 0],
        'AGE_GROUP_25-44': [1 if age_group == '25-44' else 0],
        'AGE_GROUP_45-64': [1 if age_group == '45-64' else 0],
        'AGE_GROUP_65+': [1 if age_group == '65+' else 0],
        'AGE_GROUP_<18': [1 if age_group == '<18' else 0],
        'PERP_SEX_F': [1 if perp_sex_f else 0],
        'PERP_SEX_M': [1 if perp_sex_m else 0],
        'PERP_RACE_BLACK': [1 if perp_race == 'Black' else 0],
        'PERP_RACE_WHITE': [1 if perp_race == 'White' else 0],
        'PERP_RACE_WHITE HISPANIC': [1 if perp_race == 'White Hispanic' else 0],
        'PERP_RACE_BLACK HISPANIC': [1 if perp_race == 'Black Hispanic' else 0],
        'PERP_RACE_UNKNOWN': [1 if perp_race == 'Unknown' else 0],
        'PERP_RACE_ASIAN / PACIFIC ISLANDER': [1 if perp_race == 'Asian / Pacific Islander' else 0],
        'PERP_RACE_AMERICAN INDIAN/ALASKAN NATIVE': [1 if perp_race == 'American Indian/Alaskan Native' else 0],
        'PERP_RACE_OTHERS': [1 if perp_race == 'Others' else 0],
        'Latitude': [latitude],
        'Longitude': [longitude]
    }

    user_data = pd.DataFrame(user_inputs, index=[0])
    return user_data

data_input = user_caract_entree()

# Import the dataset
df = pd.read_csv(r'C:\Users\User\PycharmProjects\PFE_Criminologie_APP\dt_streamlit.csv')
dt = df.drop(columns=['LAW_CAT_CD'])
# Concatenate data horizontally
donnee_entree = pd.concat([data_input, dt], axis=1)

# Remove unnecessary columns from donnee_entree
donnee_entree = donnee_entree.loc[:, ~donnee_entree.columns.duplicated()]

# Select only the first row
donnee_entree = donnee_entree.iloc[:1, :17]  # Keep only the first 17 columns

# Display the transformed data
st.subheader('Les caractéristiques transformées')
st.write(donnee_entree)

# Import the model
load_model = pickle.load(open(r'C:\Users\User\PycharmProjects\PFE_Criminologie_APP\model.NNclassifier.pkl', 'rb'))

# Apply the model on the input profile
prevision = load_model.predict(donnee_entree)


# Get the predicted class labels
predicted_labels = np.argmax(prevision, axis=1)


# Instanciation du LabelEncoder
label_encoder = LabelEncoder()
# Fit the label encoder on the target labels
label_encoder.fit(df['LAW_CAT_CD'])

# Decode the predicted labels using the label encoder
predicted_labels_decoded = label_encoder.inverse_transform(predicted_labels)

# Print the predicted labels
print(predicted_labels_decoded)
st.subheader('Résultat de la prévision')
st.write(predicted_labels_decoded)
