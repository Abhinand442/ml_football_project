import numpy as np
import pandas as pd
import streamlit as slt
import joblib
model_regress = joblib.load("regression_model.joblib")
model_class = joblib.load("classification_model.joblib")
model_columns_clas = joblib.load("classification_columns.joblib")
model_columns_reg  = joblib.load("regression_columns.joblib")
slt.set_page_config(page_title="MatchVision ⚽",page_icon="⚽")
slt.title('Football Player Performance Predictor')
slt.write('MatchVision: Football Player Performance Prediction')
player_name=slt.text_input('Player Name')
goals_last_5=slt.number_input('Goals scored in last 5 matches')
shots_on_target=slt.number_input('Shot on Target')
assists_last_5=slt.slider('Assists in last 5 matches',0,4,1)
pass_accuracy=slt.slider('Passing Accuracy',0.0,100.0,50.0)
fitness_score=slt.slider('Fitness Accuracy',0.0,100.0,30.0)
opponent=slt.number_input('Opponent Strength')
minutes_last_5=slt.slider('Minutes played in last 5 matches',10.0,500.0,10.0)
player_position=slt.selectbox('Player Position',['Forward','Midfielder','Defender','Goalkeeper'])
home_away=slt.selectbox('Stadium',['Home','Away'])
weather=slt.selectbox('Weather',['Rainy','Snowy','Sunny','Cloudy'])

# goals_per_90=goals_last_5/((minutes_last_5)/90)

input_df=pd.DataFrame({
    'goals_last_5':[goals_last_5],
    'assists_last_5':[assists_last_5],
    'shots_on_target':[shots_on_target],
    'pass_accuracy':[pass_accuracy],
    'fitness_score':[fitness_score],
    'opponent_strength':[opponent],
    'minutes_last_5':[minutes_last_5],
    'player_position':[player_position],
    'home_away':[home_away],
    'weather':[weather]

})
input_df=pd.get_dummies(input_df,drop_first=False)
input_df1=input_df.reindex(columns=model_columns_clas,fill_value=0)
input_df2=input_df.reindex(columns=model_columns_reg,fill_value=0)
if slt.button('Predict Performance'):
    prediction1=model_class.predict(input_df1)
    slt.write(prediction1)
    prediction2=model_regress.predict(input_df2)
    slt.write(float(prediction2))