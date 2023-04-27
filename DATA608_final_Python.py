#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed Mar 22 22:07:54 2023

@author: dsimbandumwe

"""



from dash import Dash, dcc, Input, Output, html, ctx
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc

import pandas as pd
import plotly.express as px

import pmdarima as pm
from statsmodels.tsa.api import ExponentialSmoothing
from neuralprophet import NeuralProphet




############################################################################
##
## Define Application
##
############################################################################



app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server





############################################################################
##
## Functions
##
############################################################################




##
##  Display Table
##
def create_table(df):
    columns, values = df.columns, df.values
    header = [html.Tr([html.Th(col) for col in columns])]
    rows = [html.Tr([html.Td(cell) for cell in row]) for row in values]
    table = [html.Thead(header), html.Tbody(rows)]
    return table




##
##  Process Bike Table
##
def process_bike_table(df):
    
    c_df = df.copy()   
    c_df['ride_date'] = c_df['ride_date'].dt.floor('D')   
    c_df = c_df.groupby(['station_id','station_name','ride_date'])[['bikes_out','bikes_in','bikes_chng','bikes_chng_pred']].sum().reset_index()  
    c_df = c_df[['station_id','ride_date','station_name','bikes_out','bikes_in','bikes_chng','bikes_chng_pred']] 
    c_df = c_df.dropna()
        
    return c_df




##
##  Arima Model
##
def run_arima_model(df,train_per,pred_per):
     
    # Split Data
   splt_index = round(df.shape[0] * 0.8)
    
    
    # Train data prep
   train_df = df[:splt_index]
   t_idx = round(train_df.shape[0] * (1-train_per/100))
   train_df = train_df[t_idx:].copy()
   train = train_df[['bikes_chng','ride_date']]
   train.set_index('ride_date', inplace=True)
   train = train.resample('1H').sum()

   
   
   # Test data prop
   test_df = df[splt_index:]
   t_idx = round(test_df.shape[0] * pred_per/100)
   test_df = test_df[:t_idx].copy()
   test = test_df[['bikes_chng','ride_date']]
   test.set_index('ride_date', inplace=True)
   test = test.resample('1H').sum()
    
    
    # Initiate and Fit Model
   arima_model = pm.auto_arima(train, test='adf', 
                              seasonal=True, 
                              error_action='ignore',  
                              suppress_warnings=True, 
                              stepwise=False)
    
   forecast = arima_model.predict(len(test))
   
   
   ar_df = test.copy()
   ar_df['bikes_chng_pred'] = forecast
   

   ar_df.reset_index(inplace=True)
   ar_df = ar_df[['ride_date','bikes_chng_pred']]

    
   df = pd.concat([train_df,test_df])
   df = pd.merge(df, ar_df, how="outer", on = ["ride_date"])
   df.sort_values('ride_date', inplace=True)
        
   return df



##
##  ETS Model
##
def run_ets_model(df, train_per, pred_per):
   
   splt_index = round(df.shape[0] * 0.8)
   
   
   # Train data prep
   train_df = df[:splt_index]
   t_idx = round(train_df.shape[0] * (1-train_per/100))
   train_df = train_df[t_idx:].copy()
   train = train_df[['bikes_chng','ride_date']]
   train.set_index('ride_date', inplace=True)
   train = train.resample('1H').sum()
   
   
   
   # Test data prop
   test_df = df[splt_index:]
   t_idx = round(test_df.shape[0] * pred_per/100)
   test_df = test_df[:t_idx].copy()
   test = test_df[['bikes_chng','ride_date']]
   test.set_index('ride_date', inplace=True)
   test = test.resample('1H').sum()

   
   # Initialize and Fit Model
   hwm = ExponentialSmoothing(train, seasonal_periods=24*7, 
                              trend='add', 
                              seasonal='add', 
                              use_boxcox=False, 
                              initialization_method='estimated').fit()
   
   hwm_pred = hwm.forecast(len(test))

   # Process Return Dataframe
   hwm_df = pd.DataFrame(hwm_pred)
   hwm_df.columns = ['bikes_chng_pred']
   hwm_df.reset_index(inplace=True)
   hwm_df.rename({'index': 'ride_date'}, axis=1, inplace=True)

   df = pd.concat([train_df,test_df])
   df = pd.merge(df, hwm_df, how="outer", on = ["ride_date"])
   df.sort_values('ride_date', inplace=True)
   
   return df
    


##
##  Neural Prophet Model
##
def run_prophet_model(df, train_per, pred_per):
    
   
   splt_index = round(df.shape[0] * 0.8)
   
  
   train_df = df[:splt_index]
   t_idx = round(train_df.shape[0] * (1-train_per/100))
   train_df = train_df[t_idx:]
   train = train_df[['bikes_chng','ride_date']].copy()
   train.rename({'ride_date' : 'ds' , 'bikes_chng' :'y'
       }, axis=1, inplace=True)
   train.set_index('ds', inplace=True)
   train = train.resample('1H').sum()   
   train.reset_index(inplace=True)

   test_df = df[splt_index:]
   t_idx = round(test_df.shape[0] * pred_per/100)
   test_df = test_df[:t_idx]
   test = test_df[['bikes_chng','ride_date']].copy()
   test.rename({'ride_date' : 'ds' , 'bikes_chng' :'y'
       }, axis=1, inplace=True)
   test.set_index('ds', inplace=True)
   test = test.resample('1H').sum()
   test.reset_index(inplace=True)


   
   # Prophet Forecast
   m = NeuralProphet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=True
    )
   
   m.fit(train, freq='H', validation_df=test, progress="plot")
   forecast = m.predict(test)
   
  
   proph_pred = forecast[['ds','yhat1']].copy()
   proph_pred.rename({'ds': 'ride_date',
                      'yhat1' : 'bikes_chng_pred'
                      }, axis=1, inplace=True)
   
   df = pd.concat([train_df,test_df])
   df = pd.merge(df, proph_pred, how="outer", on = ["ride_date"])
   df.sort_values('ride_date', inplace=True)
   
   return df





############################################################################
##
## COnfig and Data
##
############################################################################


# Config 
station_lst = ['7617.07']
bike_file = 'https://raw.githubusercontent.com/dsimband/DATA608_FINAL/main/data/bike.csv'




# Data
bike_df = pd.read_csv(bike_file, parse_dates=['ride_date'],dtype = {'station_id': str})
bike_df = bike_df.sort_values('ride_date', ascending=True)



model_df = pd.DataFrame({'label':['1','2','3'],
                         'value':['Auto SARIMA', 'NeuralProphet' , 'Exponential Smoothing']})


station_df = bike_df[['station_id','station_name']].drop_duplicates(keep='first')
station_df = station_df.sort_values('station_name', ascending=True)
station_df.reset_index(inplace=True,drop=True)

station_df.rename(columns = {'station_id':'label',
                             'station_name':'value'}, inplace = True)




############################################################################
##
## Display
##
############################################################################


app.layout = html.Div(id = 'parent', children = [
    html.H1(id = 'H1', children = 'DATA608 Final Project', style = {'textAlign':'center',
                                            'marginTop':20,'marginBottom':20,
                                            'marginLeft':20,'marginRight':20}),


    html.Div([
        dmc.LoadingOverlay(
            html.Div([dcc.Graph(id = 'forcast_plot')] ,
                     style = {'textAlign':'center','marginTop':10,'marginBottom':10}),
        ),
    ]),  
    
    
 
   
   html.Div([
       html.Div([
           html.Div(['Select % of Training Data To Use:'], style={'text-align':'left'}),
           dcc.Slider(id='train_per', value=100, min=50, max=100, step=10),
        ], style ={'width':'40%','display':'inline-block'} ),  
       
       html.Div([
           html.Div(['Select Prediction Range:'], style={'text-align':'left'}),
           dcc.Slider(id='pred_per', value=100, min=20, max=100, step=20),
        ], style ={'width':'40%', 'display':'inline-block', } ), 
    ],style={'textAlign':'center','marginTop':20,'marginBottom':20,'marginLeft':20,'marginRight':20,'font-size': 10,}),
    
 
   html.Div([
       html.Div([
           html.Div(['Select Model:'], style={'text-align':'left'}),
           html.Div( dcc.Dropdown(options=model_df.set_index('label')['value'].to_dict(), id='model_id', 
                                  value='3', clearable=False),),
        ], style ={'width':'30%','display':'inline-block'} ),  
       
       html.Div([
           html.Div(['Select Station:'], style={'text-align':'left'}),
           html.Div( dcc.Dropdown(options=station_df.set_index('label')['value'].to_dict(), id='station_id',
                                  value='6266.06', clearable=False),),
        ], style ={'width':'30%', 'display':'inline-block', } ),   
    ],style={'textAlign':'center','marginTop':20,'marginBottom':20,'marginLeft':20,'marginRight':20,'font-size': 10,}),   
   
   
   
   html.Div([html.Button('Predict', id='button', n_clicks=0),]),
  
    
    html.Div([
        dmc.Stack(
            children=[
                dmc.Divider(variant="solid"),
                dmc.Divider(variant="dashed"),
                dmc.Divider(variant="dotted"),
            ],style={'textAlign':'center','marginTop':20,'marginBottom':20,'marginLeft':20,'marginRight':20,'font-size': 10,}
        ),
    ]),
    
              
    html.Div([
        html.Div([
            dmc.Table(
                id = 'bike_table',
                striped=True,
                highlightOnHover=True,
                withBorder=False,
                withColumnBorders=True,
        ),], style={'width': '95%', 'float': 'center', 'display': 'inline-block','textAlign':'center'}),
    ]),  
        
    ],style = {'textAlign':'center','marginTop':20,'marginBottom':20,'marginLeft':20,'marginRight':20,
              'font-size': 10,})
   
               
               
############################################################################
##
## Callback
##
############################################################################              
    

@app.callback(
    Output("bike_table", "children"),
    Output("forcast_plot", "figure"),
    Input("station_id", "value"),
    Input("train_per", "value"),
    Input("pred_per", "value"),
    Input("model_id", "value"),
    Input('button', 'n_clicks'),
)
def graph_update(station_id,train_per,pred_per,model_id,n_clicks):
    
    
    # Look for submit button
    if not "button" == ctx.triggered_id:
        fig = px.line(height=400)
        fig.update_layout(template="simple_white", title="Citi Bike Forecast")
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return None, fig
    
    
    model = model_df['value'][model_df['label'] == model_id].iloc[0]
    station = station_df['value'][station_df['label'] == station_id].iloc[0]
    title_str = station + ' - ' + model + ' [ train: ' + str(train_per) + '%, pred: ' + str(pred_per) + '%]' 
    
    
    # filter station
    b_df = bike_df[(bike_df['station_id'] == station_id)].copy()
    
    
    
    # model selection and predictioin
    if model_id == '1':
        pred_df = run_arima_model(b_df,train_per,pred_per)
    elif model_id == '2':
        pred_df = run_prophet_model(b_df,train_per,pred_per)
    else:
        pred_df = run_ets_model(b_df,train_per,pred_per)
    
    
    
    # create graph
    fig = px.line(height=400)
    fig.add_scatter(x=pred_df['ride_date'], y=pred_df['bikes_chng'],
                    marker=dict(size=20, color="lightgray"), name='actual')
    fig.add_scatter(x=pred_df['ride_date'], y=pred_df['bikes_chng_pred'], name='forecast')
    fig.update_layout(template="simple_white", title=title_str)
    
    
    
    # create daily dataframe and table
    pred_day_df = process_bike_table(pred_df)
    bike_table = create_table(pred_day_df)

    
    return bike_table, fig
    
   

if __name__ == '__main__': 
    app.run_server()
    
    



    
    