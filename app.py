import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pickle
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc

# Load the trained model and scaler
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Real Estate Price Predictor"

# Define the layout of the app
app.layout = dbc.Container([
    html.H1("Real Estate Price Predictor", className="text-center my-4"),
    dbc.Row([
        dbc.Col([
            dbc.Label("Distance to Nearest MRT Station (meters)"),
            dbc.Input(id='input-distance', type='number', placeholder="Enter distance", min=0),
        ], width=6),
        dbc.Col([
            dbc.Label("Number of Convenience Stores Nearby"),
            dbc.Input(id='input-convenience', type='number', placeholder="Enter number", min=0),
        ], width=6),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Label("Latitude"),
            dbc.Input(id='input-latitude', type='number', placeholder="Enter latitude", step=0.0001),
        ], width=6),
        dbc.Col([
            dbc.Label("Longitude"),
            dbc.Input(id='input-longitude', type='number', placeholder="Enter longitude", step=0.0001),
        ], width=6),
    ], className="mb-3"),
    dbc.Button("Predict Price", id='predict-button', color="primary", className="mb-3"),
    html.H4(id='output-prediction', className="text-success")
], fluid=True)

# Define the callback to update the prediction
@app.callback(
    Output('output-prediction', 'children'),
    Input('predict-button', 'n_clicks'),
    State('input-distance', 'value'),
    State('input-convenience', 'value'),
    State('input-latitude', 'value'),
    State('input-longitude', 'value')
)
def predict_price(n_clicks, distance, convenience, latitude, longitude):
    if n_clicks is None:
        return ""
    if None in (distance, convenience, latitude, longitude):
        return "Please enter all input fields."
    
    # Prepare the input data
    input_data = pd.DataFrame([[distance, convenience, latitude, longitude]],
                              columns=['X2 distance to the nearest MRT station',
                                       'number of convenience stores',
                                       'latitude',
                                       'longitude'])
    # Scale the input data
    scaled_data = scaler.transform(input_data)
    
    # Predict the price
    prediction = model.predict(scaled_data)
    
    return f"Predicted Price: {prediction[0]:.2f} NTD per unit area"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
