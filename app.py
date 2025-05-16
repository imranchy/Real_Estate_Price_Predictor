import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import joblib
import numpy as np
import pandas as pd

# Load all required models and encoders
label_encoder = joblib.load('label_encoder.pkl')
poly = joblib.load('poly.pkl')
kmeans = joblib.load('kmeans_model.pkl')
cluster_model = joblib.load('cluster_model.pkl')
best_model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Real Estate Price Predictor"

# Colors and style helpers
CARD_STYLE = {
    "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.1)",
    "borderRadius": "15px",
    "padding": "20px",
    "marginBottom": "20px",
    "backgroundColor": "#f9f9f9"
}

# App layout
app.layout = dbc.Container([
    html.H1("Real Estate Price Predictor", className="text-center my-4 text-primary"),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H5("Enter Property Details", className="card-title text-center mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("📍 Distance to Nearest MRT Station (meters)"),
                    dbc.Input(id='input-distance', type='number', placeholder="e.g. 300", min=0),
                ]),
                dbc.Col([
                    dbc.Label("🛒 Number of Convenience Stores Nearby"),
                    dbc.Input(id='input-convenience', type='number', placeholder="e.g. 5", min=0),
                ]),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("🌐 Latitude"),
                    dbc.Input(id='input-latitude', type='number', placeholder="e.g. 24.9673", step=0.0001),
                ]),
                dbc.Col([
                    dbc.Label("🌐 Longitude"),
                    dbc.Input(id='input-longitude', type='number', placeholder="e.g. 121.5425", step=0.0001),
                ]),
            ], className="mb-3"),
            dbc.Button("🔍 Predict Price", id='predict-button', color="primary", className="w-100"),
        ], style=CARD_STYLE), width=12, lg=8, className="offset-lg-2")
    ]),
    
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dcc.Loading(
                    id="loading-prediction",
                    type="default",
                    children=html.H4(id='output-prediction', className="text-center text-success m-3")
                )
            ], style={**CARD_STYLE, "backgroundColor": "#eaf7ea"}),
            width=12, lg=8, className="offset-lg-2"
        )
    ])
], fluid=True)

# Prediction callback
@app.callback(
    Output('output-prediction', 'children'),
    Input('predict-button', 'n_clicks'),
    State('input-distance', 'value'),
    State('input-convenience', 'value'),
    State('input-latitude', 'value'),
    State('input-longitude', 'value')
)
def predict_price(n_clicks, distance, convenience, latitude, longitude):
    if not n_clicks:
        return ""
    if None in (distance, convenience, latitude, longitude):
        return "⚠️ Please fill in all the fields to get a prediction."

    # Construct input DataFrame
    df_input = pd.DataFrame([[distance, convenience, latitude, longitude]],
                            columns=['distance_to_the_nearest_mrt_station',
                                     'number_of_convenience_stores',
                                     'latitude', 'longitude'])

    # Predict cluster
    cluster = kmeans.predict(df_input[['latitude', 'longitude']])[0]

    # Encode cluster
    cluster_encoded = label_encoder.transform([cluster])[0]
    df_input['cluster'] = cluster_encoded

    # Predict density
    density = cluster_model.predict(df_input[['cluster', 'distance_to_the_nearest_mrt_station']])[0]
    df_input['distance_to_the_nearest_mrt_station_density'] = density

    # Drop unused columns
    df_input.drop(columns=['latitude', 'longitude'], inplace=True)

    # Reorder columns
    df_input = df_input[['distance_to_the_nearest_mrt_station',
                         'number_of_convenience_stores',
                         'cluster',
                         'distance_to_the_nearest_mrt_station_density']]

    # Scale and transform
    scaled = scaler.transform(df_input)
    poly_features = poly.transform(scaled)

    # Predict
    prediction = best_model.predict(poly_features)[0]
    return f"💰 **Predicted Price:** {prediction:.2f} NTD per unit area"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
