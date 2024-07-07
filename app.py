import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import joblib
import pandas as pd

# Load the necessary models and encoders
label_encoder = joblib.load('label_encoder.pkl')
poly = joblib.load('poly.pkl')
kmeans = joblib.load('kmeans_model.pkl')
cluster_model = joblib.load('cluster_model.pkl')
best_model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.Div([
        html.H1("Real Estate Price Predictor", style={'text-align': 'center', 'font-family': 'Arial, sans-serif', 'color': '#4A4A4A', 'margin-bottom': '20px'}),
        
        html.Div([
            html.Div([
                dcc.Input(id='distance_to_mrt', type='number', placeholder='Distance to Metro (m)',
                          style={'width': '100%', 'padding': '10px', 'margin-bottom': '10px', 'border-radius': '5px', 'border': '1px solid #ccc'}),
                dcc.Input(id='num_convenience_stores', type='number', placeholder='Number of Convenience Stores',
                          style={'width': '100%', 'padding': '10px', 'margin-bottom': '10px', 'border-radius': '5px', 'border': '1px solid #ccc'}),
                dcc.Input(id='latitude', type='number', placeholder='Latitude',
                          style={'width': '100%', 'padding': '10px', 'margin-bottom': '10px', 'border-radius': '5px', 'border': '1px solid #ccc'}),
                dcc.Input(id='longitude', type='number', placeholder='Longitude',
                          style={'width': '100%', 'padding': '10px', 'margin-bottom': '10px', 'border-radius': '5px', 'border': '1px solid #ccc'}),
                html.Button('Predict Price', id='predict_button', n_clicks=0,
                            style={'width': '100%', 'padding': '10px', 'background-color': '#28a745', 'color': 'white',
                                   'border-radius': '5px', 'border': 'none', 'font-size': '16px', 'cursor': 'pointer'})
            ], style={'width': '80%', 'margin': '0 auto'}),
        ], style={'text-align': 'center', 'font-family': 'Arial, sans-serif', 'font-size': '16px'}),
        
         html.Div(id='prediction_output', style={'text-align': 'center', 'font-size': '20px', 'margin-top': '20px', 'color': '#4A4A4A'})
    ], style={'max-width': '500px', 'margin': '0 auto', 'border': '2px solid #007f79',
              'padding': '20px', 'border-radius': '10px', 'font-family': 'Arial, sans-serif',
              'background-color': 'white', 'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)'})
], style={'background-color': '#f8f9fa', 'height': '100vh', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})

# Define callback to update output
@app.callback(
    Output('prediction_output', 'children'),
    [Input('predict_button', 'n_clicks')],
    [State('distance_to_mrt', 'value'), 
     State('num_convenience_stores', 'value'),
     State('latitude', 'value'),
     State('longitude', 'value')]
)
def update_output(n_clicks, distance_to_mrt, num_convenience_stores, latitude, longitude):
    if n_clicks > 0 and all(v is not None for v in [distance_to_mrt, num_convenience_stores, latitude, longitude]):
        # Prepare the feature vector
        features = pd.DataFrame([[distance_to_mrt, num_convenience_stores, latitude, longitude]], 
                                columns=['distance_to_the_nearest_mrt_station', 'number_of_convenience_stores', 'latitude', 'longitude'])
        
        # Predict cluster
        cluster = kmeans.predict(features[['latitude', 'longitude']])[0]

        # Encode cluster label
        cluster_encoded = label_encoder.transform([cluster])[0]

        # Use the encoded cluster label as a feature
        features['cluster'] = cluster_encoded
        
        # Apply store density features
        features['distance_to_the_nearest_mrt_station_density'] = cluster_model.predict(features[['cluster', 'distance_to_the_nearest_mrt_station']])[0]
        
        # Dropping excessive features
        features = features.drop(columns=['latitude', 'longitude'])

        # Reorder the dataframe
        features = features[['distance_to_the_nearest_mrt_station', 'number_of_convenience_stores', 'cluster', 'distance_to_the_nearest_mrt_station_density']]
        
        # Applying pretrained StandardScaler
        scaled_features = scaler.transform(features)

        # Applying pretrained PolynomialFeatures model
        poly_scaled_features = poly.transform(scaled_features)
        
        # Predict
        prediction = best_model.predict(poly_scaled_features)[0]
        return f'Predicted House Price of Unit Area: {prediction:.2f}'
    elif n_clicks > 0:
        return 'Please enter all values to get a prediction'
    return ''

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
