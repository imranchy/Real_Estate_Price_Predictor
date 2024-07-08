# Real Estate Price Predictor

This Dash application predicts real estate prices based on user input for 
Distance to the nearest Metro Station, Number of Convenience Stores nearby, 
Latitude, and Longitude. For detailed analysis and additional insights, check out the 
[Jupyter Notebook](Real_Estate_Price_Prediction.ipynb).

## Features

- User-friendly interface to input required data.
- Prediction of Real Estate prices based on the input values.
- Clean and Modern design with a responsive layout.

## Screenshots

### Initial View of the Application
![sc1](https://github.com/imranchy/Real_Estate_Price_Predictor/assets/63488646/dcda4980-4f96-4432-ae60-9d2f82d556d7)

### User Inputs Filled with Example Data and Predicted Price
![sc2](https://github.com/imranchy/Real_Estate_Price_Predictor/assets/63488646/a9ad455e-81e4-4cf1-a45a-26d577d66fea)

### Error Handling in the App
![sc3](https://github.com/imranchy/Real_Estate_Price_Predictor/assets/63488646/25e3dca0-5856-446f-bd69-311188a75396)

## How to Run the Application

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/real_estate_price_predictor.git
   cd real_estate_price_predictor

2. **Install the required packages:**
Ensure you have Python, dash, joblib and pip installed on your machine.

3. **Run the Application:**
   ```bash
   python app.py

4. **Then copy the console link onto a web browser to run the application**.

## Model Information ##
* Label Encoder: Encodes categorical labels as numerical values.
* Polynomial Features: Transforms input features into polynomial features.
* K-Means Clustering: Groups data into clusters based on latitude and longitude.
* Scaler: Standardizes features by removing the mean and scaling to unit variance.
* Best Model: The trained model used to predict the house prices.
  
## File Structure ##
* app.py: Main file to run the Dash application.
* Real estate.csv: Dataset used for the project.
* label_encoder.pkl: Pretrained label encoder.
* poly.pkl: Pretrained polynomial features transformer.
* kmeans_model.pkl: Pretrained K-Means clustering model.
* cluster_model.pkl: Pretrained clustering model for density features.
* best_model.pkl: Pretrained model for predicting house prices.
* scaler.pkl: Pretrained scaler for standardizing features.
* Real_Estate_Price_Predictor.ipynb: Jupyter notebook with additional analysis and insights.
