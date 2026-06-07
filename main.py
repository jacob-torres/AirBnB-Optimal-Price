"""Instantiates the Airbnb application."""
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Import Neural Network model
model = load_model(filepath="./models/airbnbpredict_all_data.h5")
# Define the column names in the exact order the model was trained on,
# matching the `name` attributes of the inputs in home.html
cols = ['latitude', 'longitude', 'neighbourhood', 'room_type',
        'minimum_nights', 'number_of_reviews',
        'calculated_host_listings_count', 'availability_365']


@app.route('/')
def index():
    """Homepage view."""
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Get the information from the home.html and returns the predicted value."""
    # Collect the user data in the model's expected column order
    try:
        user_data = [float(request.form[col]) for col in cols]
    except (KeyError, ValueError):
        return render_template('home.html',
                               pred='Please enter a valid number for every field.')

    user_data_np = np.array([user_data])
    # Use model to make prediction
    prediction = model.predict(user_data_np)
    predicted_price = prediction[0][0]

    if predicted_price < 0:
        return render_template('home.html',
                               pred='Your set parameters are out of range something trained model has not seen!')

    elif predicted_price > 2000000:
        return render_template('home.html',
                               pred='Your set parameters are out of range something! Please revisit them.')
    else:
        return render_template('home.html',
                               pred='Your property should be listed at {} in the '
                                    'local currency of the geo '
                                    "location. "
                                    .format(int(predicted_price)))

if __name__ == '__main__':
    app.run(debug=True)
