# Import required modules
# Flask is used to create the API server
# request is used to get incoming JSON data
# jsonify is used to return the response in JSON format

from flask import Flask, request, jsonify

# joblib is used to load the pre-trained ML model
import joblib

# numpy is used to handle numerical array operations
import numpy as np

# Initialize the Flask app
# This creates the web app instance
app = Flask(__name__)

# Load the machine learning model
# This happens once when the server starts to avoid loading it again and again for every request
# 'rf_model.pkl' is the trained model file saved earlier using joblib
model = joblib.load('rf_model.pkl')     

# Define a POST API endpoint for prediction
# When client sends a POST request to '/predict', this function will be called
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input JSON data from the request body
        data = request.get_json()

        # Convert the input list into a NumPy array and reshape it
        # reshape(1, -1) means making it suitable for a single row prediction
        input_data = np.array(data['input']).reshape(1, -1)

        # Make prediction using the loaded model
        prediction = model.predict(input_data)

        # Return the prediction as JSON response
        return jsonify({'prediction': int(prediction[0])})
    
    # If there's an error (like wrong input format), return the error message
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the app only if this file is run directly (not imported as module)
# debug=True enables automatic reload on code changes and shows detailed error logs
if __name__ == '__main__':
    app.run(debug=True)
