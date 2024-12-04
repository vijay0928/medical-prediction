from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model from a .pkl file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Extract input features from the JSON (example structure)
        features = [
            data['Unnamed: 0'],
            data['time_in_hospital'],
            data['num_lab_procedures'],
            data['num_procedures'],
            data['num_medications'],
            data['number_diagnoses'],
            data['age_start_range'],
            data['age_end_range'],
            data['gender_encoded'],
            data['insulin_change_encoded'],
            data['diabetes_drug_1_encoded'],
            data['diabetes_drug_2_encoded'],
            data['medication_change_encoded'],
            data['followup_encoded']
        ]

        # Convert input features to numpy array (model expects 2D array)
        input_array = np.array(features).reshape(1, -1)

        # Make prediction using the model
        prediction = model.predict(input_array)

        # Return prediction as JSON response
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__== '__main__':
    # Run the app on port 5000
    app.run(debug=True)