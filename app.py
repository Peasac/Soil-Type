from flask import Flask, render_template, request
import numpy as np
import joblib

# Load the model, scaler, and encoders
soil_model = joblib.load('soil_model.pkl')
scaler = joblib.load('scaler_soil.pkl')
label_encoder_crop = joblib.load('label_encoder_crop.pkl')
label_encoder_soil = joblib.load('label_encoder_soil.pkl')

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    if request.method == "POST":
        try:
            # Get input values
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            moisture = float(request.form['moisture'])
            crop_type_input = request.form['crop_type']

            # Encode crop type
            if crop_type_input in label_encoder_crop.classes_:
                crop_type = label_encoder_crop.transform([crop_type_input])[0]
            else:
                error = f"Invalid Crop Type: {crop_type_input}. Please choose from the options."
                return render_template("index.html", error=error, prediction=None)

            # Prepare input data
            input_data = np.array([[temperature, humidity, moisture, crop_type]])
            input_data_scaled = scaler.transform(input_data)

            # Predict soil type
            predicted_soil_type = soil_model.predict(input_data_scaled)
            predicted_soil_name = label_encoder_soil.inverse_transform(predicted_soil_type)[0]
            prediction = f"Predicted Soil Type: {predicted_soil_name}"

        except Exception as e:
            error = f"Error: {e}"
    
    return render_template("index.html", prediction=prediction, error=error, crops=label_encoder_crop.classes_)

if __name__ == "__main__":
    app.run(debug=True)
