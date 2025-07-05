from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('linear_regression_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')  # Create an index.html for the form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read all 13 input features from the form
        features = [
            float(request.form['CRIM']),
            float(request.form['ZN']),
            float(request.form['INDUS']),
            int(request.form['CHAS']),         # usually 0 or 1
            float(request.form['NOX']),
            float(request.form['RM']),
            float(request.form['AGE']),
            float(request.form['DIS']),
            int(request.form['RAD']),
            float(request.form['TAX']),
            float(request.form['PTRATIO']),
            float(request.form['B']),
            float(request.form['LSTAT'])
        ]

        final_input = np.array([features])
        prediction = model.predict(final_input)[0]

        return render_template('index.html', prediction_text=f'Predicted House Price: ₹{prediction * 1e5:,.2f}')  # Scaled

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        features = [
            float(data['CRIM']),
            float(data['ZN']),
            float(data['INDUS']),
            int(data['CHAS']),
            float(data['NOX']),
            float(data['RM']),
            float(data['AGE']),
            float(data['DIS']),
            int(data['RAD']),
            float(data['TAX']),
            float(data['PTRATIO']),
            float(data['B']),
            float(data['LSTAT'])
        ]
        prediction = model.predict([features])[0]
        return jsonify({'estimated_price': prediction * 1e5})  # return in rupees

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)