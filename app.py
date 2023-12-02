from flask import Flask, render_template
import numpy as np
import pandas as pd
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    
    return render_template('index.html', prediction_text='{}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)