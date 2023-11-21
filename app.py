from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the final model
with open('model.pkl','rb') as f:
    final_model=pickle.load(f)
with open('preprocessor.pkl','rb') as f:
    preprocessor=pickle.load(f)

def prepare_input_data(category, accident_type, year, month):
    df=pd.DataFrame({'Category':[category],'Type':[accident_type],'Year':[year],'Month':[month]})
    print(df)
    X=preprocessor.transform(df)
    return X

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        category = request.form['category']
        accident_type = request.form['accident_type']
        year = int(request.form['year'])
        month = int(request.form['month'])

        # Prepare input data for prediction
        
        input_data = prepare_input_data(category, accident_type, year, month)
        

        # Make prediction using the final model
        prediction = final_model.predict(input_data)[0]

        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
