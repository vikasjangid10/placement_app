from flask import Flask , request , jsonify
import pickle
import numpy as np

model = pickle.load(open('model/model.pkl', 'rb'))


app = Flask(__name__)

@app.route('/')
def home():
    return "hello world"

@app.route('/predict', methods=['POST'])
def predict():
    cgpa = request.form.get('cgpa')
    resume_score = request.form.get('resume_score')

    input_query = np.array([[cgpa,resume_score]])

    result = model.predict(input_query)[0]


    return jsonify({'placement':str(result)})

if __name__ == '__main__':
    app.run(debug=True)