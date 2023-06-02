# import dependencies for web app
import pickle

import joblib
from flask import Flask, render_template, request

app = Flask(__name__)


# Bind your url using route
@app.route('/', methods=['POST', 'GET'])
# make a function for url
def heart_attack():
    if request.method == 'POST':

        age = float(request.form['Age'])
        sex = int(request.form['sex'])
        cp = int(request.form['chest_pain'])
        trtbps = float(request.form['trtbps'])
        chol = float(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slp = int(request.form['slp'])
        caa = float(request.form['caa'])
        thall = int(request.form['thall'])

        file = open('pickle/model.pkl', 'rb')
        model = pickle.load(file)

        answer = model.predict([[age, sex, cp, trtbps, chol, fbs, restecg, thalach, exang, oldpeak, slp, caa, thall]])

        if 0:
            answer = "!Sorry you face the Heart Attack"
        else:
            answer = '!Congrats you are Fit and Fine and not Face the Heart Attack'

        return render_template('design.html', result=answer)
    return render_template('design.html')


# This Code is for Application to Run

if __name__ == '__main__':
    app.run(host='localhost', port=1000, debug=True)
