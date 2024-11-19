from flask import Flask,render_template
import pickle
from flask import request
import numpy as np
import pandas as pd

model_linear = pickle.load(open('model/linear_pipe.pkl','rb'))
#model_bgd = pickle.load(open('model/batch_pipe.pkl','rb'))
#model_mbgd = pickle.load(open('model/mini_batch_pipe.pkl','rb'))
model_stochastic = pickle.load(open('model/stochastic_pipe.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict_spent():
    avg_session_length = float(request.form.get('Avg. Session Length'))
    time_on_app = float(request.form.get('Time on App'))
    length_of_membership = float(request.form.get('Length of Membership'))

    #prdiction

    if request.form.get('predict_linear'):
        result = model_linear.predict(pd.DataFrame({'Avg. Session Length': avg_session_length, 'Time on App': time_on_app,
                                             'Length of Membership': length_of_membership}, index=[1]))
        pred = 'Linear Regression'
        return render_template('index.html' , result = [result,pred])

    elif request.form.get('predict_stochastic'):
        result = model_stochastic.predict(pd.DataFrame({'Avg. Session Length': avg_session_length, 'Time on App': time_on_app,
                                             'Length of Membership': length_of_membership}, index=[1]))
        pred = 'Sochastic Gradient Decsent'
        return render_template('index.html' , result = [result,pred])

if __name__ == '__main__':
    app.run(debug=True) #host = '0.0.0.0' , port = 8080