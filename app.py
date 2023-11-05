from flask import Flask, render_template,request
import pickle


app = Flask('__name__')

@app.route('/')
def home():
    return render_template('index.html')


scaler = pickle.load(open('models/scaler.pkl','rb'))
RidgeRegressor = pickle.load(open('models/ridge.pkl','rb'))


@app.route('/prediction',methods = ['GET','POST'])
def predictionPage():
    if request.method == "POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        DC = float(request.form.get('DC'))
        ISI = float(request.form.get('ISI'))
        BUI = float(request.form.get('BUI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))


        scaled_data = scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI,Classes,Region]])
        result = RidgeRegressor.predict(scaled_data)
        return render_template('home.html',result=result[0])
        
    else: 
        return render_template('home.html')