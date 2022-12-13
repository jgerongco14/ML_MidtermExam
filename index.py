from flask import Flask,render_template,request
import joblib
import numpy as np

app = Flask(__name__)

@app.route("/")
def myapp():
    return render_template('index.html')

naive_model = joblib.load('assets\\naivebayes_model.joblib')
dectree_model = joblib.load('assets\\decision_tree_model.joblib')

@app.route('/test', methods=['POST'])
def test():
    bp = request.form['bp']
    hcol = request.form['hcol']
    colcheck = request.form['colcheck']
    smoker = request.form['smoker']
    stroke = request.form['stroke']
    heartattk = request.form['heartattk']
    phyact = request.form['phyact']
    fruits = request.form['fruits']
    veggies = request.form['veggies']
    heaveydrinker = request.form['heaveydrinker']
    bmi = request.form['bmi']

    ans = np.array([[bp,hcol,colcheck,smoker,stroke,heartattk,phyact,fruits,veggies,heaveydrinker,bmi]])

    model_class = request.form['model']

    # prediction
    if model_class == 'naive_bayes' :
         pred = naive_model.predict(ans)
    elif model_class == 'decision_tree' :
         pred = dectree_model.predict(ans)


    # convert numeric prediction to word
    # 0 = no diabetes 1 = prediabetes 2 = Has Diabetes
    if pred[0] == 0:
        output = 'No Diabetes'
    elif pred[0] == 1:
        output = 'Prediabetes'
    elif pred[0] == 2:
        output = 'Has Diabetes'
    
    
    return render_template('result.html',answer= ans, class_model = model_class, predict=output)








