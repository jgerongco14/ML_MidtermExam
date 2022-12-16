from flask import Flask,render_template,request
import joblib
import numpy as np
from tensorflow import keras

app = Flask(__name__)

@app.route("/")
def myapp():
    return render_template('index.html')

naive_model = joblib.load('assets\\naivebayes_model.joblib')
dectree_model = joblib.load('assets\\decision_tree_model.joblib')
# knn_model = joblib.load('assets\\knn_model.joblib')
# linear_model = joblib.load('assets\\linear_reg.joblib')
# svm_model = joblib.load('assets\\svm.joblib')
ann_model = keras.models.load_model('assets\\model.h5')

@app.route('/test', methods=['POST'])
def test():
    bp = request.form['HighBP']
    hcol = request.form['HighChol']
    colcheck = request.form['CholCheck']
    smoker = request.form['Smoker']
    stroke = request.form['Stroke']
    heartattk = request.form['HeartDiseaseorAttack']
    phyact = request.form['PhysActivity']
    fruits = request.form['Fruits']
    veggies = request.form['Veggies']
    heaveydrinker = request.form['HvyAlcoholConsump']
    bmi = request.form['BMI']

    input = [bp,hcol,colcheck,smoker,stroke,heartattk,phyact,fruits,veggies,heaveydrinker,bmi]
    ans = np.array([[bp,hcol,colcheck,smoker,stroke,heartattk,phyact,fruits,veggies,heaveydrinker,bmi]])
    
    model_class = request.form['model']

    # prediction
    if model_class == 'naive_bayes' :
         pred = naive_model.predict(ans)
         output = convert(pred)
         model_use = 'Naive Bayes'
    elif model_class == 'decision_tree' :
         pred = dectree_model.predict(ans)
         output = convert(pred)
         model_use = 'Decision Tree'
#     elif model_class == 'knn' :
#          pred = knn_model.predict(ans)
#          model_use = 'KNN'
#     elif model_class == 'linear' :
#          pred = linear_model.predict(ans)
#          model_use = 'Linear Regression'
#     elif model_class == 'svm' :
#          pred = svm_model.predict(ans) 
#          model_use = 'SVM'
    elif model_class == 'ann' :
         ans = ans.astype(float)
         model_use = 'ANN'
         pred = ann_model.predict(ans)[0][0]
         if pred < 0.5 :
          output = 'Healthy' 
         else :
          output = 'Not Healthy' 




    # convert numeric prediction to word
    # 0 = no diabetes 1 = prediabetes 2 = Has Diabetes  
    return render_template('index.html',answer= input, class_model = model_use, result=pred, predict=output)


# answer= ans, class_model = model_class,
def convert(pred):
 if pred[0] == 0:
     return 'No Diabetes'
 elif pred[0] == 1:
     return 'Prediabetes'
 elif pred[0] == 2:
     return 'Has Diabetes'





