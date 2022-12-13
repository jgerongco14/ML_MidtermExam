from flask import Flask,render_template,request
# ERROR IMPORTS
import joblib
import numpy as np

app = Flask(__name__)

@app.route("/")
def myapp():
    return render_template('index.html')

naive_model = joblib.load('naivebayes_model.joblib')
dectree_model = joblib.load('decision_tree_model.joblib')

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

    # DILI PA SURE
    if model_class == 'naive_bayes' :
         pred = naive_model.predict(ans)
    
    return render_template('result.html',answer= ans, class_model = model_class, predict=pred)


if __name__ == '__main__':
    app.run(debug=True)







