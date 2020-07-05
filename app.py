from flask import Flask, render_template
import joblib

app = Flask(__name__)

# Load ML model
model1 = joblib.load('./notebooks/model_regr.pkl')
model2 = joblib.load('./notebooks/model_dtree_regr.pkl')
# Make prediction - features = ['BEDS', 'BATHS', 'SQFT', 'AGE', 'LOTSIZE', 'GARAGE']
pred1 = model1.predict([[4, 2.5, 3005, 15, 17903.0, 1]])[0][0].round(2)
pred2 = model2.predict([[4, 2.5, 3005, 15, 17903.0, 1]])[0].round(6)
#res = "Linear Regression: "+str(pred1) + "\nDecision Tree Regressor: "+str(pred2)

@app.route('/')
def index():
    return render_template('index.html', LRegr=str(pred1), DTRegr=str(pred2))