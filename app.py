from flask import Flask, render_template, request
import joblib
import requests

app = Flask(__name__)
model = joblib.load("LogisticRegression.pkl")
endpoint = "http://localhost:1234/invocations"

@app.route('/')
def home():
    result = ''
    return render_template("index.html", **locals())

@app.route('/predict', methods = ["GET","POST"])
def predict():
    sepal_length = float(request.form["sepal_length"])
    sepal_width = float(request.form["sepal_width"])
    petal_length = float(request.form["petal_length"])
    petal_width = float(request.form["petal_width"])
    # result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    # print(f"result {result}")
    # return render_template('index.html', **locals())
    inference_request = {
        "dataframe_records": [[sepal_length,sepal_width,petal_length,petal_width]]
    }
    response = requests.post(endpoint, json=inference_request)
    result = response.text
    print(f"result {result}")
    return render_template('index.html', **locals())

if __name__ == "__main__":
    app.run(debug=True, port= 5002)