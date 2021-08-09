
import numpy as np
from flask import Flask, request, render_template
import joblib as jb
import cv2
import pytesseract
from flask import Flask, render_template
from flask import *

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/heart')
def heart():
    return render_template('heart.html')


@app.route('/kidney')
def kidney():
    return render_template('kidney.html')


@app.route('/stroke')
def stroke():
    return render_template('stroke.html')


@app.route('/hypothyroid')
def hypothyroid():
    return render_template('hypothyroid.html')


@app.route('/covid')
def covid():
    return "<h2>Covid Prediction Page under Making </h2>"


@app.route('/liver')
def liver():
    return "<h2> Liver Page under Making </h2>"


@app.route('/ocr')
def upload():
    return render_template("upload.html")


def predict(values, dic):
    if len(values) == 4:
        model = jb.load('diabetes.joblib')
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    if len(values) == 18:
        model = jb.load('kidney.joblib')
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    if len(values) == 13:
        model = jb.load('heart.joblib')
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    if len(values) == 10:
        model = jb.load('stroke.joblib')
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    if len(values) == 24:
        model = jb.load('hypothyroid_new.joblib')
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]


@app.route("/predict", methods=['POST', 'GET'])
def predictPage():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict(to_predict_list, to_predict_dict)
        if pred == 1:
            pred = '---- You have diabetes ----'
        elif pred == 0:
            pred = "---- You don't have diabetes ----"
    return render_template('diabetes.html', pred=pred)


@app.route('/predictkidney', methods=['GET', 'POST'])
def predictkidney():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict(to_predict_list, to_predict_dict)
        if pred == 1:
            pred = '---- You have kidney disease ----'
        elif pred == 0:
            pred = "---- You don't have kidney disease ----"
    return render_template('kidney.html', predicted=pred)


@app.route('/predictheart', methods=['GET', 'POST'])
def predictheart():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict(to_predict_list, to_predict_dict)
        if pred == 1:
            pred = '---- You have Heart disease ----'
        elif pred == 0:
            pred = "---- You don't have Heart disease ----"
    return render_template('heart.html', predicted=pred)


@app.route('/predictstroke', methods=['GET', 'POST'])
def predictstroke():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict(to_predict_list, to_predict_dict)
        if pred == 1:
            pred = '---- Highly possible to get stroke ----'
        elif pred == 0:
            pred = "---- Low possiblility to get stroke ----"
    return render_template('stroke.html', predicted=pred)


@app.route('/predictHypothyroid', methods=['POST', 'GET'])
def predictHypothyroid():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict(to_predict_list, to_predict_dict)
        if pred == 1:
            pred = '---- You have a symtoms of Hypothyroid ----'
        elif pred == 0:
            pred = "---- You don't have Hypothyroid ----"
    return render_template('hypothyroid.html', predicted=pred)


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        img = cv2.imread(f.filename)
        config = ('-l eng+due --oem 1 --psm 3')
        extracted_text = pytesseract.image_to_string(img,
                                                     config=config)
        splits = extracted_text.splitlines()
        s = "\n".join(splits)

        return render_template("success.html",
                               name=f.filename,
                               lines=s)


if __name__ == "__main__":
    app.run(debug=True)
