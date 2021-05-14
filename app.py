import urllib
import numpy as np
import tensorflow as tf
import random

from PIL import Image
from matplotlib import pyplot
from flask import Flask, render_template, request, redirect

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

from model import *


def predict_ela_output():
    X_test = []
    img_path = 'static/uploads/test.jpg'
    ela_path = 'static/uploads/ela_image.jpg'
    X_test.append(np.array(convert_to_ela_image(img_path, 90).resize((128, 128))).flatten() / 255.0)
    X_test = np.array(X_test)
    X_test = X_test.reshape(-1, 128, 128, 3)
    im1 = convert_to_ela_image(img_path, 90)
    im1.save(ela_path)
    reconstructed_model = keras.models.load_model("./train_model/model")
    prediction = reconstructed_model.predict(X_test)
    print("the PREDICTION is")
    print(prediction)
    pred_perc = round(prediction[0][0] * 100, 2)
    return pred_perc


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route("/upload_url", methods=['GET', 'POST'])
def upload_url():
    if request.method == 'POST':
        image_url = request.form['search']
        urllib.request.urlretrieve(image_url, 'static/uploads/test.jpg')
        return render_template("1.html")


@app.route("/upload_file", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        file.save('./static/uploads/test.jpg')
        return render_template('1.html')


@app.route("/upload", methods=['GET', 'POST'])
def upload_for_processing():
    if request.method == 'POST':
        perc = predict_ela_output()
        # return render_template("result.html")
        return render_template('result.html', perc=perc)


if __name__ == '__main__':
    app.run(debug=True)
