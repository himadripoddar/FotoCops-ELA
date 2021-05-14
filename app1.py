from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import uuid
import glob
from model import *
import urllib.request
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def photo_detector():
    path = 'static/uploads/'
    uploads = sorted(os.listdir(path),
                     key=lambda x: os.path.getctime(path + x))  # Sorting as per image upload date and time
    uploads = ['uploads/' + file for file in uploads]
    uploads.reverse()
    return render_template("index.html", uploads=uploads)





app.config['UPLOAD_PATH'] = 'static/uploads'   # Storage path


@app.route("/upload", methods=['GET', 'POST'])
def upload_file():  # This method is used to upload files
    if request.method == 'POST':
        f = request.files['file']
        # print(f.filename)
        # f.save(secure_filename(f.filename))
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        return redirect("/")  # Redirect to route '/' for displaying images on front end

@app.route("/upload-url", methods=['GET', 'POST'])
def upload_url():
    if request.method == 'POST':
        image_url = request.form['fname']
        image_path = "./static/uploads/" + str(uuid.uuid4()) + ".jpg"
        # print(image_url)
        urllib.request.urlretrieve(image_url, image_path)
        return redirect("/")


@app.route("/result")
def results():
    img_path = request.args.get('img')
    ela_path = './static/ela_images/' + img_path[7:-4]+'-ela.jpg'
    img_path = './static/'+img_path
    print(img_path[9:])
    X_test = []
    X_test.append(np.array(convert_to_ela_image(img_path, 90).resize((128, 128))).flatten() / 255.0)
    X_test = np.array(X_test)
    X_test = X_test.reshape(-1, 128, 128, 3)
    im1 = convert_to_ela_image(img_path, 90)

    im1.save(ela_path)
    reconstructed_model = keras.models.load_model("./train_model/model")

    prediction = reconstructed_model.predict(X_test)
    pred_perc = round(prediction[0][1] * 100, 2)
    print(pred_perc)

    return render_template("result.html", img_path=img_path[9:], ela_path=ela_path[9:], perc=pred_perc)


if __name__ == '__main__':
    app.run(debug=True)
