import os
from pydoc import html

from flask import Flask, flash, request, redirect, url_for, render_template

UPLOAD_FOLDER = 'Media'
ALLOWED_EXTENSION = ['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/home',methods = ['GET'])

def home():
    return render_template('index.html')

@app.route('/login', methods = ['GET'])

def login():
    return render_template('login.html'),

@app.route('/register', methods = ['GET'])

def register():
    return render_template('register.html'), render_template('login.html')

@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            #print(app.config["UPLOAD_FOLDER"]+'/'+ image.filename)
            image.save(app.config["UPLOAD_FOLDER"]+'/'+ image.filename)
            filename = app.config["UPLOAD_FOLDER"]+'/'+ image.filename
            print("stored as:" + filename)
            return render_template("upload.html", uploaded_image=filename)
    return render_template("upload.html")
@app.route('/bg',methods = ['GET','POST'])
def bg():
    return render_template('bg.html')
if __name__ == '__main__':
    app.run(debug=True)

