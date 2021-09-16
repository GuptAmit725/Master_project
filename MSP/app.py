from flask import Flask
from flask import render_template
from flask import request
import os
from result import output
import pickle
import tensorflow as tf


app = Flask(__name__)
UPLOAD_FOLDER = 'C:/Users/amiti/Desktop/Master_project/MSP/static'
SAVE_IMG = 'C:/Users/amiti/Desktop/Master_project/MSP/static'

@app.route('/', methods=['GET', 'POST'])
def upload():
    print('1')
    if request.method == 'POST':
        print('2')
        image_file = request.files['image']
        if image_file:
            print('3')
            image_file.save(os.path.join(UPLOAD_FOLDER, image_file.filename))
            print('4')
            path = UPLOAD_FOLDER+'/' + image_file.filename
            print('5', path)
            get_results = output(path)
            print('6')
            pred_class,accuracy = get_results.get_class()
            print('7')
            if pred_class:
                lobe_img, grad_img = get_results.ExplainOutput(pred_class)
                print(lobe_img.shape, grad_img.shape)

                for j,i in enumerate([lobe_img,grad_img]):
                    if j==0:
                        print(i.shape)
                        tf.keras.utils.save_img(os.path.join(SAVE_IMG,'lobe_img.jpg'),i)

                    else:
                        print(i.shape)
                        tf.keras.utils.save_img(os.path.join(SAVE_IMG, 'grad_img.jpg'), i)

            #print('lobe img : ',lobe_img.shape)
            #print(8)
            #print(f"prediction : {pred_class}")
            #print(9)
            #print(accuracy, pred_class)
                return render_template("results.html", prediction = pred_class, grad_path = 'grad_img.jpg', lobe_path = 'lobe_img.jpg', org_path = image_file.filename, accuracy = 100*accuracy)
            else:
                return render_template('results.html', prediction=pred_class, gradp_path=None, lobe_path=None, org_path=None, accuracy = accuracy*100)
    return render_template('results.html', prediction=None, gradp_path=None, lobe_path=None, org_path=None, accuracy = None)


if __name__ == "__main__":
    app.run(debug=True)
