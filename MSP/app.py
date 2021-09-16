from flask import Flask, render_template, request
import os
from result import output
import tensorflow as tf
from logger import logs


app = Flask(__name__)
UPLOAD_FOLDER = 'C:/Users/amiti/Desktop/Master_project/MSP/static'
SAVE_IMG = 'C:/Users/amiti/Desktop/Master_project/MSP/static'

log_it = logs()
log_it.erase_file()

@app.route('/upload-analyse', methods=['GET', 'POST'])
def upload():
    log_it.log(f"{'='*50} Starting the log {'='*50}")
    log_it.log("=====In the '/' route .=====")
    if request.method == 'POST':
        log_it.log(f"=====When the method is POST=====")
        image_file = request.files['image']
        log_it.log(f"=====The image has been uploaded.=====")
        if image_file:
            log_it.log(f"=====The image is valid.=====")
            image_file.save(os.path.join(UPLOAD_FOLDER, image_file.filename))
            path = UPLOAD_FOLDER+'/' + image_file.filename
            log_it.log(f"=====The image is saved at {path}.=====")
            get_results = output(path)
            pred_class,accuracy = get_results.get_class()
            log_it.log(f"=====Prediction is  {pred_class}.=====")
            if pred_class:
                lobe_img, grad_img = get_results.ExplainOutput(pred_class)
                log_it.log(f"=====Successfully got explanation results and saving them at {SAVE_IMG}. =====")
                for j,i in enumerate([lobe_img,grad_img]):
                    if j==0:
                        log_it.log(f"=====shape of lobe_img : {i.shape}.=====")
                        tf.keras.utils.save_img(os.path.join(SAVE_IMG,'lobe_img.jpg'),i)

                    else:
                        log_it.log(f"=====shape of grad_img : {i.shape}.=====")
                        tf.keras.utils.save_img(os.path.join(SAVE_IMG, 'grad_img.jpg'), i)
                log_it.log(f"=====Copying the log file and emptying the original file.=====")
                return render_template("results.html", prediction = pred_class, grad_path = 'grad_img.jpg', lobe_path = 'lobe_img.jpg', org_path = image_file.filename, accuracy = 100*accuracy)
            else:
                log_it.log(f"=====The image uploaded is not valid.=====")
                log_it.log(f"=====Copying the log file and emptying the original file.=====")
                return render_template('results.html', prediction=pred_class, gradp_path=None, lobe_path=None, org_path=None, accuracy = accuracy*100)

    log_it.log(f"=====The method is GET not POST.=====")
    log_it.log(f"=====Copying the log file and emptying the original file.=====")
    return render_template('results.html', prediction=None, gradp_path=None, lobe_path=None, org_path=None, accuracy = None)


if __name__ == "__main__":
    app.run(debug=True)
