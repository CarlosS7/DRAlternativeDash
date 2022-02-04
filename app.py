from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import uuid as id
import os
import pandas as pd
from cigna_scikit_models import training_loop
import time


from matplotlib.figure import Figure
import base64
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.csv']
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route('/upload')
def upload_file():
   return render_template('upload.html')
	
uuid_id = []
@app.route('/uploader', methods = ['GET', 'POST'])
def uploader_file():
   if request.method == 'POST':
      f = request.files['file']
      os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)
      uuid = str(id.uuid1())
      f.save(os.path.join(app.instance_path, 'uploads', uuid))
      return redirect(url_for('column_selector', uuid=uuid)), uuid_id.append(uuid)

@app.route('/selector/<uuid>')#, methods = ['GET', 'POST'])
def column_selector(uuid):
   #if request.method == 'POST':
   orig_df = pd.read_csv(os.path.join(app.instance_path, 'uploads', str(uuid)))
   vars = list(orig_df.columns)
   return render_template('variable_select.html', vars = vars, uuid=uuid)
  
   
@app.route('/chosen_train', methods = ['GET', 'POST'])
def chosen_train():
   uuid = uuid_id.pop()
   if request.method == 'POST':
      #uuid = request.args.get("uuid")
      orig_df = pd.read_csv(os.path.join(app.instance_path, 'uploads', str(uuid)))
      chosen_var = request.form.get('variable', None)
      results, best_params, classifier_best_test = training_loop(orig_df, chosen_var)
      #time.sleep(7)
      #return render_template('simple2.html',  tables=[results.to_html(classes='data'), classifier_best_test, best_params.to_html(classes='data')], titles= ['na', 'Training Accuracies', "Classifier with Best Test Accuracy: ", "Best Parameters"])
      return render_template('simple2.html')

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
   app.run(run='0.0.0.0', debug = True)