import os, subprocess
from flask import Flask, request, render_template, g, redirect, Response
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/uploads'

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/userstudy.html', methods=['GET', 'POST'])
def user():
	if request.method == 'POST':
		file = request.files['file']
		# if user does not select file, browser also
		# submit an empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file:
			filename = secure_filename(file.filename)
			pathname = os.path.join('templates/uploads/', filename)
			file.save(pathname)
			global UPLOADED_FILE
			UPLOADED_FILE = pathname
			print(UPLOADED_FILE)
			return redirect(request.url)

	return render_template("userstudy.html")

@app.route('/demo.html', methods=['GET', 'POST'])
def demo():
	if request.method == 'POST':
		file = request.files['file']
		# if user does not select file, browser also
		# submit an empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file:
			filename = secure_filename(file.filename)
			pathname = os.path.join('templates/uploads/', filename)
			file.save(pathname)
			global UPLOADED_FILE
			UPLOADED_FILE = pathname
			print(UPLOADED_FILE)
			return redirect(request.url)
	return render_template("demo.html")

@app.route('/index.html')
def home():
	return render_template("index.html")

@app.route('/results.html')
def res():
	if UPLOADED_FILE:
		proc = subprocess.Popen(['python3', 'find_match_user_study.py',  'video_database.db', '20170512-110547/20170512-110547.pb', 'hopenet_robust_alpha1.pkl', UPLOADED_FILE], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
		out = proc.communicate()[0]
		out = out.decode('ascii')
		print('-------------------------------------OUTPUT:----------------------------------')
		out = out.split('\n')
		print(out)
		res = {}
		for x in range(2,6):
			file_returned = out[x]
			f_name = file_returned.split('/')[-1]
			res['file_returned_' + str(x)] = "static/" + f_name
			print(file_returned, f_name)

			cmd = "cp " + file_returned + " static/" + f_name
			print(cmd)
			os.system(cmd)

		
	return render_template("results.html", result=res)

@app.route('/results-final.html')
def res_final():
	if UPLOADED_FILE:
		proc = subprocess.Popen(['python3', 'find_match.py',  'video_database.db', '20170512-110547/20170512-110547.pb', 'hopenet_robust_alpha1.pkl', UPLOADED_FILE], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
		out = proc.communicate()[0]
		out = out.decode('ascii')
		print(out)
		out = out.split('\n')
		file_returned = out[1]
		f_name = file_returned_1.split('/')[-1]

		cmd = "cp " + file_returned + " static/" + f_name
		print(cmd)
		os.system(cmd)

		file_returned = 'static/' + f_name

	return render_template("results-final.html", file_returned=file_returned)

@app.route('/handle_data.html', methods=['POST'])
def handle_data():
	render_template("userstudy.html")
	projectpath = request.form['file']
	print(projectpath)


if __name__ == "__main__":
  import click

  @click.command()
  @click.option('--debug', is_flag=True)
  @click.option('--threaded', is_flag=True)
  @click.argument('HOST', default='0.0.0.0')
  @click.argument('PORT', default=8111, type=int)
  def run(debug, threaded, host, port):
    """
    This function handles command line parameters.
    Run the server using:

        python server.py

    Show the help text using:

        python server.py --help

    """

    HOST, PORT = host, port
    print("running on " + HOST + ":" + str(PORT))
    app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)


  run()
