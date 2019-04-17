import os
from flask import Flask, request, render_template, g, redirect, Response

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)


@app.route('/')
def index():
	return render_template("index.html")

@app.route('/userstudy.html')
def user():
	return render_template("userstudy.html")

@app.route('/demo.html')
def demo():
	return render_template("demo.html")

@app.route('/index.html')
def home():
	return render_template("index.html")

@app.route('/results.html')
def res():
	return render_template("results.html")

@app.route('/results-final.html')
def res_final():
	return render_template("results-final.html")


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
