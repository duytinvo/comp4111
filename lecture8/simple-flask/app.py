from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
	return render_template("home.html")

@app.route('/greeting')
def greeting():
	return "Hello, hope your day is going well"

if __name__ == "__main__":
	app.run(debug=True)