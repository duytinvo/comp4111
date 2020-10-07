from flask import Flask, render_template, request
import pickle


#initialize the flask app
app = Flask(__name__)
#load the saved model (lr-iris.pkl)
model = pickle.load(open('lr-iris.pkl', 'rb'))


#tell Flask what to do when we load the home page of our website
@app.route('/')
def home():
	#when we start the Flask server, it redirects to the home.html file by default 
    return render_template('home.html')

#redirecting the API to predict the iris flower species
# "methods" define all the allowable HTTP requests that you will accept on this webpage 
@app.route('/predict', methods=['POST'])
def predict():

	#the user inputs for the sepal length and petal length
	sepal_length = request.form['a']
	petal_length = request.form['b']
	#cast to real numbers
	x1 = float(sepal_length)
	x2 = float(petal_length)
	#predction with pretrained logistic regression model for the values of each feature
	pred = model.predict([[x1, x2]])
	return render_template('result.html', prediction=pred)

#app.run() will run the web page locally, hosted on your computer
if __name__ == "__main__":
    app.run(debug=True)