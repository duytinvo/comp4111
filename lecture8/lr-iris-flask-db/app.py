from flask import Flask, render_template, request
import pickle
import numpy as np
import sqlite3


#initialize the flask app
app = Flask(__name__)
#load the saved logistic regression model (lr-iris.pkl)
model = pickle.load(open('lr-iris.pkl', 'rb'))

  
@app.route('/')
def home():
	#when we start the Flask server, it redirects to the home.html file by default 
    return render_template('home.html')

#SQL
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

#connect to db that will show past inputs with the prediction
@app.route('/past')
def past():
	conn = get_db_connection()
	posts = conn.execute('SELECT * FROM posts').fetchall()
	conn.close()
	return render_template('past.html', posts=posts)
	

#redirecting the API to predict the iris flower species
@app.route('/predict', methods=['POST'])
def predict():

	#the user inputs for the sepal length and petal length
	sepal_length = request.form['a']
	petal_length = request.form['b']
	#cast to real numbers
	x1 = float(sepal_length)
	x2 = float(petal_length)
	#prediction with pretrained logistic regression model with test sample
	pred = model.predict([[x1, x2]])
	
	#update the SQLite database 
	conn = get_db_connection()
	if pred == 0:
		prediction = "Setosa"
	else:
		prediction = "Versicolor"
	
	conn.execute("INSERT INTO posts (sepal, petal, prediction) VALUES (?, ?, ?)",(sepal_length,petal_length,prediction))
	conn.commit()
	conn.close()
	return render_template('result.html', prediction=pred)

#app.run() will run the web page locally, hosted on your computer
if __name__ == "__main__":
    app.run(debug=True)