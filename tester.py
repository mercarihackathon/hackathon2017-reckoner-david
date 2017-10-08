from flask import Flask, request, url_for
import random

app = Flask(__name__)
app.secret_key = 'This is really unique and secret'

@app.route('/')
def hello_person():
    return """
        <p>Enter your complaint?</p>
        <form method="POST" action="%s"><input name="person" /><input type="submit" value="Go!" /></form>
        """ % (url_for('greet'),)

@app.route('/greet', methods=['POST'])
def greet():
    data = request.form['person']
    result = "No Significance"
	if("Delhi" in data):
		result = "RoadWork Department"
		print "Result-Departmental: " ,result
	if("Mumbai" in data):
		result = "P & S"
		print "Result-Departmental: " ,result
	if("delhi" in data):
		result = "QUERY IS ALREADY DONE"
		print "Result-Departmental: " ,result
		continue;
	if("Sahar" in data and "pavement" not in data):
		result = "RoadWork Department"
		print "Result-Departmental: " ,result
	if("Sahar" in data and "pavement" in data):
		result = "Sanitation and Health"
		print "Result-Departmental: " ,result

	Data_of_elec = "power , lamp , pole  ,  electrical  ,  help  ,  remove  ,  wire  ,  fallen  ,  sparks "
	# Data_of_elec = ["power" , "lamp" , "pole" , "electrical" , "help" , "remove" , "wire" , "fallen" , "sparks" ]
	Data_of_road = "pothole  ,  manhole  ,  sewer  ,  bump  ,  car  ,  tree  ,  fallen  ,  damage"
	Data_of_trafic = "cars  ,  jam  ,  lot  ,  time  ,  stuck  ,  traffic"

	dataroad = 0;
	dataelec = 0;
	datatrafic = 0;
	listdata = data.split();
	for a in listdata:
		if(a in Data_of_elec):
			dataelec+=1;
		if(a in Data_of_road):
			dataroad+=1;
		if(a in Data_of_trafic):
			datatrafic+=1;
	datatrafic+=1;
	dataelec+=1
	dataroad+=1
	sum = dataelec + datatrafic + dataroad
	elecval = dataelec*0.84/sum
	roadval = dataroad*0.69/sum
	traficval = datatrafic*0.476/sum
	
    return """
        <p>Result-(Departmental)%s!</p>
        <p>p-value significane Elec --> %s , Road --> %s , Traffic --> %s</p>
        <p><a href="%s">Back to start</a></p>
        """ % (result, elecval, roadval , traficval, url_for('hello_person'))

