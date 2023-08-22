from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
import numpy as np
import os
from flask import jsonify
import joblib
import json
import re
from pymongo import MongoClient

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = ''
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['DEBUG'] = True
db = SQLAlchemy(app)

connection_string = ''
client = MongoClient("")
mongodb = client.mongo

class User(db.Model):
    userid = db.Column('userid', db.Integer, primary_key=True)
    first_name = db.Column(db.String(255), nullable=False)
    last_name = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    isDoctor = db.Column(db.Boolean, nullable=False)
    specialid = db.Column(db.Integer, db.ForeignKey('specialization.specialid'))

    def check_password(self, password):
        return self.password == password

class Specialization(db.Model):
    specialid = db.Column('specialid', db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            session['user_id'] = user.userid
            if user.isDoctor == 1:
                return redirect(url_for("doctordashboard"))
            else:
                return redirect(url_for("patientdashboard"))
        else:
            return 'Invalid email or password.'
    else:
        return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        specialization_name = request.form['specialization']

        user = User.query.filter_by(email=email).first()
        if user:
            return 'Email already exists.'

        specialization = Specialization.query.filter_by(name=specialization_name).first()

        new_user = User(
            first_name=first_name,
            last_name=last_name,
            email=email,
            password=password,
            isDoctor=(role == 'doctor'),
            specialid=specialization.specialid if specialization else None
        )
        db.session.add(new_user)
        db.session.commit()
        session['user_id'] = new_user.userid

        if new_user.isDoctor == 1:
            return redirect(url_for("doctordashboard"))
        else:
            return redirect(url_for("patientdashboard"))
    else:
        specializations = Specialization.query.all()
        return render_template('register.html', specializations=specializations)

@app.route("/adddoctor")
def adddoctor():
    return render_template("adddoctor.html")

@app.route("/addpatient")
def addpatient():
    return render_template("addpatient.html")

@app.route("/doctorrecords")
def doctorRecords():
    return render_template("doctorrecords.html", patient_id=session.get('user_id'))

@app.route('/patientrecords')
def patientRecords():
    return render_template('patientrecords.html', doctor_id=session.get('user_id'))

pcosmodel = load_model('models/pcosmodel.h5')
diseasemodel = joblib.load('models/diseasemodel.joblib')
diabetesmodel = joblib.load('models/diabetesmodel.joblib')

@app.route("/doctordashboard")
def doctordashboard():
    return render_template("doctordashboard.html")

@app.route("/patientdashboard")
def patientdashboard():
    return render_template("patientdashboard.html")

@app.route('/get_pcos_response', methods=['POST'])
def get_pcos_response():
    def get_key(val, dic):
        for key, value in dic.items():
             if val == value:
                 return key
        return "key doesn't exist"

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = 'test_pcos_image.jpg'
    file.save(os.path.join(os.getcwd(), filename))

    try:
        image = load_img(os.path.join(os.getcwd(), filename), target_size=(224, 224))
        img = np.array(image)
        img = img / 255.0
        img = img.reshape(1,224,224,3)
        prediction = pcosmodel.predict(img)
        l = {"infected":prediction[0][0], "notinfected":prediction[0][1]}
        j = prediction.max()
        response = get_key(j, l)
        print(response)
        return jsonify({"result": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_disease_response', methods=['POST'])
def get_disease_response():
    input_features = request.form.get('input_features')
    if input_features is None:
        return jsonify({"error": "Missing input features"}), 400

    input_features = re.sub(r'\s*,\s*', ',', input_features)
    input_features = input_features.replace(" ", "_")
    input_features = input_features.split(',')

    feature_dict = {
        # Features used while training diseasemodel
        # "feature_1" : 0, "feature_2" : 0, ..., "feature_132" : 0
    }
    for feature in input_features:
        feature_dict[feature] = 1
    input_array = np.array(list(feature_dict.values())).reshape(1, -1)

    try:
        prediction = diseasemodel.predict(input_array)
        prediction_list = prediction.tolist()
        json_response = jsonify({"result": str(prediction_list[0])})
        return json_response, 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_diabetes_response', methods=['POST'])
def get_diabetes_response():
    pregnancies = int(request.form.get('pregnancies'))
    skin_thickness = float(request.form.get('skinThickness'))
    insulin = float(request.form.get('insulin'))
    diabetes_pedigree_function = float(request.form.get('diabetesPedigreeFunction'))
    test_data = np.array([[pregnancies, skin_thickness, insulin, diabetes_pedigree_function]])
    (mean, std) = joblib.load('models/scaler_params.pkl')
    standardized_test_data = (test_data - mean) / std
    try:
        predictions = diabetesmodel.predict(standardized_test_data)
        if predictions[0] == 1.0:
            output = 'diabetes'
        else:
            output = 'no diabetes'
        json_response = jsonify({"result": output})
        return json_response, 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_patient_records', methods=['POST'])
def get_patient_records():
    data = request.get_json()
    doctor_id = data.get('doctor_id')

    if not doctor_id:
        return jsonify({'error': 'Missing doctor_id in request data'}), 400
    patients_collection = mongodb.patientrecords
    doctor = patients_collection.find_one({'_id': doctor_id})

    if not doctor:
        return jsonify({'error': 'Doctor not found'}), 404

    patient_records = doctor['patients']

    return jsonify({'patient_records': patient_records}), 200

@app.route('/get_doctor_records', methods=['POST'])
def get_doctor_records():
    data = request.get_json()
    patient_id = data.get('patient_id')

    if not patient_id:
        return jsonify({'error': 'Missing patient_id in request data'}), 400
    doctors_collection = mongodb.doctorrecords
    patient = doctors_collection.find_one({'_id': patient_id})

    if not patient:
        return jsonify({'error': 'Patient not found'}), 404

    doctor_records = patient['doctors']

    return jsonify({'doctor_records': doctor_records}), 200

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.config['SECRET_KEY'] = ''
    app.config['SESSION_COOKIE_SECURE'] = True
    app.run(ssl_context='adhoc', host='0.0.0.0', port=0)
