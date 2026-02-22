from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# ---------------- SECURITY CONFIG ---------------- #
app.config['SECRET_KEY'] = 'mysecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ---------------- LOAD MODELS ---------------- #
diabetes_predict = pickle.load(open('diabetes.pkl', 'rb'))
heart_predict = pickle.load(open('heart.pkl', 'rb'))
parkinsons_predict = pickle.load(open('parkinsons.pkl', 'rb'))

# ---------------- USER MODEL ---------------- #
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---------------- HOME ---------------- #
@app.route('/')
def home():
    return render_template("home.html")

# ---------------- AUTH ROUTES ---------------- #
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')

        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()

        if user and bcrypt.check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('home'))

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ---------------- DISEASE PAGES (PROTECTED) ---------------- #
@app.route('/diabetes')
@login_required
def diabetes():
    return render_template("diabetes.html")

@app.route('/parkinsons')
@login_required
def parkinsons():
    return render_template("parkinsons.html")

@app.route('/heartdisease')
@login_required
def heartdisease():
    return render_template("heartdisease.html")

# ---------------- PREDICTIONS ---------------- #
@app.route('/predictdiabetes', methods=['POST'])
@login_required
def predictdiabetes():
    int_features = [x for x in request.form.values()]
    processed_feature = [np.array(int_features, dtype=float)]
    prediction = diabetes_predict.predict(processed_feature)

    if prediction[0] == 1:
        display_text = "This person has Diabetes"
    else:
        display_text = "This person doesn't have Diabetes"

    return render_template('diabetes.html', output_text="Result: {}".format(display_text))

@app.route('/predictparkinsons', methods=['POST'])
@login_required
def predictparkinsons():
    int_features = [x for x in request.form.values()]
    processed_feature = [np.array(int_features, dtype=float)]
    prediction = parkinsons_predict.predict(processed_feature)

    if prediction[0] == 1:
        display_text = "This person has Parkinson's"
    else:
        display_text = "This person doesn't have Parkinson's"

    return render_template('parkinsons.html', output_text="Result: {}".format(display_text))

@app.route('/predictheartdisease', methods=['POST'])
@login_required
def predictheartdisease():
    int_features = [x for x in request.form.values()]
    processed_feature = [np.array(int_features, dtype=float)]
    prediction = heart_predict.predict(processed_feature)

    if prediction[0] == 1:
        display_text = "This person has Heart Disease"
    else:
        display_text = "This person doesn't have Heart Disease"

    return render_template('heartdisease.html', output_text="Result: {}".format(display_text))

# ---------------- CREATE DATABASE ---------------- #
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)