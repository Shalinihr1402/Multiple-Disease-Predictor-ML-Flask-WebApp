from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
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
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    disease_type = db.Column(db.String(50), nullable=False)
    result = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ---------------- HOME ---------------- #
@app.route('/')
def home():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
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
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()

        if user and bcrypt.check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('home'))
        else:
            return render_template('login.html', message="Invalid username or password")

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

    # ✅ Save to database
    new_prediction = Prediction(
        username=current_user.username,
        disease_type="Diabetes",
        result=display_text
    )
    db.session.add(new_prediction)
    db.session.commit()

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

    # ✅ Save to database
    new_prediction = Prediction(
        username=current_user.username,
        disease_type="Parkinsons",
        result=display_text
    )
    db.session.add(new_prediction)
    db.session.commit()

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

    # ✅ Save to database
    new_prediction = Prediction(
        username=current_user.username,
        disease_type="Heart Disease",
        result=display_text
    )
    db.session.add(new_prediction)
    db.session.commit()

    return render_template('heartdisease.html', output_text="Result: {}".format(display_text))
@app.route('/dashboard')
@login_required
def dashboard():
    total_users = User.query.count()
    total_predictions = Prediction.query.count()

    diabetes_count = Prediction.query.filter_by(disease_type="Diabetes").count()
    parkinsons_count = Prediction.query.filter_by(disease_type="Parkinsons").count()
    heart_count = Prediction.query.filter_by(disease_type="Heart Disease").count()

    recent_predictions = Prediction.query.order_by(Prediction.timestamp.desc()).limit(5).all()

    return render_template(
        "dashboard.html",
        total_users=total_users,
        total_predictions=total_predictions,
        diabetes_count=diabetes_count,
        parkinsons_count=parkinsons_count,
        heart_count=heart_count,
        recent_predictions=recent_predictions
    )
# ---------------- CREATE DATABASE ---------------- #
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)