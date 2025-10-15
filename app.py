from flask import Flask, render_template, url_for, redirect, request, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from recommender import hybrid_recommend

# === Load classifier model ===
with open(r"C:\Users\VAGDEVI\Desktop\hybrid_recommender\classifier_model.pkl", "rb") as f:
    classifier = pickle.load(f)

with open(r"C:\Users\VAGDEVI\Desktop\hybrid_recommender\image_features_all.pkl", "rb") as f:
    image_features = pickle.load(f)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'

# === Initialize extensions ===
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# === Load product dataset ===
CSV_FILE_PATH = r"C:\Users\VAGDEVI\Desktop\hybrid_recommender\data\ecom_dataset.csv"
df = pd.read_csv(CSV_FILE_PATH)
df["ProdID"] = df["ProdID"].astype(str)

# === Models ===
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)
    search_history = db.Column(db.Text)

# === Forms ===
class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Register')

    def validate_username(self, username):
        if User.query.filter_by(username=username.data).first():
            raise ValidationError('That username already exists. Please choose a different one.')

class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Login')

# === Login loader ===
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# === Helper ===
def get_user_search_history(user):
    if not user.search_history:
        return []
    return user.search_history.split(',')

# === Routes ===
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash("Uh oh! Wrong credentials. Try again!", "error")
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/track_click', methods=['POST'])
@login_required
def track_click():
    prod_id = request.form.get('prod_id')
    if prod_id:
        history_list = get_user_search_history(current_user)
        if prod_id in history_list:
            history_list.remove(prod_id)
        history_list.insert(0, prod_id)
        history_list = history_list[:10]
        current_user.search_history = ','.join(history_list)
        db.session.commit()
    return '', 204

def multi_hybrid_recommend(prod_ids, top_n=10):
    all_scores = []

    for pid in prod_ids:
        try:
            recs = hybrid_recommend(pid, top_n=20)
            all_scores.extend(recs)
        except Exception as e:
            print(f"Error getting recommendations for {pid}: {e}")
            continue

    if not all_scores:
        return []

    recs_df = pd.DataFrame(all_scores).drop_duplicates(subset="ProdID")
    recs_df["ProdID"] = recs_df["ProdID"].astype(str)
    recs_df = recs_df[~recs_df["ProdID"].isin(prod_ids)]
    
    valid_ids = df["ProdID"].astype(str).unique()
    recs_df = recs_df[recs_df["ProdID"].isin(valid_ids)]

    merged = recs_df.merge(df, on="ProdID", how="left")

    if "Name" not in merged.columns or "Description" not in merged.columns:
        print("⚠️ 'Name' or 'Description' missing after merge.")
        merged["Name"] = ""
        merged["Description"] = ""

    tfidf = TfidfVectorizer(max_features=100)
    text_data = merged[["Name", "Description"]].fillna('').agg(' '.join, axis=1)

    if text_data.str.strip().replace('', np.nan).dropna().empty:
        print("❌ Text data is empty or only contains stopwords.")
        return []

    tfidf_matrix = tfidf.fit_transform(text_data).toarray()

    image_vectors = []
    for pid in merged["ProdID"]:
        vec = image_features.get(pid)
        if vec is None or not isinstance(vec, (np.ndarray, list)):
            image_vectors.append(np.zeros(512, dtype=np.float32))
        else:
            image_vectors.append(np.array(vec, dtype=np.float32))
    image_df = pd.DataFrame(image_vectors)

    combined = pd.concat([pd.DataFrame(tfidf_matrix), image_df], axis=1)
    combined = combined.apply(pd.to_numeric, errors='coerce').fillna(0)

    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)

    predictions = classifier.predict(combined_scaled)
    merged["PredictedInterest"] = predictions

    filtered = merged[merged["PredictedInterest"] == 1]
    filtered = filtered.sort_values(by="HybridScore", ascending=False)

    return filtered.head(top_n).to_dict(orient="records")

@app.route('/dashboard')
@login_required
def dashboard():
    products = df.sample(10).to_dict(orient="records")
    search_history = get_user_search_history(current_user)
    top_n = 8
    if search_history:
        recommended_products = multi_hybrid_recommend(search_history, top_n=top_n)
    else:
        recommended_products = df.sort_values(by="stars", ascending=False).head(top_n).to_dict(orient="records")
    return render_template('dashboard.html', user=current_user, products=products, recommended_products=recommended_products)

@app.route('/search-results', methods=['POST'])
@login_required
def search_results():
    search_query = request.form.get("search", "").strip().lower()
    sort_by = request.form.get("sort_by", "")
    matching_products = df[
        df["Name"].str.contains(search_query, case=False, na=False) |
        df["Category"].str.contains(search_query, case=False, na=False) |
        df["Description"].str.contains(search_query, case=False, na=False)
    ]
    if sort_by == "price_asc":
        matching_products = matching_products.sort_values(by="Price", ascending=True)
    elif sort_by == "price_desc":
        matching_products = matching_products.sort_values(by="Price", ascending=False)
    elif sort_by == "rating_desc":
        matching_products = matching_products.sort_values(by="stars", ascending=False)

    return render_template("search_results.html", user=current_user, search_query=search_query, products=matching_products.to_dict(orient="records"))

@app.route('/product/<prodid>')
@login_required
def product_page(prodid):
    product = df[df["ProdID"] == prodid].to_dict(orient="records")
    if not product:
        return "Product not found", 404
    product = product[0]
    recommendations = hybrid_recommend(prodid)
    recommended_ids = [prod["ProdID"] for prod in recommendations if isinstance(prod, dict)]
    recommended_products = df[df["ProdID"].isin(recommended_ids)].to_dict(orient="records")
    return render_template('product.html', user=current_user, product=product, recommendations=recommended_products)

@app.route('/recommendations')
@login_required
def get_recommendations():
    prod_id = request.args.get("prod_id")
    top_n = int(request.args.get("top_n", 5))
    recommended_products = hybrid_recommend(prod_id=prod_id, top_n=top_n)
    recommendations_list = []
    for prod in recommended_products:
        product_data = df[df["ProdID"] == prod["ProdID"]].to_dict(orient="records")
        if product_data:
            recommendations_list.append({
                "ProdID": prod["ProdID"],
                "Name": product_data[0]["Name"],
                "ImageURL": url_for('static', filename=f'images/{prod["ProdID"]}.jpg'),
                "HybridScore": prod["HybridScore"]
            })
    return jsonify(recommendations_list)

if __name__ == "__main__":
    app.run(debug=True)