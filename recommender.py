import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE 

warnings.filterwarnings("ignore", category=UserWarning)

# === Load Dataset ===
df = pd.read_csv(r"C:\Users\VAGDEVI\Desktop\hybrid_recommender\data\ecom_dataset.csv")
df = df.dropna(subset=['image_path']).reset_index(drop=True)

# === Load VGG Image Features ===
with open(r"C:\Users\VAGDEVI\Desktop\hybrid_recommender\image_features_all.pkl", "rb") as f:
    image_features = pickle.load(f)

# === Combine Text Fields for TF-IDF ===
df['text'] = df[['Name', 'Description']].fillna('').agg(' '.join, axis=1)

# === TF-IDF Vectorization for Global Content Similarity ===
tfidf = TfidfVectorizer(stop_words='english', dtype=np.float32)
tfidf_matrix = tfidf.fit_transform(df['text'].fillna(''))
content_similarity = cosine_similarity(tfidf_matrix, dense_output=False)

# === Normalize Helper ===
def normalize_scores(scores):
    min_val, max_val = np.min(scores), np.max(scores)
    return np.zeros_like(scores) if min_val == max_val else (scores - min_val) / (max_val - min_val)

# === Image-based similarity using VGG features ===
def get_visual_similarity_vector(prod_idx, df_ref, image_features, threshold=0.94):
    base_pid = df_ref.iloc[prod_idx]['ProdID']
    base_vec = image_features.get(base_pid)
    if base_vec is None:
        return np.zeros(len(df_ref), dtype=np.float32)

    sim_vector = []
    for i in range(len(df_ref)):
        other_pid = df_ref.iloc[i]['ProdID']
        other_vec = image_features.get(other_pid)
        if other_vec is None:
            sim_vector.append(0.0)
        else:
            sim_score = cosine_similarity([base_vec], [other_vec])[0][0]
            sim_vector.append(1.0 if sim_score > threshold else 0.0)
    return np.array(sim_vector, dtype=np.float32)

# === Classifier Training ===
def train_classifier_model(df, image_features):
    df['label'] = (df['stars'] >= 2).astype(int)

    # Text vectorization
    df['text'] = df[['Name', 'Description']].fillna('').agg(' '.join, axis=1)
    tfidf = TfidfVectorizer(max_features=300, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['text']).toarray()
    tfidf_df = pd.DataFrame(tfidf_matrix, index=df.index)

    # ✅ Save vectorizer AFTER fitting
    with open(r"C:\Users\VAGDEVI\Desktop\hybrid_recommender\tfidf_vectorizer.pkl", "wb") as f_vec:
        pickle.dump(tfidf, f_vec)

    # Image features
    image_vectors = []
    for pid in df['ProdID']:
        vec = image_features.get(pid)
        if vec is None or not isinstance(vec, (np.ndarray, list)):
            image_vectors.append(np.zeros(512, dtype=np.float32))
        else:
            image_vectors.append(np.array(vec, dtype=np.float32))
    image_df = pd.DataFrame(image_vectors, index=df.index)

    # Combine features
    combined_features = pd.concat([tfidf_df, image_df], axis=1)
    combined_features = combined_features.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Normalize
    scaler = StandardScaler()
    combined_features = pd.DataFrame(scaler.fit_transform(combined_features), index=combined_features.index)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        combined_features, df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    # === Apply SMOTE Oversampling ===
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train classifier
    clf = XGBClassifier(
        scale_pos_weight=1,  # Using SMOTE, so balanced
        max_depth=5,
        learning_rate=0.1,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss')

    clf.fit(X_train_resampled, y_train_resampled)

    # Predict with tuned threshold
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    #print("\n📊 CLASSIFICATION REPORT (Threshold = 0.5):\n")
    #print(classification_report(y_test, y_pred))

    #auc = roc_auc_score(y_test, y_proba)
    #print(f"🔸 ROC AUC: {auc:.4f}")

    # Threshold sweep
    
    #for t in [0.2, 0.3, 0.4, 0.5]:
    #    y_thresh = (y_proba >= t).astype(int)
    #    print(f"\n🔎 Threshold = {t}")
    #    print(classification_report(y_test, y_thresh))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    # ✅ Save classifier model
    with open(r"C:\Users\VAGDEVI\Desktop\hybrid_recommender\classifier_model.pkl", "wb") as f:
        pickle.dump(clf, f)

    print("✅ Classifier trained and saved.")
    return clf


# === Hybrid Recommendation (Content + Image + Rating) ===
def hybrid_recommend(prod_id, top_n=10):
    if prod_id not in df['ProdID'].values:
        print("❌ Product ID not found in dataset.")
        return []

    idx = df.index[df['ProdID'] == prod_id].tolist()[0]

    # Content Score
    content_scores = normalize_scores(content_similarity[idx].toarray().flatten())

    # Image Score
    visual_scores = get_visual_similarity_vector(idx, df, image_features)

    # Combine Scores
    alpha, beta = 0.7, 0.3
    df_local = df.copy()
    df_local['ContentScore'] = content_scores
    df_local['VisualScore'] = visual_scores
    df_local['HybridScore'] = (alpha * df_local['ContentScore']) + (beta * df_local['VisualScore'])

    # Remove input product
    df_local = df_local[df_local['ProdID'] != prod_id]

    # Apply filtering
    filtered = df_local[df_local['ContentScore'] > 0.01]

    if filtered.empty:
        print("⚠️ No relevant recommendations found.")
        return []

    # Sort by score
    top_final = (
        filtered.sort_values(by=["HybridScore", "stars"], ascending=[False, False])
        .head(top_n)
    )

    print(f"📦 Total candidates: {len(df_local)}")
    print(f"✅ Filtered by content: {len(filtered)}")
    print(f"🌟 Final recommendations: {len(top_final)}")

    return top_final[["ProdID", "Name", "stars", "HybridScore"]].to_dict(orient="records")

if __name__ == "__main__":
    model_path = r"C:\Users\VAGDEVI\Desktop\hybrid_recommender\classifier_model.pkl"
    if not os.path.exists(model_path):
        print("🔧 Training classifier model...")
        classifier_model = train_classifier_model(df, image_features)
    else:
        print("📂 Classifier already exists. Skipping training.")
        
train_classifier_model(df, image_features)


