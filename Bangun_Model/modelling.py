import mlflow
import dagshub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import os

#konfigurasi DagsHub MLflow
REPO_OWNER = 'KinoVelverika'
REPO_NAME = 'bangun-sistem-ML'

#Int Dagshub Connection
try:
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME)
    mlflow.set_experiment("Eksperimen_VSCode_Knv")
    print("✅ Terhubung ke DagsHub MLflow")
except Exception as e:
    print(f"❌ Gagal terhubung ke DagsHub: {e}")

#Load Data (ngambil data dari folder luar)
try:
    df = pd.read_csv('credit_risk_dataset.csv')
except:
    print("⚠️ File dataset tidak ketemu di '../', membuat dummy data...")
    from sklearn.datasets import make_classification
    X_dummy, y_dummy = make_classification(n_samples=500, random_state=42)
    df = pd.DataFrame(X_dummy, columns=[f'f{i}' for i in range(20)])
    df['approved'] = y_dummy

#Preprocessing singkat
if 'income' in df.columns:
    features = ['income', 'age', 'loan_amount']
    target = 'approved'
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    X = df[features]
    y = df[target]
else:
    # Jika dummy sklearn
    X = df.drop('approved', axis=1)
    y = df['approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#training loop dengan MLflow
estimators_list = [50, 100, 150]

print("🚀 Mulai Training...")

for n in estimators_list:
    with mlflow.start_run(run_name=f"Model_RF_{n}"):
        #latih Model
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        
        #Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        
        #Logging
        # Log Parameter
        mlflow.log_param("n_estimators", n)
        mlflow.log_param("algorithm", "RandomForest")
        
        #Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        
        #Log Model (Artifact utama)
        mlflow.sklearn.log_model(model, "model_output")
        
        print(f"   --> Model n={n} Selesai. Akurasi: {acc:.4f}")

print("🏁 Semua eksperimen selesai! Cek Dashboard DagsHub Anda.")