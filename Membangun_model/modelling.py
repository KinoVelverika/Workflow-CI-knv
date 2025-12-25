import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import dagshub

#connection setup
dagshub.init(repo_owner='KinoVelverika', repo_name='Workflow-CI-knv', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/KinoVelverika/Workflow-CI-knv.mlflow")
mlflow.set_experiment("Submission_Akhir_Timothy")

#PREPROCESSING
print("Membaca data...")
df = pd.read_csv('credit_risk_dataset.csv')


df.dropna(inplace=True)

# Tentukan Fitur (X) dan Target (y)
X = df[['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']]
y = df['loan_status']

# membagi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Training dan Tracking
estimators = [50, 100]

for n in estimators:
    with mlflow.start_run(run_name=f"Model_RF_{n}"):
        print(f"Training dengan {n} pohon...")
        
        # Latih Model
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi
        acc = accuracy_score(y_test, model.predict(X_test))
        
        # Log ke DagsHub
        mlflow.log_param("n_estimators", n)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model_rf")
        
        print(f"Selesai! Akurasi: {acc}")

print("Semua proses selesai!")