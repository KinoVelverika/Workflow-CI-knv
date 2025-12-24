import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def data_preprocessing(file_path):

    #Load Data
    df = pd.read_csv(file_path)
    
    #Preprocessing (Scaling Fitur Numerik)
    scaler = StandardScaler()
    features = ['income', 'age', 'loan_amount', 'credit_score']
    
    if all(col in df.columns for col in features):
        df[features] = scaler.fit_transform(df[features])
    
    #Splitting Data
    target = 'approved'
    X = df.drop(target, axis=1)
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

#Blok Test
if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test = data_preprocessing('credit_risk_dataset.csv')
        
        print("✅ Otomatisasi Preprocessing Berhasil!")
        print(f"Ukuran X_train: {X_train.shape}")
        print(f"Ukuran X_test:  {X_test.shape}")
        
    except Exception as e:
        print(f"❌ Terjadi Error: {e}")