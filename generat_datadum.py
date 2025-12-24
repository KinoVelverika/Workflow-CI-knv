import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000
data = {
    'income': np.random.normal(5000, 1500, n),
    'age': np.random.randint(20, 60, n),
    'loan_amount': np.random.randint(1000, 10000, n),
    'credit_score': np.random.randint(300, 850, n),
    'approved': np.random.choice([0, 1], n)
}
df = pd.DataFrame(data)
df.to_csv('credit_risk_dataset.csv', index=False)
print("Dataset dibuat!")