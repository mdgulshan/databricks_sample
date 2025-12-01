import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the dataset and train the model
def train_and_log_model():
    # Dummy data
    data = {
        'feature1': [1, 2, 3, None, 5],
        'feature2': [5, 4, 3, 2, 1],
        'target': [0, 1, 0, 1, 0]
    }
    
    df = pd.DataFrame(data)
    df.fillna(df.mean(), inplace=True)

    # Preprocess
    scaler = StandardScaler()
    df[['feature1', 'feature2']] = scaler.fit_transform(df[['feature1', 'feature2']])
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=100, max_depth=5)
        model.fit(train_df.drop("target", axis=1), train_df["target"])

        predictions = model.predict(test_df.drop("target", axis=1))
        accuracy = accuracy_score(test_df["target"], predictions)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log and register the model
        mlflow.sklearn.log_model(model, "random_forest_model")
        model_uri = f"runs:/{run.info.run_id}/random_forest_model"
        mlflow.register_model(model_uri, "RandomForestModel")

if __name__ == "__main__":
    train_and_log_model()
