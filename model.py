import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

def train_model():
    # Read TAB separated file
    df = pd.read_csv("data.csv", sep="\t")

    # Convert datetime
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')

    # Feature Engineering
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year

    # Lag feature
    df['lag_1'] = df['DOM_MW'].shift(1)

    df = df.dropna()

    # Train/Test split
    train = df.iloc[:-1000]
    test = df.iloc[-1000:]

    features = ['hour', 'dayofweek', 'month', 'year', 'lag_1']
    target = 'DOM_MW'

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train[features], train[target])

    predictions = model.predict(test[features])
    mae = mean_absolute_error(test[target], predictions)

    joblib.dump(model, "model.pkl")

    with open("mae.txt", "w") as f:
        f.write(str(round(mae, 2)))

    print("Model trained successfully!")
    print("MAE:", mae)

if __name__ == "__main__":
    train_model()
