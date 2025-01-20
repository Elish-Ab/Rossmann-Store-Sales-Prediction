import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


def load_data(train_path, test_path):
    """Load train and test datasets."""
    train = pd.read_csv(train_path, parse_dates=['Date'])
    test = pd.read_csv(test_path, parse_dates=['Date'])
    return train, test


def preprocess_data(train, test):
    """Preprocess train and test datasets."""
    # Handle missing values
    train['Open'] = train['Open'].fillna(1)
    test['Open'] = test['Open'].fillna(1)

    # Ensure 'CompetitionDistance' exists
    train['CompetitionDistance'] = train.get('CompetitionDistance', 0)
    test['CompetitionDistance'] = test.get('CompetitionDistance', 0)

    # Fill missing categorical columns with 'Unknown'
    categorical_columns = ['StoreType', 'Assortment', 'PromoInterval']
    for col in categorical_columns:
        if col not in train.columns:
            train[col] = 'Unknown'
        if col not in test.columns:
            test[col] = 'Unknown'

    # Extract date features
    for df in [train, test]:
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['DayOfWeek'] = df['Date'].dt.dayofweek

    # Filter only open stores for training
    train = train[train['Open'] == 1]

    return train, test


def build_pipeline():
    """Build a pipeline for preprocessing and modeling."""
    numeric_features = ['CompetitionDistance', 'DayOfWeek', 'Promo', 'Year', 'Month', 'Day', 'WeekOfYear']
    categorical_features = ['StoreType', 'Assortment', 'PromoInterval']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    return pipeline


def train_evaluate_model(pipeline, train):
    """Train and evaluate the model using cross-validation."""
    X = train[['CompetitionDistance', 'DayOfWeek', 'Promo', 'Year', 'Month', 'Day', 'WeekOfYear', 
               'StoreType', 'Assortment', 'PromoInterval']]
    y = train['Sales']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [10, 20, None],
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=3)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_val)

    print("Best Parameters:", grid_search.best_params_)
    print("Mean Absolute Error:", mean_absolute_error(y_val, y_pred))
    print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_val, y_pred)))

    return best_model


def make_predictions(model, test):
    """Make predictions on the test set."""
    X_test = test[['CompetitionDistance', 'DayOfWeek', 'Promo', 'Year', 'Month', 'Day', 'WeekOfYear', 
                   'StoreType', 'Assortment', 'PromoInterval']]

    test['Sales'] = model.predict(X_test)
    return test[['Id', 'Sales']]


def save_predictions(predictions, output_path):
    """Save predictions to a CSV file."""
    predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def main():
    train_path = 'train.csv'
    test_path = 'test.csv'
    output_path = 'predictions.csv'

    # Step 1: Load data
    train, test = load_data(train_path, test_path)

    # Step 2: Preprocess data
    train, test = preprocess_data(train, test)

    # Step 3: Build and train model
    pipeline = build_pipeline()
    best_model = train_evaluate_model(pipeline, train)

    # Step 4: Make predictions
    predictions = make_predictions(best_model, test)

    # Step 5: Save predictions
    save_predictions(predictions, output_path)


if __name__ == "__main__":
    main()
