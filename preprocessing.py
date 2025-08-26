from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
    
    def fit(self, X, y=None):
        self.q1 = np.percentile(X, 25, axis=0)
        self.q3 = np.percentile(X, 75, axis=0)
        self.iqr = self.q3 - self.q1
        return self
    
    def transform(self, X):
        lower = self.q1 - self.factor * self.iqr
        upper = self.q3 + self.factor * self.iqr
        return np.clip(X, lower, upper)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = pd.DataFrame(X.copy())
        # New feature: Capital net (gain - loss)
        df['capital_net'] = df['capital-gain'] - df['capital-loss']
        # New feature: Hours per week bins
        df['hours_per_week_bin'] = pd.cut(df['hours-per-week'], 
                                         bins=[0, 20, 40, 60, 100], 
                                         labels=['low', 'normal', 'high', 'very_high'])
        return df

def get_preprocessing_pipeline():
    # Numeric features
    numeric_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('clipper', OutlierClipper()),
        ('scaler', StandardScaler())
    ])
    
    # Categorical features
    categorical_features = [
        'workclass', 'education', 'marital-status', 'occupation', 
        'relationship', 'race', 'sex', 'native-country', 'hours_per_week_bin'
    ]
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Full pipeline with feature engineering
    full_pipeline = Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline 