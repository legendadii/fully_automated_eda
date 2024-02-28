from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler,OneHotEncoder,OrdinalEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd

train = pd.read_csv("C:\\Users\\DELL\\Downloads\\apple_quality.csv")

last_column_name = train.columns[-1]
y_train=train[last_column_name]

train=train.drop(columns=last_column_name)
# train=train.drop(columns='A_id')

if y_train.dtype not in ['int64', 'float64']:
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
else:
    pass


numerical_features = []
categorical_features = []

for column in train.columns:
    if train[column].dtype in ['int64', 'float64']:
        numerical_features.append(column)
    else:
        categorical_features.append(column)

print("Numerical features:", numerical_features)
print("Categorical features:", categorical_features)

correlation_matrix = train[numerical_features].corr().abs()
correlated_features = set()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:  
            colname = correlation_matrix.columns[i]  
            correlated_features.add(colname)


df_filtered = train.drop(columns=correlated_features)

features=df_filtered.columns



print(features)

def remove_outliers_iqr(data, threshold=1.4):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    outliers = (data < lower_bound) | (data > upper_bound)
    cleaned_data = data[~outliers]
    return cleaned_data

def custom_transformer(data):
    return data.apply(remove_outliers_iqr)

numerical_features = features

numerical_transformer = Pipeline(steps=[
    ('imputer', 'passthrough'),
    ('scaler', StandardScaler())
])

categorical_features1=[]
categorical_transformer1 = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder())
])

categorical_features=categorical_features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

custom_transformer = FunctionTransformer(custom_transformer)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('custom', custom_transformer, numerical_features),  
        ('categ', categorical_transformer, categorical_features), 
        ('categ1', categorical_transformer1, categorical_features1), 
        
    ])

smote_tomek = SMOTE(sampling_strategy='minority')
tomek_links = TomekLinks()


clf = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('sampling', smote_tomek),  # Include SMOTE
    ('undersampling', tomek_links),  # Include Tomek Links
    ('classifier', DecisionTreeClassifier())
])

param_grid = {
    'preprocessor__num__imputer': [SimpleImputer(strategy='mean'), SimpleImputer(strategy='median'), KNNImputer(n_neighbors=5), IterativeImputer(max_iter=10, random_state=0)],
    'preprocessor__num__scaler': [StandardScaler(), MinMaxScaler()],
    'preprocessor__custom__func': [remove_outliers_iqr, None],  
}

grid_search = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy')
grid_search.fit(train, y_train)

print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)


