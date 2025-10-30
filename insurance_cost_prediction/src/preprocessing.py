import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def split_data(df):
    """"
    split the df passed in, and return X_train, X_test, y_train_log, y_test_log.
    The split is done using stratification, and target variable is log-transformed
    """
    full_df = df.copy()
    X = full_df.drop("charges", axis = 1)
    y = full_df["charges"]

    # create stratification bins
    y_bins = pd.qcut(y, q = 5, labels = False, duplicates = "drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        test_size = 0.2,
        random_state = 42,
        stratify = y_bins
    )

    # trainsform the target variable (every processing on the data should be done
    # only after splitting to prevent data leakage)
    # we use np.log1p which is log(1 + x) to avoid errors ig charges = 0
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    return X_train, X_test, y_train_log, y_test_log

def create_features(df):
    """Engineers new features on the input dataframe."""
    # Make a copy to avoid changing the original df
    df_new = df.copy()

    # bmi which is over 30 is considered as obese
    df_new["is_obese"] = (df_new["bmi"] >= 30).astype(int)

    # age_squared = polynomial
    df_new["age_squared"] = df_new["age"] ** 2

    # convert children to categorical 
    df_new["children"] = df_new["children"].astype(str)

    # ----Interaction Switch Features----
    # smoker_x_obese - interaction (shown that smoker + obese have higher leap)
    df_new["smoker_x_obese"] = ((df_new["smoker"] == "yes") & 
                            (df_new["is_obese"] == 1)).astype(int)

    # sex_x_smoker - interaction (shown that smoker + male have higher leap)
    df_new["is_male_x_smoker"] = ((df_new["smoker"] == "yes") & 
                                (df_new["sex"] == "male")).astype(int)

    #region + smoker - interaction
    df_new['smoker_in_northwest'] = ((df_new['smoker'] == 'yes') & 
    (df_new['region'] == 'northwest')).astype(int)

    df_new['smoker_in_southeast'] = ((df_new['smoker'] == 'yes') & 
    (df_new['region'] == 'southeast')).astype(int)
                                        
    df_new['smoker_in_southwest'] = ((df_new['smoker'] == 'yes') &
    (df_new['region'] == 'southwest')).astype(int)

    print("create_features built successfully.")

    return df_new

def build_preprocessor():
    """
    Builds and returns the unfitted preprocessing pipeline.
    """
    numeric_features = ['age', 'bmi', 'age_squared']
    categorical_features = ['sex', 'smoker', 'region', 'children', 'is_obese',
                            'smoker_x_obese', 'is_male_x_smoker', 'smoker_in_northwest',
                            'smoker_in_southeast', 'smoker_in_southwest']

    # create the transformers
    # Numeric Transformer: Scale data
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Categorical Transformer: One-hot encode data
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(
            handle_unknown='ignore', sparse_output=False, drop = "first"
            ))
    ])
    # handle_unknown = "ignore" help our code to not crash when it sees an unknown category
    # it will just create a row of all zeros for the categories it knows
    # sparse_output = False: StandardSclaer output a dense NumPy array, making this into
    # False will also output a standard NumPy array so that they are compatible and
    # easier to read

    # 3. Create the ColumnTransformer
    # This applies the right transformer to the right columns
    process_features = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep any columns not listed (just in case)
    )

    print("process_features built successfully.")

    preprocessor = Pipeline(steps = [
        ("create_features", FunctionTransformer(create_features)),
        ("process_features", process_features)
    ])

    return preprocessor