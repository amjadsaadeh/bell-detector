import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import json
import dvc.api
from dvclive.xgb import DVCLiveCallback
from dvclive import Live
from sklearn.preprocessing import LabelEncoder

def prepare_data(df):
    # Convert MFCC features to 1D arrays
    X = np.vstack([x.flatten() for x in df['mfcc_features']])
    # Convert labels to binary (background=0, non-background=1)
    le = LabelEncoder()
    # Convert to inary problem
    df['label'] = df['label'].apply(lambda x: 'background' if x == 'background' else 'bell')
    y = le.fit_transform(df['label'])
    return X, y, le

def main():
    params = dvc.api.params_show()
    model_params = params['model']
    
    # Load data
    df = pd.read_hdf('./data/mffc_data.h5', key='data')
    X, y, le = prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['training']['test_size'], random_state=42, stratify=y
    )
    
    with Live() as live:
        # Train model
        model = xgb.XGBClassifier(objective='binary:logistic',
                                callbacks=[DVCLiveCallback()],
                                **model_params)
        model.fit(
            X_train, 
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=True
        )
        
        # Evaluate and save metrics
        y_pred = model.predict(X_test)
        y_pred_decoded = le.inverse_transform(y_pred)
        y_test_decoded = le.inverse_transform(y_test)
        live.log_sklearn_plot("confusion_matrix", y_test_decoded, y_pred_decoded)
        # Save model
        model_path = Path('./models/xgboost_model.json')
        model_path.parent.mkdir(exist_ok=True)
        model.save_model(model_path)
        live.log_artifact(model_path)

if __name__ == '__main__':
    main()
