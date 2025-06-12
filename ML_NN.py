import numpy as np
import pandas as pd
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import kerastuner as kt
import os
import shutil
from sklearn.feature_selection import mutual_info_regression


# ---- Settings ----
RUN_TUNING = True  # Set to False to reuse the saved best model
BEST_MODEL_PATH = "best_nn_model.h5"

# ---- Load and preprocess the data ----
print('----    Loading data    ----')
df = pd.read_csv("all_features_pp_spin_fix_FM.csv")


X = df.drop(columns=['energy', 'mag_order', 'tot_mag_mom', 'E_formation', 'encoded'])
y = df['tot_mag_mom']

# mi = mutual_info_regression(X, y, random_state=0)
# mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
# top_features = mi_series.head(30).index.tolist()

# X_top = X[top_features]
print('----    Done!    ----')

# Split data into features and target
data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(
   X, 
    y, 
    test_size=0.25, 
    random_state=1
)

# Standardize the features
scaler = StandardScaler()
data_train_x = scaler.fit_transform(data_train_x)
data_test_x = scaler.transform(data_test_x)

# ---- Define the model architecture function for KerasTuner ----
# def build_model(hp):
#     model = Sequential()
    
#     model.add(Dense(units=hp.Int('input_units', min_value=64, max_value=1024, step=64),
#                     activation='relu', input_dim=data_train_x.shape[1]))

#     for i in range(hp.Int('num_layers', 1, 3)):
#         model.add(Dense(units=hp.Int(f'layer_{i}_units', min_value=64, max_value=1024, step=64),
#                         activation='relu'))
#         model.add(Dropout(hp.Float(f'layer_{i}_dropout', 0.0, 0.5, step=0.05)))
#         model.add(BatchNormalization())

#     model.add(Dense(1))

#     model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-5, 1e-2, sampling='log')),
#                   loss='mean_squared_error')

#    return model

def build_model(hp):
    model = Sequential()
    
    # Input layer
    model.add(Dense(
        units=hp.Int('input_units', min_value=32, max_value=1024, step=32),
        activation='relu',
        input_dim=data_train_x.shape[1]
    ))

    # Hidden layers
    for i in range(hp.Int('num_layers', 1, 3)):  #Up to 3 hidden layers 
        model.add(Dense(
            units=hp.Int(f'layer_{i}_units', min_value=32, max_value=1024, step=32),
            activation=hp.Choice(f'layer_{i}_activation', ['relu', 'tanh']),
        ))
        model.add(Dropout(
            rate=hp.Float(f'layer_{i}_dropout', min_value=0.0, max_value=0.5, step=0.025)
        ))
        if hp.Boolean(f'batch_norm_{i}'):
            model.add(BatchNormalization())

    # Output layer
    model.add(Dense(1))

    # Optimizer with learning rate
    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-5, 1e-2, sampling='log')),
        loss='mean_squared_error'
    )

    return model


# ---- Run tuning or reuse model ----
if RUN_TUNING:
    # Remove existing tuner directory 
    shutil.rmtree('tuner_dir/nn_tuning_random', ignore_errors=True)

    #tuner = kt.Hyperband(
        #build_model,
        #objective='val_loss',
        #max_epochs=20,
        #factor=4,
        #directory='tuner_dir',
        #project_name='nn_tuning')
    tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=150,  # increase for a more exhaustive search
    executions_per_trial=1,  # Number of times to train each config
    directory='tuner_dir',
    project_name='nn_tuning_random')

    print("----    Searching for the best hyperparameters    ----")
    tuner.search(data_train_x, data_train_y, validation_split = 0.2, epochs=15, batch_size=32)
    print("----    Done!    ----")

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save(BEST_MODEL_PATH)
else:
    print("----    Loading best saved model    ----")
    best_model = load_model(BEST_MODEL_PATH)

# ---- Evaluate model ----
best_model.summary()
y_pred = best_model.predict(data_test_x)
mse = mean_squared_error(data_test_y, y_pred)
r2 = r2_score(data_test_y, y_pred)

print(f"Neural Network Test MSE: {mse:.3f}, R²: {r2:.3f}")

# # Recreate the tuner just to load past results
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=150,  # should match what you used before
    executions_per_trial=1,
    directory='tuner_dir',
    project_name='nn_tuning_random'
)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Save them to a file
with open("best_hyperparameters_tot_mag_mom.txt", "w") as f:
    for hp_name in best_hps.values:
        f.write(f"{hp_name}: {best_hps.get(hp_name)}\n")