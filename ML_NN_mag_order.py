import numpy as np
import pandas as pd
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
import kerastuner as kt
import os
import shutil
from sklearn.feature_selection import mutual_info_classif  #classification variant

# ---- Settings ----
RUN_TUNING = True # Set to False to reuse the saved best model
BEST_MODEL_PATH = "best_nn_model_classification.h5"

# ---- Load and preprocess the data ----
print('----    Loading data    ----')
df = pd.read_csv("all_features_pp_spin_fixed.csv")

X = df.drop(columns=['energy', 'encoded', 'tot_mag_mom', 'E_formation', 'mag_order'])  # exclude target and energy
y = df['encoded']  # new target for classification

# Label encode mag_order
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)
num_classes = y_categorical.shape[1]


print('----    Done!    ----')

# ---- Train-test split ----
data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(
   X,
   y_categorical,
   test_size=0.25,
   random_state=1
)

# ---- Feature scaling ----
scaler = StandardScaler()
data_train_x = scaler.fit_transform(data_train_x)
data_test_x = scaler.transform(data_test_x)

# ---- Model definition function for KerasTuner ----
def build_model(hp):
    model = Sequential()

    # Input layer
    model.add(Dense(
        units=hp.Int('input_units', min_value=32, max_value=864, step=32),
        activation='relu',
        input_dim=data_train_x.shape[1]
    ))

    # Hidden layers
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(
            units=hp.Int(f'layer_{i}_units', min_value=32, max_value=768, step=32),
            activation=hp.Choice(f'layer_{i}_activation', ['relu', 'tanh']),
        ))
        model.add(Dropout(
            rate=hp.Float(f'layer_{i}_dropout', min_value=0.0, max_value=0.5, step=0.025)
        ))
        if hp.Boolean(f'batch_norm_{i}'):
            model.add(BatchNormalization())

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-5, 1e-2, sampling='log')),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ---- Run tuning or reuse model ----
if RUN_TUNING:
    shutil.rmtree('tuner_dir/nn_tuning_random_classification', ignore_errors=True)

    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=150,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name='nn_tuning_random_classification'
    )

    print("----    Searching for the best hyperparameters    ----")
    tuner.search(data_train_x, data_train_y, validation_data=(data_test_x, data_test_y), epochs=10, batch_size=32)
    print("----    Done!    ----")

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save(BEST_MODEL_PATH)
else:
    print("----    Loading best saved model    ----")
    best_model = load_model(BEST_MODEL_PATH)

# ---- Evaluate the model ----
best_model.summary()

y_pred_probs = best_model.predict(data_test_x)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(data_test_y, axis=1)
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')  # Use 'macro' or 'micro' if preferred

print("---- Classification Report ----")
#print(classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_))
print("Accuracy:", accuracy_score(y_true_classes, y_pred_classes))
print(f"Weighted F1 Score: {f1:.3f}")

# ---- Save best hyperparameters ----
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=150,
    executions_per_trial=1,
    directory='tuner_dir',
    project_name='nn_tuning_random_classification'
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

with open("best_hyperparameters_mag_order.txt", "w") as f:
    for hp_name in best_hps.values:
        f.write(f"{hp_name}: {best_hps.get(hp_name)}\n")
