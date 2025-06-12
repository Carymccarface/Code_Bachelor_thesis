import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer



# Load data
print('----    Loading data    ----')
#data = pd.read_csv('all_features_pp.csv')
data = pd.read_csv("all_features_pp_spin_fix_nonFM.csv")
print('----    Done!    ----')
features_num = data.shape[1]
print(f'----    {features_num} features    ----')


# Split input features and target
print('----    Splitting data to train/test    ----')
data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(
    data.drop(columns=['tot_mag_mom','E_formation','mag_order', 'encoded']), # Remove both outputs
    data['E_formation'], # Choose which output is target mag_mom/energy
    test_size=0.25,
    random_state=2
)
print('----    Done!    ----')


### Uncomment to reduce number of features used
# k = 500  

# # K = 3 XGBoost Test MSE: 0.616, R²: 0.939, random state = 2 using:
#         #  1: jml_jv_enp
#         #  2: jml_oq_enp
#         #  3: Radii gamma 

# # K = 3 XGBoost Test MSE: XGBoost Test MSE: 1.070, R²: 0.893, random state = 1 using:
#         #  1: jml_jv_enp
#         #  2: jml_elec_aff_add_voro_coord
#         #  3: jml_oq_enp


# print(f'----    Finding {k} best features    ----')
# selector = SelectKBest(mutual_info_regression, k=k)
# data_train_x_reduced = selector.fit_transform(data_train_x, data_train_y)
# data_test_x_reduced = selector.transform(data_test_x)
# print('----    Done!    ----')

# # Print the names of selected features
# if k <= 30:
#     selected_feature_names = data_train_x.columns[selector.get_support()]
#     print(f"----    Selected top-{k} features:    ----")
#     for i, fname in enumerate(selected_feature_names, 1):
#         print(f"{i:2d}: {fname}")


### Uncomment to Use all features
data_train_x_reduced = data_train_x
data_test_x_reduced = data_test_x

                    ### --- Random Forest Model --- ###

if 1 == 1: # Set 1 if training RF
    print("\n----    Training Random Forest    ----")

    best_rf_params = {              # Add to list if you find new better
            'n_estimators': 550,    # [200, 250]
            'max_depth': 25,        # [20, 25]
            'max_features': 0.3,    # [sqrt, 0.1]
            'min_samples_split': 3, # [2, 2]
            'random_state': 1 
        }

    if 1 == 0: # Set 1 if Grid Searching
        rf_param_grid = {
            'n_estimators': [350, 550],
            'max_depth': [25],
            'max_features': [ 0.1, 0.3],
            'min_samples_split': [3, 4]
        }

        rf_clf = RandomForestRegressor(random_state=1)

        rf_grid_search = GridSearchCV(
            estimator=rf_clf,
            param_grid=rf_param_grid,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1,
            verbose=1
        )

        rf_grid_search.fit(data_train_x_reduced, data_train_y)

        print("Random Forest best parameters:", rf_grid_search.best_params_)
        print("Random Forest best CV MSE:", -rf_grid_search.best_score_)

        rf_best_model = rf_grid_search.best_estimator_
        rf_y_pred = rf_best_model.predict(data_test_x_reduced)

        rf_mse = mean_squared_error(data_test_y, rf_y_pred)
        rf_r2 = r2_score(data_test_y, rf_y_pred)

        print(f"Random Forest Test MSE: {rf_mse:.3f}, R²: {rf_r2:.3f}")

    else: # Use known best parameters, Should probably do more grid searches
        # Create and train the model
        rf_model = RandomForestRegressor(**best_rf_params)
        rf_model.fit(data_train_x_reduced, data_train_y)

        # Predict on the test set
        rf_y_pred = rf_model.predict(data_test_x_reduced)

        # Evaluate
        rf_mse = mean_squared_error(data_test_y, rf_y_pred)
        rf_r2 = r2_score(data_test_y, rf_y_pred)
        print(f"Random Forest Test MSE: {rf_mse:.3f}, R²: {rf_r2:.3f}")



                    ### --- XGBoost Model --- ### 

            ### Set 1 if training XGB ###
if 1 == 0: 
    print("\n----    Training XGBoost    ----")
    
    ##### From gridsearch
    # For Energy
    # best_params_energy = { 
    #      'n_estimators': 1200,
    #      'max_depth': 4,
    #      'learning_rate': 0.05,
    #      'alpha': 0.05, 
    #      'lambda': 0,}
    # For Magnetism
    best_params_mag = { 
         'n_estimators': 700,
         'max_depth': 4,
         'learning_rate': 0.05,
         'alpha': 0.05, 
         'lambda': 0,}
    
    # Sanitize column names to be compatible with XGBoost
    data_train_x.columns = data_train_x.columns.map(str).str.replace('[\[\]<>]', '', regex=True)
    data_test_x.columns = data_test_x.columns.map(str).str.replace('[\[\]<>]', '', regex=True)

    if 1 == 1: # Set 1 if gridsearching
        xgb_param_grid = {
            'n_estimators': [700, 1000],
            'learning_rate': [0.05, 0.1],
            'max_depth': [4],
                    #Lägga till L1, L2 ?
             'alpha': [0.05,],       # L1 regularization
             'lambda': [0],       # L2 regularization
                    #Subsamples
            #'subsample': [0.6, 0.8, 1.0],
            #'colsample_bytree': [0.6, 0.8, 1.0],
        }

        xgb_clf = XGBRegressor(random_state=1, verbosity=0, tree_method='hist')

        xgb_grid_search = GridSearchCV(
            estimator=xgb_clf,
            param_grid=xgb_param_grid,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1,
            verbose=1
        )

        xgb_grid_search.fit(data_train_x_reduced, data_train_y)

        print("XGBoost best parameters:", xgb_grid_search.best_params_)
        print("XGBoost best CV MSE:", -xgb_grid_search.best_score_)

        xgb_best_model = xgb_grid_search.best_estimator_
        xgb_y_pred = xgb_best_model.predict(data_test_x_reduced)

        xgb_mse = mean_squared_error(data_test_y, xgb_y_pred)
        xgb_r2 = r2_score(data_test_y, xgb_y_pred)

        print(f"XGBoost Test MSE: {xgb_mse:.3f}, R²: {xgb_r2:.3f}")


    else: # Use known best parameters
        # Create and train the model
        xgb_model = XGBRegressor(**best_params_mag)
        xgb_model.fit(data_train_x_reduced, data_train_y,  )    

        # Predict on the test set
        y_pred = xgb_model.predict(data_test_x_reduced)

        # Evaluate
        xgb_mse = mean_squared_error(data_test_y, y_pred)
        xgb_r2 = r2_score(data_test_y, y_pred)

        print(f"XGBoost Test MSE: {xgb_mse:.3f}, R²: {xgb_r2:.3f}")
