import pandas as pd
from sklearn.svm import SVR
from voxel_tbss_process import MLDataProcessor
from voxel_tbss_train import MLModelTrainer
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

class MLModelRunner:
    def __init__(self) -> None:
        self.results = {}
        self.param_grid_dict = {}
        self.models = {}
    
    def parameter_init(self):
        param_grid_svr = {
            'C': [1, 10, 20, 50, 100], 
            'kernel' : ['linear', 'rbf'],
            'gamma' : ['auto', 'scale', 0.01, 0.1],
            'epsilon': [0.01, 0.1, 1] 
        }
        self.models = {
        'SVR': (SVR(), param_grid_svr)
        # 'Random Forest': (RandomForestRegressor(), rf_param_grid),
        # 'Linear Regression': (LinearRegression(),lr_param_grid)
        }
    
    def run_model_for_modality(self, modality, random_seeds_list):
        processor = MLDataProcessor(modality)
        mae_scores, r2_scores = {}, {}
        for random_seed in random_seeds_list:
            mae_scores[random_seed], r2_scores[random_seed] = defaultdict(list), defaultdict(list)
            actual_age_list, predicted_age_list = [], []
            for fold in range(1,11):
                directory = '/imaging/correia/wl05/final_scripts/Voxel/train_test_processed/'
                X_train = pd.read_csv(f"{directory}{modality}_{random_seed}_X_train_fold{str(fold)}_tbss-new.csv")
                X_test = pd.read_csv(f"{directory}{modality}_{random_seed}_X_test_fold{str(fold)}_tbss-new.csv")
                Y_train = pd.read_csv(f"{directory}{modality}_{random_seed}_Y_train_fold{str(fold)}_tbss-new.csv")
                Y_test = pd.read_csv(f"{directory}{modality}_{random_seed}_Y_test_fold{str(fold)}_tbss-new.csv")

                
                X_train, X_test = processor.z_score_normalisation(X_train, X_test)
                Y_group_train = processor.get_age_groups(X_train, Y_train)
                scores_dict, prediction_dict = self.run_across_models(X_train, Y_train, X_test, Y_test, Y_group_train, modality, random_seed, fold)
            
                for key in scores_dict:
                    mae_scores[random_seed][key].append(scores_dict[key][0])
                    r2_scores[random_seed][key].append(scores_dict[key][1])
            
                actual_age_list += prediction_dict['y_test'].values.tolist()
                predicted_age_list += prediction_dict['predicted_brain_age'].tolist()
                print(actual_age_list)
                print(predicted_age_list)
                  
                self.prediction_plot_across_folds(actual_age_list, predicted_age_list, random_seed, modality)  
        return mae_scores, r2_scores
        
    def run_across_models(self, X_train, Y_train, X_test, Y_test, Y_group_train, modality, random_seed, count):
        scores_dict = {} 
        for name, (model, param_grid) in self.models.items():
            trainer = MLModelTrainer(name, model, X_train, X_test, Y_train.values.ravel(), Y_test.values.ravel(), Y_group_train, param_grid)
            trainer.train()
            predicted_brain_age = trainer.predict()
            mae, r2 = trainer.evaluate()
            scores_dict[name]=[mae, r2]
            prediction_dict = {'y_test':Y_test, 'predicted_brain_age': predicted_brain_age}
            
            plt.figure(figsize=(8, 6))
            plt.scatter(Y_test, predicted_brain_age, alpha=0.5)
            plt.title('Predicted Brain Age vs Actual Age for '+ modality + str(random_seed))
            plt.xlabel('Actual Age')
            plt.ylabel('Predicted Brain Age')
            plt.plot([Y_test.min(), Y_test.max()], 
                [Y_test.min(), Y_test.max()], 'r--')  
            plt.grid(True)
            plt.show()
            directory = '/imaging/correia/wl05/final_scripts/Voxel/voxel_prediction_plots'
            filename = name+modality+str(random_seed) + str(count) + '_tbss-new'
            filepath = os.path.join(directory, filename)
            plt.savefig(filepath)
            
        return scores_dict, prediction_dict
    
    def prediction_plot_across_folds(self, actual_age, predicted_age, random_seed, modality):
        plt.figure(figsize=(8, 6))
        plt.scatter(actual_age, predicted_age, alpha=0.5)
        plt.title('Predicted Brain Age vs Actual Age for '+ modality + str(random_seed))
        plt.xlabel('Actual Age')
        plt.ylabel('Predicted Brain Age')
        plt.plot([min(actual_age), max(actual_age)], 
                [min(actual_age), max(actual_age)], 'r--')  
        plt.show()
        directory = '/imaging/correia/wl05/final_scripts/Voxel/voxel_prediction_plots'
        # filename = modality+str(random_seed)
        filename = modality+str(random_seed) + '_tbss-new'
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath)