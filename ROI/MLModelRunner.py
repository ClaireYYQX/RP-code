import pandas as pd
from sklearn.svm import SVR
from MLDataProcessor import MLDataProcessor
from MLModelTrainer import MLModelTrainer
from collections import defaultdict
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
        }
    
    def run_model_for_modality(self, modality, random_seeds_list, ROI_method='JHU'):
        processor = MLDataProcessor(modality, ROI_method)
        processor.ROI_df()
        mae_scores, r2_scores = {}, {}
        for random_seed in random_seeds_list:
            folds_data = processor.stratified_sampling(random_seed)
            mae_scores[random_seed], r2_scores[random_seed] = defaultdict(list), defaultdict(list)
            actual_age_list, predicted_age_list = [], []
            count = 0
            for fold in folds_data:
                count += 1
                x_train, y_train = fold['x_train'], fold['y_train']
                x_test, y_test = fold['x_test'], fold['y_test']
                y_group_train = fold['y_group_train']
                unique, counts = np.unique(y_group_train, return_counts=True)
                distribution = dict(zip(unique, counts))
                # print(f"Fold {fold}: Class distribution: {distribution}")
                x_train, x_test = processor.z_score_normalisation(x_train, x_test)
                # print(x_train)
                # print(x_test)
                # print(y_train)
                # print(y_test)
                
                x_train, x_test, weighted_scores = processor.Factor_Analysis(x_train, x_test)
                print(weighted_scores)
                print(modality)
                scores_dict, prediction_dict = self.run_across_models(x_train, y_train, x_test, y_test, y_group_train, modality, random_seed, count)
            
                save_dir = '/imaging/correia/wl05/final_scripts/ROI/feature_ranking/'
                
                weighted_scores.to_csv(f"{save_dir}{modality}_weighted_scores_{random_seed}_fold{str(count)}.csv")
            
                for key in scores_dict:
                    mae_scores[random_seed][key].append(scores_dict[key][0])
                    r2_scores[random_seed][key].append(scores_dict[key][1])
                
                actual_age_list += prediction_dict['y_test'].tolist()
                predicted_age_list += prediction_dict['predicted_brain_age'].tolist()
                    
            self.prediction_plot_across_folds(actual_age_list, predicted_age_list, random_seed, modality)
                    
        return mae_scores, r2_scores
        
    def run_across_models(self, x_train, y_train, x_test, y_test, y_group_train, modality, random_seed, count):
        scores_dict = {}
        for name, (model, param_grid) in self.models.items():
            trainer = MLModelTrainer(name, model, x_train, x_test, y_train, y_test, y_group_train, param_grid)
            trainer.train()

            predicted_brain_age = trainer.predict()
            # print(predicted_brain_age)
            # print(y_test)
            mae, r2 = trainer.evaluate()
            scores_dict[name]=[mae, r2]
            prediction_dict = {'y_test':y_test, 'predicted_brain_age': predicted_brain_age}
            
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, predicted_brain_age, alpha=0.5)
            plt.title('Predicted Brain Age vs Actual Age for '+ modality + str(random_seed))
            plt.xlabel('Actual Age')
            plt.ylabel('Predicted Brain Age')
            plt.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--') 
            plt.grid(True)
            plt.show()
            directory = '/imaging/correia/wl05/final_scripts/ROI/prediction_plots'
            filename = name+modality+str(random_seed) + str(count)
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
        plt.grid(True)
        plt.show()
        directory = '/imaging/correia/wl05/final_scripts/ROI/prediction_plots'
        filename = modality+str(random_seed)
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath)