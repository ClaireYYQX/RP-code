import pandas as pd
from voxel_tbss_run import MLModelRunner
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os

class Runner:
    def __init__(self) -> None:
        self.modalities = ['DTI', 'FWE']
        self.model_performance = []
        self.random_seed_list = [i for i in range(41,51)]
    
    def run(self):
        model_runner = MLModelRunner()
        model_runner.parameter_init()
        mae_scores_dict = defaultdict(dict)
        r2_scores_dict = defaultdict(dict)
        mae_mean = defaultdict(dict)
        r2_mean = defaultdict(dict)
        for modality in self.modalities:
            mae_scores, r2_scores = model_runner.run_model_for_modality(modality, self.random_seed_list)
            for random_seed in self.random_seed_list:
                mae_scores_dict[random_seed][modality] = mae_scores[random_seed]['SVR']
                r2_scores_dict[random_seed][modality] = r2_scores[random_seed]['SVR']
        
        for random_seed in self.random_seed_list:
            mae_scores_df = pd.DataFrame(mae_scores_dict[random_seed])
            r2_scores_df = pd.DataFrame(r2_scores_dict[random_seed])
            self.performance_boxplot(mae_scores_df, 'mae', random_seed)
            self.performance_boxplot(r2_scores_df, 'r2', random_seed)
            mae_mean['DTI'][random_seed] = mae_scores_df.loc[:, 'DTI'].mean()
            mae_mean['FWE'][random_seed] = mae_scores_df.loc[:, 'FWE'].mean()
            r2_mean['DTI'][random_seed] = r2_scores_df.loc[:, 'DTI'].mean()
            r2_mean['FWE'] [random_seed]= r2_scores_df.loc[:, 'FWE'].mean()
            
        mean_mae_df = pd.DataFrame.from_dict(mae_mean)
        mean_r2_df = pd.DataFrame.from_dict(r2_mean)
        mean_mae_df.to_csv('voxel_mae_mean_tbss-new.csv', index=True)
        mean_r2_df.to_csv('voxel_r2_mean_tbss-new.csv', index=True)
        self.average_boxplot(mean_mae_df, 'mae')
        self.average_boxplot(mean_r2_df, 'r2')
    
    def performance_boxplot(self, scores_df, metric, random_seed):
        # Boxplot for MSE and R2
        print(scores_df)
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=scores_df)
        plt.title('Comparison for ' + metric+str(random_seed))
        plt.ylabel(metric)
        plt.show()     
        directory = '/imaging/correia/wl05/final_scripts/Voxel/voxel_performance_plots/'  
        filename = metric+str(random_seed)+'_tbss-new'
        plt.savefig(f'{directory}{filename}')     
    
    def performance_output(self):
        df = pd.DataFrame(columns=['modality', 'MLmodel', 'MAE', 'R_squared'])
        row_number = 0
        for modality, MLmodel, Mean_Absolute_Error, R_squared in self.model_performance:
            df.loc[row_number] = [modality, MLmodel, Mean_Absolute_Error, R_squared]
            row_number += 1
        print(df)
        print(df.mean(axis=0))
        
    def average_boxplot(self, mean_df, metric):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=mean_df, showfliers=True)
        plt.title('Comparison for ' + metric)
        plt.ylabel(metric)
        plt.show()     
        directory = '/imaging/correia/wl05/final_scripts/Voxel/voxel_performance_plots/'  
        filename = metric+'_'+str(self.random_seed_list[0])+'-'+str(self.random_seed_list[-1])+'_tbss-new'
        plt.savefig(f'{directory}{filename}')  

if __name__ == '__main__':
    program_runner = Runner()
    program_runner.run()