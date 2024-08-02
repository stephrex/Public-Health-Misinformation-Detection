import os

project_dir = os.getcwd()

data_path = project_dir + '/Data/'

output_dir = project_dir + '/Output/'

save_model_dir = project_dir + '/Output/Models/'

save_model_plot = project_dir + '/Output/Plots/'

corona_best_model = save_model_dir + '/Coronavirus_Models/randomforest'

maternity_best_model = save_model_dir + 'Maternity_Models/svm'

lassafever_best_model = save_model_dir + 'Lassafever_Models/svm'