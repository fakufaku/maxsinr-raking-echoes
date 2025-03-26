recipes_dir := "recipes/exp2_run_for_good_models"
results_dir := "results_tmp/talsp_good_baselines" 

# sync results from sap.sacs19 to local
sync_model_results_sumo_to_local:
    rsync -avz sap.sacs19:/home/diego/Documents/Code/NeuralSteerer/{{results_dir}}/models/  {{results_dir}}/models

# sync results from local to sap.sacs19 
sync_model_results_local_to_sum:
    rsync -avz {{results_dir}}/models sap.sacs19:/home/diego/Documents/Code/NeuralSteerer/{{results_dir}}/models/