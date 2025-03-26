# sync results from sap.sacs19 to local
sync_model_results_sumo_to_local:
    rsync -avz dicarlod@sumo.sap:/home/dicarlod/Documents/Code/maxsinr-raking-echoes/results /Users/chutlhu/Documents/Code/maxsinr-raking-echoes/

# sync results from local to sap.sacs19 
sync_model_results_local_to_sum:
    rsync -avP /Users/chutlhu/Documents/Code/maxsinr-raking-echoes  dicarlod@sumo.sap:/home/dicarlod/Documents/Code/