import sys
import os

from optuna import samplers

import dataset_names
import MainModule

liste_samplers = ["Default", "NSGAIISampler", "CmaEsSampler", "MOTPESampler"]

if __name__ == "__main__":
	liste = dataset_names.get_database_list_from_arguments(sys.argv)

	print("datasets to work on")
	print(liste)
	for dataset_name in liste:
		for sampler in liste_samplers:
			search_type = f"OptunaSearch{sampler}"
			MainModule.find_result_for_one(search_type,dataset_name,100)
	

