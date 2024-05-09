import sys
import os

import dataset_names
import MainModule

if __name__ == "__main__":
	liste = dataset_names.get_database_list_from_arguments(sys.argv)

	print("datasets to work on")
	print(liste)
	search_type = "HyperoptSearchTPE"
	for dataset_name in liste:
		MainModule.find_result_for_one(search_type,dataset_name,n_iter = 100,search_max_epoch=1000)
	

	
