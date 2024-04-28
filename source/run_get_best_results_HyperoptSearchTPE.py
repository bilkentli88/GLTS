import sys
import helper_experiments
import dataset_names

if __name__ == "__main__":
	#print(sys.argv)
	liste = dataset_names.get_database_list_from_arguments(sys.argv)
	for dataset_name in liste:
		helper_experiments.get_best_result_for_one_dataset(
			search_type="HyperoptSearchTPE"
			,dataset_name= dataset_name
			,n_iter=100
			,search_max_epoch = 1000
			,best_result_max_epoch= 5000
			)
