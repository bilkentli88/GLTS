import pandas as pd
import glob
import re

from dataset_names import list_run_again


result_files_folder = ""

df_learning_shapelets = pd.read_csv('../../result_csv_files/learning_shapelets_results.csv')
df_learning_shapelets = df_learning_shapelets.set_index("dataset_name")
#print(df_learning_shapelets.dtypes)


liste_result_files = glob.glob("../../result_csv_files/result_files_shapelet/best_results_according_to_*.csv")
print(liste_result_files)

print("learning_shapelets",df_learning_shapelets.shape)
#print(df_learning_shapelets.head())

df_joined = df_learning_shapelets

for filename in liste_result_files:


	method_name = re.sub(r".*best_results_according_to_", "", filename)
	method_name = re.sub(r"_search_max.*", "", method_name)

	#print(method_name)
	df = pd.read_csv(f"{filename}")

	df = df[["dataset_name","accuracy_score"]]
	df.rename(columns={"accuracy_score": f"accuracy_score_{method_name}"},inplace=True)
	duplicates = df[df["dataset_name"].duplicated()]
	#print(duplicates)

	df = df.set_index("dataset_name")
	#print(df.dtypes)

	df_joined = df_joined.join(df,on="dataset_name",how="outer")


	#print(method_name,df_joined.shape)
	#print(df_joined.head())




df_joined["max_val_method_name"] = df_joined.idxmax(axis=1)

df_joined.to_csv("../../result_csv_files/combined_comparison.csv")
df_joined.to_csv("../../result_csv_files/result_files_shapelet/combined_comparison.csv")



print(df_joined['max_val_method_name'].value_counts())

#print(df_joined.columns)

df_run_again = df_joined[df_joined.index.isin(list_run_again)].copy()

df_run_again.to_csv("../../result_csv_files/run_again.csv")


