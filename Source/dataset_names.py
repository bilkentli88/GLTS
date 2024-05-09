def get_database_list_from_arguments(sys_argv):
	""" get which datasets will be used from command line.
	if nothing is given all datasets will be used
	string dataset name can be given like Adiac
	index of dataset can be given like 2
	dataset range can be given like 2:5 
	"""

	liste = list_one
	if len(sys_argv) == 2:
		arg = sys_argv[1]

		if ":" in arg:
			arg_arr = arg.split(":")
			if (len(arg_arr)) == 3:
				increment = int(arg_arr[2])
			else:
				increment = 1
			start = int(arg_arr[0])
			end = int(arg_arr[1])
			liste = liste[start:end:increment]
		elif arg.isnumeric():
			index = int(arg)
			liste = liste[index:index+1]
		else:
			# dataset name is given
			index = liste.index(arg) 
			liste = liste[index:index+1]

	if len(sys_argv) == 3:
		start = int(sys_argv[1])
		end = int(sys_argv[2])
		liste = liste[start:end]
	return liste


list_one = ["CBF"]


list_2_class = [

"BeetleFly",
"BirdChicken",
"Coffee",
"Computers",
"DistalPhalanxOutlineCorrect",
"Earthquakes",
"ECG200",
"ECGFiveDays",
"FordA",
"FordB",
"GunPoint",
"Ham",
"HandOutlines",
"Herring",
"ItalyPowerDemand",
"Lightning2",
"MiddlePhalanxOutlineCorrect",
"MoteStrain",
"PhalangesOutlinesCorrect",
"ProximalPhalanxOutlineCorrect",
"ShapeletSim",
"SonyAIBORobotSurface1",
"SonyAIBORobotSurface2",
"Strawberry",
"ToeSegmentation1",
"ToeSegmentation2",
"TwoLeadECG",
"Wafer",
"Wine",
"WormsTwoClass",
"Yoga"
]


list_image_datasets = [
"DiatomSizeReduction",
"DistalPhalanxOutlineAgeGroup",
"DistalPhalanxOutlineCorrect",
"DistalPhalanxTW",
"FaceAll",
"FaceFour",
"FacesUCR",
"FiftyWords",
"Fish",
"HandOutlines",
"Herring",
"MiddlePhalanxOutlineAgeGroup",
"MiddlePhalanxOutlineCorrect",
"MiddlePhalanxTW",
"OSULeaf",
"PhalangesOutlinesCorrect",
"ProximalPhalanxOutlineAgeGroup",
"ProximalPhalanxOutlineCorrect",
"ProximalPhalanxTW",
"ShapesAll",
"SwedishLeaf",
"Symbols",
"WordSynonyms",
"Yoga"
#"Medicals", # does not exists
#"MixedShapes", # does not exists
#"MixedShapesSmallTrain", # does not exists
]


list_learning_shapelet_datasets = [

"Adiac",
"Beef",
"BirdChicken",
"ChlorineConcentration",
"Coffee",
"DiatomSizeReduction",
"ECGFiveDays",
"FaceFour",
"ItalyPowerDemand",
"Lightning7",
"MedicalImages",
"MoteStrain",
"SonyAIBORobotSurface1",
"SonyAIBORobotSurface2",
"Symbols",
"Trace",
"TwoLeadECG"

]

list_all_86 = [

"Adiac",
"ArrowHead",
"Beef",
"BeetleFly",
"BirdChicken",
"Car",
"CBF",
"ChlorineConcentration",
"CinCECGtorso",
"Coffee",
"Computers",
"CricketX",
"CricketY",
"CricketZ",
"DiatomSizeReduction",
"DistalPhalanxOutlineCorrect",
"DistalPhalanxOutlineAgeGroup",
"DistalPhalanxTW",
"Earthquakes",
"ECG200",
"ECG5000",
"ECGFiveDays",
"ElectricDevices",
"FaceAll",
"FaceFour",
"FacesUCR",
"FiftyWords",
"Fish",
"FordA",
"FordB",
"GunPoint",
"Ham",
"HandOutlines",
"Haptics",
"Herring",
"InlineSkate",
"InsectWingbeatSound",
"ItalyPowerDemand",
"LargeKitchenAppliances",
"Lightning2",
"Lightning7",
"Mallat",
"Meat",
"MedicalImages",
"MiddlePhalanxOutlineCorrect",
"MiddlePhalanxOutlineAgeGroup",
"MiddlePhalanxTW",
"MoteStrain",
"NonInvasiveFatalECGThorax1",
"NonInvasiveFatalECGThorax2",
"OliveOil",
"OSULeaf",
"PhalangesOutlinesCorrect",
"Phoneme",
"Plane",
"ProximalPhalanxOutlineCorrect",
"ProximalPhalanxOutlineAgeGroup",
"ProximalPhalanxTW",
"RefrigerationDevices",
"ScreenType",
"ShapeletSim",
"ShapesAll",
"SmallKitchenAppliances",
"SonyAIBORobotSurface1",
"SonyAIBORobotSurface2",
#"StarlightCurves",
"Strawberry",
"SwedishLeaf",
"Symbols",
"SyntheticControl",
"ToeSegmentation1",
"ToeSegmentation2",
"Trace",
"TwoLeadECG",
"TwoPatterns",
"UWaveGestureLibraryX",
"UWaveGestureLibraryY",
"UWaveGestureLibraryZ",
"UWaveGestureLibraryAll",
"Wafer",
"Wine",
"WordSynonyms",
"Worms",
"WormsTwoClass",
"Yoga",
]






list_run_again =[
'ArrowHead',
'Beef',
'CinCECGtorso',
'CricketX',
'CricketY',
'CricketZ',
'FaceFour',
'FacesUCR',
'FiftyWords',
'Fish',
#'FordA',
#'FordB',
#'InsectWingbeatSound',
'Lightning7',
'Mallat',
'MoteStrain',
'Phoneme',
#'UWaveGestureLibraryAll',
'UWaveGestureLibraryX',
'UWaveGestureLibraryZ',
]


list_altay_run_again = [
"DistalPhalanxOutlineCorrect",
"DistalPhalanxOutlineAgeGroup",
"DistalPhalanxTW",

]
