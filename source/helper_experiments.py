import glob
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
from sklearn.model_selection import RandomizedSearchCV
from hyperopt import hp,fmin,tpe,STATUS_OK,Trials,anneal,partial,mix,rand
from ShapeletGeneration import ShapeletGeneration3LN as ShapeletNN
from helper_skorch import ShapeletRegularizedNet
from helper_datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from set_seed import set_seed


set_seed(8)
# int or float (percentage)
BAG_SIZE_START = 4
BAG_SIZE_END = 0.4
BAG_SIZE_COUNT = 40

LIST_STRIDE_RATIOS = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
PROTOTYPE_COUNT = 100
LIST_N_PROTOTYPES = np.linspace(10, 300, PROTOTYPE_COUNT, endpoint=True, dtype=int).tolist()
# LIST_LEARNING_RATES =  [0.01, 0.05,0.1,0.2]
LIST_LEARNING_RATES = [0.0001]
def get_list_bag_sizes(time_series_length):
    bag_size_start = BAG_SIZE_START
    bag_size_end = BAG_SIZE_END
    if BAG_SIZE_START < 0:
        bag_size_start = int(time_series_length * BAG_SIZE_START)

    if BAG_SIZE_END < 0:
        bag_size_end = int(time_series_length * BAG_SIZE_END)

    list_bag_sizes = np.linspace(bag_size_start, bag_size_end, BAG_SIZE_COUNT, endpoint=True, dtype=int)
    list_bag_sizes = np.unique(list_bag_sizes).tolist()
    return list_bag_sizes

def hyperopt_objective(search_space):
	dataset_name = search_space["dataset_name"]
	train, y_train, test, y_test = load_dataset(dataset_name)
	y_train = torch.from_numpy(y_train).float()
	y_train_labels = np.argmax(y_train,axis=1)
	n_classes = y_train.shape[1]

	nn_shapelet_generator = ShapeletNN(
         n_prototypes = search_space["n_prototypes"]
         , bag_size = search_space["bag_size"]
         , n_classes = n_classes
         , stride_ratio = search_space["stride_ratio"]
         , features_to_use_str=search_space["features_to_use_str"])

	cv_count = search_space["cv_count"]

	max_epoch = search_space["max_epoch"]
	net = get_skorch_regularized_classifier(nn_shapelet_generator,max_epoch)

	y_train_labels = np.argmax(y_train,axis=1)
	cv_count = search_space["cv_count"]

	net.fit(train,y_train_labels)
	y_predicted = net.predict(test)
	y_test_labels = np.argmax(y_test, axis=1)
	accuracy = accuracy_score(y_predicted, y_test_labels)
	# We aim to maximize accuracy, therefore we return it as a negative value
	return {'loss': -accuracy, 'status': STATUS_OK }

def find_best_hyper_params_hyperopt_search_annealing(dataset_name
	,n_iter = 100
	,search_max_epoch = 100
	,cv_count = 3
	,features_to_use_str="min,max,mean"
	):
	return find_best_hyper_params_hyperopt_search(dataset_name
		,search_algorithm=anneal.suggest
		, n_iter = n_iter
		, search_max_epoch = search_max_epoch
		, cv_count = cv_count
		,features_to_use_str = features_to_use_str
		)

def find_best_hyper_params_hyperopt_search_tpe(dataset_name
	,n_iter = 100
	,search_max_epoch = 100
	,cv_count = 3
	,features_to_use_str="min,max,mean"
	):
	return find_best_hyper_params_hyperopt_search(dataset_name
		,search_algorithm=tpe.suggest
		, n_iter = n_iter
		, search_max_epoch = search_max_epoch
		, cv_count = cv_count
		,features_to_use_str = features_to_use_str
		)


def find_best_hyper_params_hyperopt_search_partial(dataset_name
                                                   , n_iter=100
                                                   , search_max_epoch=100
                                                   , cv_count=3
                                                   , features_to_use_str="min,max,mean"
                                                   ):
    search_algorithm = partial(mix.suggest,
                               p_suggest=[
                                   (.1, rand.suggest),
                                   (.2, anneal.suggest),
                                   (.7, tpe.suggest)])

    return find_best_hyper_params_hyperopt_search(dataset_name
                                                  , search_algorithm=search_algorithm
                                                  , n_iter=n_iter
                                                  , search_max_epoch=search_max_epoch
                                                  , cv_count=cv_count
                                                  , features_to_use_str=features_to_use_str
                                                  )


def find_best_hyper_params_hyperopt_search(dataset_name
                                           , search_algorithm
                                           , n_iter=100
                                           , search_max_epoch=100
                                           , cv_count=3
                                           , features_to_use_str="min,max,mean"

                                           ):
    train, y_train, test, y_test = load_dataset(dataset_name)
    y_train = torch.from_numpy(y_train).float()
    y_train_labels = np.argmax(y_train, axis=1)
    n_classes = y_train.shape[1]

    list_bag_sizes = get_list_bag_sizes(train.shape[1])

    # print("list_bag_sizes",list_bag_sizes)

    search_space = {'stride_ratio': hp.choice('stride_ratio', LIST_STRIDE_RATIOS),
                    'n_prototypes': hp.choice('n_prototypes', LIST_N_PROTOTYPES),
                    'cv_count': hp.choice('cv_count', [cv_count]),
                    "dataset_name": hp.choice("dataset_name", [dataset_name]),
                    "max_epoch": hp.choice("max_epoch", [search_max_epoch]),
                    'features_to_use_str': hp.choice('features_to_use_str', [features_to_use_str]),
                    'bag_size': hp.choice('bag_size', list_bag_sizes),
                    "lr": hp.choice("lr", LIST_LEARNING_RATES)
                    }

    trials = Trials()

    best = fmin(fn=hyperopt_objective,
                space=search_space,
                algo=search_algorithm,
                max_evals=n_iter,
                trials=trials,
                return_argmin=False)
    d = {}
    d["dataset_name"] = dataset_name
    d["search_type"] = "HyperoptSearch"
    d["features_to_use"] = features_to_use_str
    # d["best_hyper_params"] = str(opt.best_params_)
    d["bag_size"] = best["bag_size"]
    d["features_to_use"] = best["features_to_use_str"]
    d["n_classes"] = n_classes
    d["n_prototypes"] = best["n_prototypes"]
    d["stride_ratio"] = best["stride_ratio"]
    d["best_score"] = -1 * min(trials.losses())
    d["n_iter"] = n_iter
    d["cv_count"] = cv_count

    return d

def find_best_hyper_params_randomized_search(dataset_name
                                             , n_iter=100
                                             , search_max_epoch=100
                                             , cv_count=3
                                             , features_to_use_str="min,max,mean"

                                             ):
    train, y_train, test, y_test = load_dataset(dataset_name)
    y_train = torch.from_numpy(y_train).float()
    y_train_labels = np.argmax(y_train, axis=1)
    n_classes = y_train.shape[1]

    net = get_skorch_regularized_classifier(ShapeletNN, search_max_epoch)

    list_bag_sizes = get_list_bag_sizes(train.shape[1])

    params = {
        'lr': LIST_LEARNING_RATES,
        'module__stride_ratio': LIST_STRIDE_RATIOS,
        'module__bag_size': list_bag_sizes,
        'module__n_prototypes': LIST_N_PROTOTYPES,
        'module__n_classes': [n_classes],
        'module__features_to_use_str': [features_to_use_str]
    }

    opt = RandomizedSearchCV(net, params
                             , refit=True
                             , n_iter=n_iter
                             , cv=cv_count
                             , scoring='accuracy')

    opt.fit(train, y_train_labels)

    d = {}
    d["dataset_name"] = dataset_name
    d["search_type"] = "BayesSearchCV"
    d["features_to_use"] = features_to_use_str
    # d["best_hyper_params"] = str(opt.best_params_)
    d["bag_size"] = opt.best_params_["module__bag_size"]
    d["features_to_use"] = opt.best_params_["module__features_to_use_str"]
    d["n_classes"] = opt.best_params_["module__n_classes"]
    d["n_prototypes"] = opt.best_params_["module__n_prototypes"]
    d["stride_ratio"] = opt.best_params_["module__stride_ratio"]
    d["best_score"] = opt.best_score_
    d["n_iter"] = n_iter
    d["cv_count"] = cv_count

    return d

def find_best_hyper_params_bayesian_search(dataset_name
                                           , n_iter=100
                                           , search_max_epoch=100
                                           , cv_count=3
                                           , features_to_use_str="min,max,mean"

                                           ):
    train, y_train, test, y_test = load_dataset(dataset_name)
    y_train = torch.from_numpy(y_train).float()
    y_train_labels = np.argmax(y_train, axis=1)
    n_classes = y_train.shape[1]

    net = get_skorch_regularized_classifier(ShapeletNN, search_max_epoch)

    list_bag_sizes = get_list_bag_sizes(train.shape[1])

    params = {
        'lr': LIST_LEARNING_RATES,
        'module__stride_ratio': LIST_STRIDE_RATIOS,
        'module__bag_size': list_bag_sizes,
        'module__n_prototypes': LIST_N_PROTOTYPES,
        'module__n_classes': [n_classes],
        'module__features_to_use_str': [features_to_use_str]
    }
    print("dataset_name", dataset_name, "params", params)

    opt = BayesSearchCV(
        net,
        params,
        n_iter=n_iter,
        cv=cv_count,
        scoring='accuracy'
    )

    opt.fit(train, y_train_labels)

    d = {}
    d["dataset_name"] = dataset_name
    d["search_type"] = "BayesSearchCV"
    d["features_to_use"] = features_to_use_str
    # d["best_hyper_params"] = str(opt.best_params_)
    d["bag_size"] = opt.best_params_["module__bag_size"]
    d["features_to_use"] = opt.best_params_["module__features_to_use_str"]
    d["n_classes"] = opt.best_params_["module__n_classes"]
    d["n_prototypes"] = opt.best_params_["module__n_prototypes"]
    d["stride_ratio"] = opt.best_params_["module__stride_ratio"]
    d["best_score"] = opt.best_score_
    d["n_iter"] = n_iter
    d["cv_count"] = cv_count

    return d

def find_test_results(dataset_name
                      , bag_size
                      , n_prototypes
                      , stride_ratio
                      , max_epoch
                      , features_to_use_str):
    train, y_train, test, y_test = load_dataset(dataset_name)
    # Standardize the data
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    y_train = torch.from_numpy(y_train).float()
    y_train_labels = np.argmax(y_train, axis=1)
    n_classes = y_train.shape[1]

    nn_shapelet_generator = ShapeletNN(
        n_prototypes=n_prototypes
        , bag_size=bag_size
        , n_classes=n_classes
        , stride_ratio=stride_ratio
        , features_to_use_str=features_to_use_str)

    net = get_skorch_regularized_classifier(nn_shapelet_generator, max_epoch)

    y_train_labels = np.argmax(y_train, axis=1)

    y_test_labels = np.argmax(y_test, axis=1)



    net.fit(train, y_train_labels)
    y_predict = net.predict(test)
    score = accuracy_score(y_test_labels, y_predict)

    return score

def get_skorch_regularized_classifier(nn_shapelet_generator, max_epochs):
    net = ShapeletRegularizedNet(
        nn_shapelet_generator,
        max_epochs=max_epochs,
        lr=0.001,
        train_split=None,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
    )
    return net
def get_best_result_for_one_dataset(search_type, dataset_name, n_iter, search_max_epoch, best_result_max_epoch):
    search_filename = get_filename_output_for_search(search_type, dataset_name, n_iter, search_max_epoch)
    if not os.path.exists(search_filename):
        print(f"{search_filename} does not exists, SKIPPING")
        return

    output_best_results_filename = get_filename_output_for_best_results(search_type, dataset_name, n_iter,
                                                                        search_max_epoch, best_result_max_epoch)
    print("output_best_results_filename", output_best_results_filename)
    if os.path.exists(output_best_results_filename):
        df_results = pd.read_csv(output_best_results_filename)
        # list_of_results_dict = df_results.to_dict('records')
        row_count = df_results[df_results["dataset_name"] == dataset_name].shape[0]
        if row_count > 0:
            print(f"{dataset_name} exists in df_results, skipping")
            return

    df = pd.read_csv(search_filename)
    print(f"RUNNING for {dataset_name}")
    bag_size = int(df["bag_size"][0])
    n_prototypes = df["n_prototypes"][0]
    stride_ratio = df["stride_ratio"][0]
    features_to_use_str = df["features_to_use"][0]
    accuracy_score = find_test_results(
        dataset_name=dataset_name
        , bag_size=bag_size
        , n_prototypes=n_prototypes
        , stride_ratio=stride_ratio
        , max_epoch=best_result_max_epoch
        , features_to_use_str=features_to_use_str
    )
    print(dataset_name, accuracy_score)
    d = {}
    d["dataset_name"] = dataset_name
    d["bag_size"] = bag_size
    d["n_prototypes"] = n_prototypes
    d["stride_ratio"] = stride_ratio
    d["max_epoch"] = best_result_max_epoch
    d["features_to_use_str"] = features_to_use_str
    d["features_to_use_str"] = features_to_use_str
    d["accuracy_score"] = accuracy_score
    # d["search_type"] = search_type

    if os.path.exists(output_best_results_filename):
        df_results = pd.read_csv(output_best_results_filename)
        df_results = df_results.append(d, ignore_index=True)
    else:
        list_of_results_dict = []
        list_of_results_dict.append(d)
        df_results = pd.DataFrame(list_of_results_dict)

    print(d)
    df_results.to_csv(output_best_results_filename, index=False)


def get_filename_output_for_search(search_type, dataset_name, n_iter, max_epoch):
    output_filename = f"output_results/{search_type}_params_{dataset_name}_n_iter_{n_iter}_max_epoch_{max_epoch}.csv"
    return output_filename


def get_filename_output_for_best_results(search_type, dataset_name, n_iter, search_max_epoch, best_result_max_epoch):
    output_filename = f"output_results/best_results_according_to_{search_type}_search_max_epoch_{search_max_epoch}_search_n_iter_{n_iter}_result_max_epoch_{best_result_max_epoch}.csv"
    return output_filename


dispatcher = {'BayesSearch': find_best_hyper_params_bayesian_search,
			  "RandomizedSearch": find_best_hyper_params_randomized_search,
              "HyperoptSearchAnnealing":find_best_hyper_params_hyperopt_search_annealing,
			  "HyperoptSearchTPE":find_best_hyper_params_hyperopt_search_tpe,
			  "HyperoptSearchPartial":find_best_hyper_params_hyperopt_search_partial
			}


def find_result_for_one(search_type, dataset_name, n_iter=100, search_max_epoch=100):
    output_filename = get_filename_output_for_search(search_type, dataset_name, n_iter, search_max_epoch)

    if os.path.isfile(output_filename):
        print(f"dataset:{dataset_name} exists in {output_filename}")
    else:
        print(f"RUNNING for dataset:{dataset_name}")

        result_dict = dispatcher[search_type](
            dataset_name
            , n_iter=n_iter
            , search_max_epoch=search_max_epoch
        )
        result_dict["search_type"] = search_type
        print(result_dict)

        df = pd.DataFrame(result_dict, index=[0])
        df.to_csv(output_filename, index=False)
        print(f"Saved dataset:{dataset_name}, in {output_filename}")

