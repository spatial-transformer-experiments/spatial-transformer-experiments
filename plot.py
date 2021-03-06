import json
import pathlib
import matplotlib.pyplot as plt
import math

def main():
    experiment_base_dir = pathlib.Path("experiments")
    experiment_list_of_dicts = [] 
    for experiment_dir in experiment_base_dir.iterdir():
        if experiment_dir.is_dir() and experiment_dir.name != "test":
            with open(str(experiment_dir.joinpath("results.json"))) as json_file:
                result_dict = json.load(json_file)
                experiment_list_of_dicts.append(result_dict)
    
    experiment_list_of_dicts.sort(key=lambda d: d['experiment_name'])
    names = [experiment_dict["experiment_name"][0:6] for experiment_dict in experiment_list_of_dicts]
    accuracies = [experiment_dict["test_accuracy"] for experiment_dict in experiment_list_of_dicts]

    #add the claimed accuracy from https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
    names = ["baseline"] + names
    accuracies = [98.69] + accuracies

    fig, ax = plt.subplots()
    plt.title('Accuracy Comparison')
    ax.yaxis.labelpad = 40
    plt.xlabel("Test Accuracy [%]")
    plt.xlim( (math.floor(min(accuracies)), 100))
    plt.barh(names[::-1],accuracies[::-1])
    plt.tight_layout()
    plt.savefig("AccuracyComparison.png")
    

if __name__ == "__main__":
    main()        