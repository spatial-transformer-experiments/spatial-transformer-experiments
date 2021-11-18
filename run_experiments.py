from train import train
from models import Net, Net_CoordConv

#rerun basline after reorganisation
def run_experiments():
    #experiment 1 run baseline after reorganising the code
    
    # train(model=Net(),experiment_name="exp_01_run_01_baseline")
    # train(model=Net(),experiment_name="exp_01_run_02_baseline")
    # train(model=Net(),experiment_name="exp_01_run_03_baseline")

    #
    train(model=Net_CoordConv(),experiment_name="exp_02_run_01_baseline")
    train(model=Net_CoordConv(bypass_localisation=True),experiment_name="exp_03_run_01_bypass_localisation")
    train(model=Net_CoordConv(use_coordconf_classifier=True),experiment_name="exp_04_run_01_coordconf_classifier")
    train(model=Net_CoordConv(use_coordconf_classifier=True,bypass_localisation=True),experiment_name="exp_05_run_01_coordconf_classifier_bypass_localisation")
    train(model=Net_CoordConv(use_coordconf_localisation=True),experiment_name="exp_06_run_01_coordconf_localistation")
    train(model=Net_CoordConv(use_coordconf_localisation=True,use_coordconf_classifier=True),experiment_name="exp_07_run_01_coordconf_localistation_coordconf_classifier")
    


if __name__=="__main__":
    run_experiments()
