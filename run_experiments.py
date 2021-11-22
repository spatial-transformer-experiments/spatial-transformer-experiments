from train import train
from models import Net, Net_CoordConv

#rerun basline after reorganisation
def run_experiments():

    # Run baseline after reorganising the code    
    train(model=Net(),experiment_name="exp_01_run_01_reproduce_baseline")
    
    # Runs the Net_CoordConv with the configuration that corresponds to Net with increased n epochs
    train(model=Net_CoordConv(),experiment_name="exp_02_run_01_reproduce_baseline",n_epochs=30)

    # Determine  how big the influence of the localisation network is
    # In order to do that the localistation network is bypassed
    train(model=Net_CoordConv(bypass_localisation=True),experiment_name="exp_03_run_01_bypass_localisation",n_epochs=30)
    
    # Same bypass idea but this time the convolution layers of the classifier are replaced by coord conf layers  
    train(model=Net_CoordConv(use_coordconf_classifier=True,bypass_localisation=True),experiment_name="exp_04_run_01_coordconf_classifier_bypass_localisation",n_epochs=30)
    
    # Use coordconf for the classifier
    train(model=Net_CoordConv(use_coordconf_classifier=True),experiment_name="exp_05_run_01_coordconf_classifier",n_epochs=30)
    
    # Use coordconf for the localisation
    train(model=Net_CoordConv(use_coordconf_localisation=True),experiment_name="exp_06_run_01_coordconf_localistation",n_epochs=30)

    # Use coordconf for both the classifier and the localisation
    train(model=Net_CoordConv(use_coordconf_localisation=True,use_coordconf_classifier=True),experiment_name="exp_07_run_01_coordconf_localistation_coordconf_classifier",n_epochs=30)
    

    #train(model=Net_CoordConv_Homography(),experiment_name="exp_07_run_01_homography",epochs=40)
    # train(model=Net_CoordConv_Homography(use_coordconf_classifier=True),experiment_name="exp_08_run_01_homography_coordconf_classifier")    
    # train(model=Net_CoordConv_Homography(use_coordconf_localisation=True),experiment_name="exp_09_run_01_homography_coordconf_localistation")
    # train(model=Net_CoordConv_Homography(use_coordconf_localisation=True,use_coordconf_classifier=True),experiment_name="exp_10_run_01_homography_coordconf_localistation_coordconf_classifier")
    


if __name__=="__main__":
    run_experiments()
