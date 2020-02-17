base_augment.ipynb
dataloader_augmented.py
 - This is the Model and training code for the baseline model used with augmented images.
 - The dataloader was modified to augment the files online.  It must be specified in the file. 
 - It is self contained as long as the csv files are specified and data is at the paths in the csv files.

Base_Model.ipynb
 - This is the Model and training code for the baseline model.  
 - It is self contained as long as the csv files are specified and data is at the paths in the csv files.

base_TF.ipynb
 - This is the Model and training code for the ResNet50 feature exctractor and corresponding baseline model.  
 - It is self contained as long as the csv files are specified and data is at the paths in the csv files.

fcn8s.ipynb
dataloader_fcn8s.py
dataloader.py
 - This is the Model and training code for the FC8NS model.  These dataloaders were used to get resized 
   and original images.
 - It is self contained as long as the csv files are specified and data is at the paths in the csv files.

Final_base_dice.ipynb
 - This is the Model and training code for the modified base model to address class imbalance.  
 - It is self contained as long as the csv files are specified and data is at the paths in the csv files.

nn_output_parser.ipynb
 - Because of issues with training we output all reporting to files.  This file parses these files to extract the 
   information reported.

Test_Model_Eval.ipynb
utils_test_model_eval.py
 - This file reports information relevant to our report including images and IoUs for the required classes.
 - The specified utils file is modified to report the required classes.

U_fcn.py
U_starter.ipynb
 - This is the Model and training code for the modified base model to address class imbalance.
 - The model and training code are in separate files.  
 - It is self contained as long as the csv files are specified and data is at the paths in the csv files.

utils.py
 - The original utils file filled in with the necessary code.
