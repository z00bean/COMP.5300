COMP.5300: Advanced Topics in Deep Learning

In order to run this code you need the libraries in the requirements file, approved access to the CheXpert dataset and GPU system with CUDA 10 or higher.
This code executes in three main stages:
1. Dataset manipulation and data transformation: This stage prepares the data by extracting pairs of frontal and lateral images from the CheXpert dataset.
2. Train all the models one by one.
pip install -r requirements.txt
python COMP.5300/bin/train.py COMP.5300/config/cfg1.json logdir --num_workers 8 
This has to be done for all models which are being trained.
Note: For conditional probability computation using data label ontoloy, this training has to be done for each sub-tree. The trained models will be saved in the log folder.

3. Test the performance of the models.
python classification/bin/test.py
After the results are stored in the CSV file, it is fed to the in/roc.py .
This has to be done for each of the result CSV files.
