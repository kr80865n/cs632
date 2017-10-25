# cs632
Deep Learning
Short installation guide for Mac/Linux, using virtualenv.

$ cd ~
$ virtualenv --system-site-packages ~/my_env
$ source ~/my_env/bin/activate
$ git clone https://github.com/random-forests/cs632.git
$ cd cs632/setup
$ pip install -r requirements.txt
$ curl -O http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
$ tar xvf cifar-10-python.tar.gz
$ python train.py
$ python predict.py
Model file =my_model.h5
predictive text file-> predictor.txt
Process to Execute
step 1: Execute train.py
by executing train.py we create the model and model with weights is saved with file name my_model.h5
step 2: Execute predict.py 
which loads the model and executes the model on test batch and predict.txt file is created with the prediction of cat and dog.
