import uuid
from sklearn import svm
import numpy as np
from app.tools import *
from app.sensor import *

class Classifier():
    """
    Contains classification algorithm. Each instance is classifying a specific type of event.
    """
    def __init__(self,norm_func=None):
        """
        Set the training vector with 0 and initialize the classifier
        """
        ## The unique identifier
        self.id=uuid.uuid4()
        ## The size of the sliding window for learning (keep only last points)
        self.window_size=20
        ## Internal classification algorithm
        self.classifier=None
        self.fitted=False
        ## Initialize memory with zeros
        self.train=[]
        self.init_training(norm_func)
        self.__reset_classifier()

    def init_training(self,norm_func=None,n=5):
        if norm_func!=None:     # initialize the training
            self.train=[norm_func(np.random.normal(loc=Sensor.mu,scale=Sensor.sd)) for _ in range(n)]
        else:
            self.train=[]

    def set_window_size(self,value):
        """
        Set the size of the window (lenght of training vector)

        Args:
        value: the lenght
        """
        self.window_size=value

    def __reset_classifier(self,debug=False):
        """
        Create a new classifier and train it with the contents of self.train
        """
        self.classifier=svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        if len(self.train)>0:
            if self.fitted==False:
                print_debug(debug,str(self.id)+" initialized")
            self.fitted=True
            self.classifier.fit(np.asarray(self.train).reshape(-1,1))

    def classify(self,value):
        """
        Classify the value.

        Args:
        value: the measurement to classify

        Returns:
        a tuple containing the classification and the confidence (distance to hyperplane).
        If the classifier is not fitted (no points in the training vector), the classification will be None
        """
        val=np.asarray(value).reshape(1, -1)
        if self.fitted:
            label=self.classifier.predict(val)[0]==-1 # outliers get label -1
            conf=self.classifier.decision_function(val)[0][0] # outliers get label -1
        else:                                                 # returns None, will be removed when aggregating answers
            label=None
            conf=None
        return (label,conf)

    def learn(self,value,classification,debug=False):
        """
        Update classifier with new datapoint.
        If the point is classified as outlier, it is removed from the training vector if present.
        If the point is classified as normas, it is added to the training vector if not present.
        If a previous classification of the same value was correct, the training vector is not modified.

        Args:
        value: The data point.
        classification: The true label, True means outlier.

        Kwargs:
        debug: enables debug output
        """
        if classification:      # an outlier
            print_debug(debug,"Removing outlier from training")
            # remove that point from the training vector
            for j in [i for i,x in enumerate(self.train) if x == value]:
                del self.train[j]
            if len(self.train)==0:
                self.fitted=False
                self.init_training()
        else:      # the point is normal
            self.update_training(value,debug)
        self.__reset_classifier(debug)

    def update_training(self,value,debug=False):
        """
        Update classifier with new datapoint

        Args:
        value: The data point.

        Kwargs:
        debug: enables debug output
        """
        if value not in self.train:
            self.train.insert(0,value) # record the new value
            if len(self.train)>self.window_size:
                self.train.pop()    # remove the last value
        self.__reset_classifier(debug)
