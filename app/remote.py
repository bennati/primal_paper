import numpy as np
from app.tools import *

class Remote():
    """
    This object represents the central unit. Its task is to provide the ground truth about events and record alarms (to compute system accuracy)
    """
    def __init__(self,supervised=False):
        """
        Initializes the Remote object

        KWargs:
        supervised: if true, the remote will give feedback to every alarm it receives
        """
        ## The unique ID
        self.id="remote"
        ## The list recording all alarms. A quadruplet (time,agentID,s_type,sensorID)
        self.alarms=[]
        ## The ground truth about the events. A triplet (s_type,time,location)
        self.gt=[]
        ## Whether to provide feedback after an alarm is reported
        self.supervised=supervised

    ## Getters and setters

    def get_id(self):
        """
        Return the object's ID (which is always the string 'remote')
        """
        return self.id

    def get_alarms(self):
        """
        Returns the alarms received by the remote
        """
        return self.alarms

    def set_alarms(self,val):
        """
        Resets the alarms to a value

        Args:
        val: the new alarms
        """
        self.alarms=val

    def set_ground_truth(self,gt):
        """
        Sets the ground truth about the events

        Args:
        gt: the new ground truth
        """
        self.gt=gt

    def set_supervised(self,val):
        """
        Setter for self.supervised
        """
        self.supervised=val

    def get_supervised(self):
        """
        Getter for self.supervised
        """
        return self.supervised

    ## Methods

    def ground_truth(self,s_type,t,loc):
        """
        Return the real label of an event.

        Args:
        s_type: The type of event.
        loc: The location where this event happened.
        t: The time at which this event happened.

        Returns:
        A boolean classification.
        """
        return (s_type,t,loc) in self.gt

    def send_alarm(self,agent_id,s_type,t,loc,debug=False):
        """
        Receive an alarm from an agent, record it and give back the ground truth.

        Args:
        agent_id: The ID of the agent.
        s_type: The type of event.
        loc: The location where this event happened.
        t: The time at which this event happened.

        Returns:
        A boolean classification, or None if unsupervised.
        """
        self.alarms.append([t,agent_id,s_type,loc])
        if (s_type,t,loc) in self.gt:
            print_debug(debug,"Remote: True positive reported")
        else:
            print_debug(debug,"Remote: False positive reported")
        ret = None
        if self.supervised:
            ret= self.ground_truth(s_type,t,loc)
        return ret

    def compute_score(self):
        """
        Checks how many alarms have been transmitted and compares them with the ground truth about events.

        Returns: the score (precision,recall,f1)
        """
        alarms=[(e,t,s) for [t,a,e,s] in self.alarms] # extract the data sent to the remote
        try:
            true_positives=sum([(a in self.gt) for a in alarms]) # check how many are true events
            false_positives=(len(alarms)-true_positives)/float(len(self.gt))
            true_positives/=float(len(self.gt))
            false_negatives=sum([(a not in alarms) for a in self.gt])/float(len(self.gt))
            prec,rec=compute_prec_recall(true_positives,false_positives,false_negatives)
        except ZeroDivisionError:
            print("Lenght of ground truth is zero")
            prec=np.nan
            rec=np.nan
        return prec,rec,compute_f1(prec,rec)

    def compute_partial_score(self,time):
        """
        Checks how many alarms have been transmitted and compares them with the ground truth about events.

        Args:
        time: the current timestep

        Returns: the score (precision,recall,f1)
        """
        alarms=[(e,t,s) for [t,a,e,s] in self.alarms] # extract the data sent to the remote
        gt=[(e,t,s) for (e,t,s) in self.gt if t<=time]
        try:
            true_positives=sum([(a in gt) for a in alarms]) # check how many are true events
            false_positives=(len(alarms)-true_positives)/float(len(gt))
            true_positives/=float(len(gt))
            false_negatives=sum([(a not in alarms) for a in gt])/float(len(gt))
            prec,rec=compute_prec_recall(true_positives,false_positives,false_negatives)
        except ZeroDivisionError:
            print("Lenght of ground truth is zero")
            prec=np.nan
            rec=np.nan
        return prec,rec,compute_f1(prec,rec)
