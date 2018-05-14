"""
Comment:
If we allow agents to learn from what other agents transmit, performance of calibrated learning changes.
Calibrated learning sends only the required information, which does not include the ID of the agent in agent-to-agent communication.
Not transmitting that value has an impact on performance because the agent cannot distinguish between neighbors.
A classifier is associated to each of the neighbors. If their id is the same (None) then all data transmitted will be used to train the same classifier.
For this reason, a calibrated agent has only 2 classifiers: one trained on its sensor, the other on all neighbors' data.
If we transmit the ID the number of classifiers will become number of neighbors + 1.
Interestingly if one classifier is trained with more data, the performance is much lower (similar to if that classifier would not be present at all)
"""

import uuid
import numpy as np
from app.tools import *
from app.classifier import *
from app.transmitter import *
from app.sensor import *

class Agent():
    """
    An agent classifies measuremnts from a set of sensors.

    Agent receives and processes measurements from a number of sensors.
    Each agent is connected to a set of neighbors.
    Each agent can classify a different types of events, classifiers are contained in a dictionary {key:eventtype val:classifier} in attribute self.dic.
    When a sensor communicates its type to the agent, a new classifier is added to the dictionary.
    """

    def __init__(self, neighbors,classifier=None,remote=None,network=None,debug=False,learn=False,calibrated=False,classify=False,train=True,pretrain=False,learn_from_neighbors=False,single=False):
        """
        Args:
        neighbors: The neighbors of this agent, it must be a list.

        Kwargs:
        classifier: A dictionary containing a classifier for each event type.
        remote: The ID of the remote unit to which alarms are sent.
        network: The network object through which messages are sent.
        learn: If enabled, reinforcement learning is used to reduce the information transmitted
        calibrated: If true, the protocol gets calibrated from the start. Works only if learn is also enabled.
        classify: If disabled, the internal classifier is not used and all measurements are classified as outliers. Side effect: everything is transmitted.
        train: if enabled, the classifiers are trained. Defaults to True
        pretrain: if enabled, the classifiers are pretrained
        learn_from_neighbors: If enabled, measurements communicated by the neighbors are used to train the classifier.
        single: If enabled, the agent will keep only one classifier. Otherwise one classifier for each sensor or agent that transmits to it.
        debug: enable debug output
        """
        ## The unique identifier of this agent
        self.id=uuid.uuid4()
        print_debug(debug,"Creating agent "+str(self.id))
        ## The list of neighbors
        self.ns = neighbors
        ## The network object
        self.net=network
        ## The ID of the central unit
        self.remote=remote
        ## The classifier to use
        if classifier:
            self.cls=classifier
        else:
            self.cls=Classifier
        ## The dictionary containing the registered sources, on which to train the classifiers
        self.src_ids={}
        ## The dictionary containing the classifiers
        self.dic={}
        # associate labels to values
        ## Transmitter object associated with remote
        self.t_remote=Transmitter(Sensor.protocol)
        ## Transmitter object associated with other agents
        self.t_agent=Transmitter(Sensor.protocol)
        ## TODO place in network with other requirements of protocol
        if learn and calibrated:          # calibrate the transmitters
            self.t_remote=Transmitter(Sensor.protocol,eps=0.0) # no random choice
            self.t_remote.Q.q[(1,str((False,True,True,False,True)))]=5 # time,s_type and sensor_id
            self.t_agent=Transmitter(Sensor.protocol,eps=0.0) # no random choice
            self.t_agent.Q.q[(1,str((True,False,True,False,False)))]=5 # value and s_type
        # intelligence
        ## determines if the protocol is learned
        self.learn_protocol=learn
        ## determine if the transmitter is calibrated
        self.transmitter_calibrated=calibrated
        ## determines if the classification is learned
        self.enable_classification=classify
        ## the threshold for considering a classification reliable (margin support vectors)
        self.thresh=0.01
        ## function that determines when to communicate with agents
        self.comm=lambda obj,outlier,confidence : (
            confidence==None or (abs(confidence)<=obj.thresh) # low confidence
        )
        ## function that determines when to communicate with remote
        self.comm_rem=lambda obj,outlier,confidence : outlier
        ## determines if the classifiers are trained
        self.train_classifier=train
        ## determines if the classifiers are pretrained
        self.pretrain_classifier=pretrain
        ## Determines if measurements communicated by negihbors are memorized or not
        self.learn_from_neighbors=learn_from_neighbors
        ## Determines if the agent will keep only one classifier (for each event type) or one for each sensor or agent
        self.single_classifier=single
        ## Determines the function used to parse the answers of neighbors
        self.parse_fct=lambda ans,norm: np.sum([a*c/norm for a,c in ans]) # weighted normalized average
        #self.parse_fct=lambda ans,norm: np.sum([a for a,c in ans]) # standard average

    # Setters and getters

    def get_id(self):
        """
        Returns the agent's ID

        Returns:
        the agent's ID
        """
        return self.id

    def set_comm(self,fun):
        """
        Sets the function that determines when to communicate to agents
        """
        self.comm=fun

    def set_comm_rem(self,fun):
        """
        Sets the function that determines when to communicate to remote
        """
        self.comm_rem=fun

    def set_neighbors(self,neighs):
        """
        Sets the neighbors

        Args:
        neighs: the list of neighbors
        """
        ns=[n for n in neighs if n.get_id()!=self.id]
        self.ns=ns

    def set_network(self,net):
        """
        Sets the network object

        Args:
        net: the Network object
        """
        self.net=net

    def set_remote(self,rem):
        """
        Sets the remote object

        Args:
        rem: the Remote object
        """
        self.remote=rem

    def set_learn_protocol(self,boolean):
        """
        Sets the value of learn_protocol

        Args:
        boolean: the value
        """
        self.learn_protocol=boolean

    def set_classify(self,boolean):
        """
        Sets the value of enable_classification

        Args:
        boolean: the value
        """
        self.enable_classification=boolean

    def set_train_classifier(self,boolean):
        """
        Sets the value of train_classifier

        Args:
        boolean: the value
        """
        self.train_classifier=boolean

    # Methods

    def tokey(self,src,flag=True):
        """
        Converts a source ID to a key to be used in the dictionaries self.dic and self.src_ids

        If flag is False, ignore the value of parameter single_classifier.
        """
        return ("default" if flag and self.single_classifier else str(src))

    def register_source(self,s_type,src_id,debug=False):
        """
        Updates the dictionary src_ids
        """
        if s_type not in self.src_ids:
            print_debug(debug,"Agent "+str(self.id)+" registers source "+self.tokey(src_id,False)+" for NEW type "+str(s_type))
            self.src_ids[s_type]=[self.tokey(src_id,False)]
        elif self.tokey(src_id,False) not in self.src_ids[s_type]:
            print_debug(debug,"Agent "+str(self.id)+" registers source "+self.tokey(src_id,False)+" for type "+str(s_type))
            self.src_ids[s_type].append(self.tokey(src_id,False)) # extend the dictionary

    def source_registered(self,s_type,src_id):
        """
        Checks if the source is registered.

        If the source is registered, the agent can learn its measurements.
        """
        if s_type in self.src_ids:
            return self.tokey(src_id,False) in self.src_ids[s_type]
        else:
            raise Exception("Unrecognized sensor type")

    def __is_margin_support(classif,confid):
        return classif or abs(confid)<=self.thresh

    def learn(self,value,classif,s_type,src_id,debug=False):
        """
        Trains the classifiers on the current measurement.

        The classifier is trained on all points that are classified as normal.
        The classifier is trained only on the measurements coming from registered sources.
        Usually only sensors are registered sources, but if the parameter learn_from_neighbors is enabled, agents are allowed to learn also from their neighbors.

        If the parameter single_classifier is enabled, the agent has only one default classifier that is trained on all points from all registered sourced.
        The agents must differenciate from registered and not registered sources even if this parameter is enabled, but it might not differenciate during training.

        Args:
        value: the measurement
        classif: its classification (True=outlier). Can have value None if classifier is unsure (e.g. not fitted), in that case the point is recorded.
        s_type: the type of sensor
        src_id: the ID of the source reporting the measurement.

        Kwargs:
        debug: enable debug output
        """
        if self.train_classifier:
            if self.learn_from_neighbors:
                self.register_source(s_type,src_id,debug=debug) # add the source (agent) to the registered sources
                self.init_classifier(s_type,src_id,debug=debug) # creates a new classifier for the source (agent)
            try:
                if self.source_registered(s_type,src_id): # it can learn only from registered sources
                    self.dic[s_type][self.tokey(src_id)].learn(value,classif,debug=debug) # trains the right classifier (could be default)
            except Exception as e:
                print(e)
                print(s_type)
                print(src_id)
                print(self.dic)
                print(self.src_ids)
        else:
            print_debug(debug,"Not training classifier")


    def init_classifier(self,s_type,src_id,debug=False,norm_func=None):
        """
        Communicate to the agent the availability of a new sensor type, create a new classifier if necessary.
        An agent is able to classify a type only if it has at least one sensor of that type. Sensors communicate their type to the agents during initialization.

        Args:
        s_type: The sensor type (dictionary key).
        src_id: The sensor ID (inner dictionary key).

        Kwargs:
        norm_func: the function used to initialize the classifier
        debug: enable debug output

        """
        print_debug(debug,"Agent "+str(self.id)+" received an init request for type "+str(s_type))
        fct=norm_func if self.pretrain_classifier else None
        if s_type not in self.dic:
            print_debug(debug,"Agent "+str(self.id)+" adds type "+str(s_type)+" to the dictionary")
            self.dic[s_type]={self.tokey(src_id): self.cls(norm_func=fct)} # create a new dictionary
        elif self.tokey(src_id) not in self.dic[s_type]:
            self.dic[s_type].update({self.tokey(src_id): self.cls(norm_func=fct)}) # extend the dictionary


    def classify(self,s_type,value):
        """
        Classify as outlier a measurement of a certain type.

        Args:
        s_type: the classifier to use
        value: the measurement to classify

        Raises:
        KeyError: if key is unknown

        Returns:
        a pair, classification and confidence interval. If enable_classification is disabled it classifies all values as outliers (with confidence None)
        """
        if self.enable_classification:
            return self.parse_answers([d.classify(value) for d in self.dic[s_type].values()])       # returns the classification and the confidence interval
        else:
            return (True,None)

    def is_outlier(self,value,t,loc,s_type,e_type,forwarded=None,debug=False):
        """
        Classify a measurement.

        Receive a reading from a sensor and classify it either or not as an outlier.

        If the sensor type is registered, use the internal classifier, otherwise rely on the neighbors.
        The value is communicated to the neighbors if
        - There are neighbors, and
        - This measurement does not come from another agent (avoid transmission loops), and
        - Classification is disabled, or the confidence is below a threshold.
        The answers of the neighbors are aggregated with the function self.parse_answer()

        Once the classification is established (and is not None), the value is added to the training set by self.learn(),
        and an alarm is sent to remote if the measurement is classified as an outlier.

        Args:
        value: The value of the measurement to classify.
        t: The current time.
        loc: The ID of the sensor transmitting this measurement.
        s_type: The type of sensor sending this measurement, used to select the appropriate classifier.
        e_type: The type of event that is measured.

        Kwargs:
        forwarded: Whether this measurement comes from another agent, in that case do not ask for neighbors' opinions.
        debug: enable debug output

        Returns:
        A ternary classification (True or False or None) and its confidence
        """
        if not self.net:
            print("Warning: network uninitialized for agent "+str(self.id))
        outlier=None
        confidence=None
        # if forwarded==None:       # ask opinion to neighbors
        #     for neigh in self.ns:
        #             print_debug(debug,"Asking neighbor "+str(neigh.get_id()))
        #         if self.net:
        #             self.net.send_msg(self.id,neigh.get_id())
        #         else:
        #             print("Warning: network uninitialized for agent "+str(self.id))
        #         answers.append(neigh.is_outlier(value,t,None,s_type,e_type,forwarded=self.id))
        if s_type in self.dic:
            (outlier,confidence)=self.classify(s_type,value) # classifies taking into consideration the opininon of all classifiers trained on this sensor type.
            print_debug(debug,"Agent "+str(self.id)+" received value "+str(value)+" of type "+str(s_type)+(" from agent "+str(forwarded) if forwarded else " from sensor "+str(loc))+" and classified it as "+("outlier" if outlier==True else "not outlier")+", with confidence "+str(confidence))
            # if self.ns:                                     # there are neighbors
            #     outlier=self.parse_answers(outlier,answers) # update decision
        else:
            print_debug(debug,"Agent "+str(self.id)+" received value "+str(value)+" of type "+str(s_type)+(" from agent "+str(forwarded) if forwarded else " from sensor "+str(loc))+" but is unable to classify it"+(", relying on neighbors" if self.ns else ""))
        # ask opinion to neighbors and discard own uncertain classification
        if (self.ns              # there are neighbors
            and not forwarded
            and ((not self.enable_classification) # transmit all events if classification is disabled
                 or self.comm(self,outlier,confidence))):
            choice=self.__define_protocol(zip(Sensor.protocol,[value,t,s_type,e_type,loc]),destination='agent') # find the labels to send
            answers=[]
            for neigh in self.ns:
                print_debug(debug,"--- Asking neighbor "+str(neigh.get_id()))
                ans,trans_cost,reward=self.net.send_msg(src_id=self.id,dest=neigh,time=t,msg=choice,debug=debug) # send to neighbor
                self.__learn_protocol(reward) # learn protocol
                answers.append(ans)
            outlier,confidence=self.parse_answers(answers) # use only the neighbors' decisions
        # learn the data point
        self.learn(value,outlier,s_type,(forwarded or loc),debug=debug)
        ## send alarm to remote if outlier
        if (not forwarded
            and self.comm_rem(self,outlier,confidence)):
            choice=self.__define_protocol(zip(Sensor.protocol,[value,t,s_type,e_type,loc]),destination='remote') # find the labels to send
            print_debug(debug,"--- Sending to remote, all costs apply")
            gt,trans_cost,reward=self.net.send_msg(src_id=self.id,dest=self.remote,time=t,msg=choice,debug=debug) # log communication in network
            self.__learn_protocol(reward,destination='remote') # learn protocol
            ## debug
            if gt[0]==True:
                string="an event"
            elif gt[0]==False:
                string="a false alarm"
            else:
                string="no feedback"
            print_debug(debug,"Remote responds with: "+string)
            if gt[0] != None and gt[0] != outlier: # update classification.
                self.process_feedback(t,value,gt[0],s_type,e_type,loc,debug=debug,requested=True) # Classifier checks for and removes duplicates.
        return outlier,confidence

    def parse_answers(self,answers):
        """
        Take a decision based on the answers of the neighbors.

        Args:
        answers: A list of pairs containing the classifications,confidence from the neighbors.

        Returns:
        A pair containing a boolean classification and a confidence.
        """
        ans=[((1 if a else -1),abs(c)) for (a,c) in answers if (a != None and c!=None)] # remove uncertain answers and compute abs of confidences
        cl=[c for a,c in ans]                              # confidences
        if ans:
            classif=self.parse_fct(ans,np.sum(cl))
            return classif>0,np.mean(cl) # classification closer to 0 or 1?, average confidence
        else:                                      # neighbors are undecided
            if all([a for a,c in answers]):        # classification is disabled
                return True,None
            else:
                return None,None

    def __define_protocol(self,protocol,destination="agent",debug=False):
        """
        Returns the values that have to be transmitted

        Args:
        protocol: a list of pairs containing a label and a value, one for each piece of information that can be transmitted

        Kwargs:
        destination: either agent or remote, used to select the appropriate transmitter
        debug: enable debug output

        Returns:
        a dictionary of labels, those contained in protocol, with their corresponding value or with value None, if they do not have to be transmitted
        """
        if self.learn_protocol:
            if destination=="remote":
                labels=self.t_remote.choose_action()
                print_debug(debug,"Transmitting "+str(labels)+" to remote")
            elif destination=="agent":
                labels=self.t_agent.choose_action()
                print_debug(debug,"Transmitting "+str(labels)+" to agents")
            choice=dict([(l,v) if l in labels else (l,None) for (l,v) in protocol]) # get the values for these labels
            return choice
        else:
            return dict(protocol)     # return all values

    def __learn_protocol(self,reward,destination="agent"):
        if self.learn_protocol and not self.transmitter_calibrated:
            if destination=="remote":
                protocol=self.t_remote.learn(reward) # the classifier wants positive values (rewards) so we invert the costs
            elif destination=="agent":
                protocol=self.t_agent.learn(reward)

    def process_feedback(self,t,value,classif,s_type,e_type,src,debug=False,requested=True):
        """
        Receives feedback from remote and updates its training accordingly.
        It can also propagate the feedback to the neighbors, although they might receive the feedback even if they have not saved the measurement in the first place

        Kwargs:
        requested: if the feedback comes from a request of the agent, it that case it comes as a binary answer. If not requested, the message must also include content and so the cost increases.
        """
        print_debug(debug,"Updating classification "+str(classif)+" at time "+str(t)+" with value "+str(value)+" from agent "+str(self.get_id()))
        if requested:
            msg={"feedback":classif}
        else:
            msg={"feedback":classif,"value":value,"sensor_id":src,"s_type":s_type,"time":t}
        self.net.compute_costs(msg,"remote",self.get_id()) # record cost of transmitting feedback
        self.learn(value,classif,s_type,src,debug=debug)
        # Propagate feedback to neighbors
        for nbr in self.ns:
            if nbr.learn_from_neighbors: # if the agent can learn from its neighbors, it could have saved the measurement. Update its training
                choice=self.__define_protocol(zip(Sensor.protocol,[value,t,s_type,e_type,src]),destination='agent') # find the labels to send
                choice.update({"feedback":classif})
                self.net.compute_costs(choice,self.get_id(),nbr.get_id()) # record cost of transmitting feedback
                nbr.learn(value,classif,s_type,self.get_id(),debug=debug)
