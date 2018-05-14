import uuid
import numpy as np
from app.tools import *

class Sensor():
    """
    A sensor measures events and sends them for classification to a list of agents, to which it is connected to.
    """

    ## The protocol, what can be transmitted
    protocol=["value","time","s_type","e_type","sensor_id"]
    mu=0
    sd=0.4

    def __init__(self, norm_distr,event_distr,agents,s_type,e_type):
        """
        Args:
        norm_distr: The value distribution of the normal sensor measurements. It must be a function centered in 0 with support [-1,1[, each measurement is a random point on this function.
        event_distr: The value distribution of the events this sensor is measuring. It must be a function centered in 0 with support [-1,1[, each measurement is a random point on this function.
        agents: A sensor is connected to a set of agents, it must be a list.
        s_type: The sensor type.
        e_type: The event type measured by this sensor
        """
        ## The unique ID
        self.id=uuid.uuid4()
        ## The normal measurement function
        self.n = norm_distr
        ## The event function
        self.e = event_distr
        ## The sensor type
        self.s_type = s_type
        ## The event type
        self.e_type= e_type
        ## The network object through which to send the measurements
        self.net=None
        ## The list of agents to which to send the measurements
        self.a=None
        self.set_agents(agents) # initialize agents

    # Setters and getters

    def get_id(self):
        """
        Returns the sensor's ID
        """
        return self.id

    def get_agents(self):
        """
        Returns the agents associated with this sensor
        """
        return self.a

    def get_type(self):
        """
        Returns the sensor's type
        """
        return self.s_type

    def get_etype(self):
        """
        Returns the event's type
        """
        return self.e_type

    def set_distr(self,event_distr):
        """
        Sets the function for events

        Args:
        event_distr: the function
        """
        self.e= event_distr

    def set_normal_distr(self,norm_distr):
        """
        Sets the function for normal measurements

        Args:
        norm_distr: the function
        """
        self.n= norm_distr

    def set_agents(self,agents):
        """
        Sets the agents associated with the sensor and initializes their classifiers

        Args:
        agents: the list of agents
        """
        self.a = agents
        for agent in self.a:
            agent.register_source(self.s_type,self.id) # sensors are always registered sources
            agent.init_classifier(self.s_type,self.id,norm_func=self.n) # create a classifier

    def set_network(self,net):
        """
        Sets the sensor's network

        Args:
        net: the Network object
        """
        self.net=net

    # Methods

    def measure(self,time,debug=False,event=False):
        """
        Measure one value and communicate it to the agents.

        Args:
        time: the current timestep

        Kwargs:
        event: if true an event is generated
        debug: enables the debug output

        Returns:
        the measurement
        """
        if event:
            measurement=self.e(np.random.normal(loc=self.mu,scale=self.sd)) # support from -1 to 1 (1% of samples fall outside)
        else:
            measurement=self.n(np.random.normal(loc=self.mu,scale=self.sd))
        for agent in self.a:
            if self.net:
                self.net.send_msg(src_id=self.id,dest=agent,debug=debug,time=time,msg=dict(zip(self.protocol,[measurement,time,self.s_type,self.e_type,self.id])))
            else:
                print("Warning: network uninitialized for sensor "+str(self.id))
        return measurement
