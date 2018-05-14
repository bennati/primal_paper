import itertools
import numpy as np
from operator import add

#from app.tools import *

def print_debug(debug,str):
    if debug:
        print(str)


class Network():
    """
    This class simulates the network on which transmissions are sent.

    Sensors and agents transmit their messages through the network. The network is implemented as a single object that is held by all sensors and agents.
    The task of the network is to compute the costs of each transmission: the total cost (messages sent) and the privacy loss (locations sent).
    The network contains a topology matrix, that identifies the distance between each sensor and agent.
    The distance is binary: either 0 (pair is physically connected) or 1 (pair is connected through the network).
    """
    def __init__(self,agent_ids,sensor_ids,cost_unit,privacy_unit):
        """
        Args:
        agent_ids: A list of IDs of the agents connected to the network.
        sensor_ids: A list of IDs of the sensor connected to the network.
        cost_unit: How much does a single transmission cost, in terms of technological cost.
        privacy_unit: How much does a single transmission cost, in terms of privacy cost
        """
        ## The counter with the total transmission cost
        self.cost=0
        ## The increment for each transmission
        self.cost_unit=cost_unit
        ## The counter with the total transmission cost
        self.privacy_cost=0
        ## The increment in privacy for each transmission
        self.privacy_unit=privacy_unit
        ## The average cost feedback the agents receive for every timestep. A dictionary with timesteps as keys and values a pair with cost and counter. Warning! Some timesteps might not get recorded (if there is no transmission)
        self.avg_cost_agents={}
        ## The average cost feedback the agents receive for every time they transmit to remote
        self.avg_cost_remote={}
        ## The average success rate of transmitting between agents
        self.success_rate_agents={}
        ## The average success rate of transmitting to remote
        self.success_rate_remote={}
        ## The number of communications between agents at every timestep
        self.comm_counter_agents={}
        ## The number of communications to remote at every timestep
        self.comm_counter_remote={}
        ## The list with the IDs of all agents in the system
        self.agent_ids=agent_ids
        ## The list with the IDs of all agents in the system
        self.sensor_ids=sensor_ids
        ## The topology: a table containing whether sensors and agents are local or remote, trasmission costs depend on this distance.
        self.topology=np.zeros((len(sensor_ids),len(agent_ids)))

    # setters and getters

    def reset_cost(self):
        """
        Resets the cost counter to 0
        """
        self.cost=0

    def reset_privacy(self):
        """
        Resets the privacy counter to 0
        """
        self.privacy_cost=0

    def reset_topology(self,value):
        """
        Sets all connections in the topology to the same value

        Args:
        value: the new distance
        """
        for s in self.sensor_ids:
            for a in self.agent_ids:
                self.set_topology(s,a,value)

    def set_topology(self,src,dest,value):
        """
        Sets the distance between a source and a destination to a given value

        Args:
        src: a sensor ID
        dest: an agent ID
        value: a number

        Raises:
        TypeError: if value is none
        ValueError: if value is not a number
        """
        self.topology[self.sensor_ids.index(src),self.agent_ids.index(dest)]=float(value)

    def get_cost(self):
        """
        Returns the cost counter
        """
        return self.cost

    def get_privacy_cost(self):
        """
        Returns the privacy counter
        """
        return self.privacy_cost

    def increment_avg_cost(self,time,increment):
        """
        Updates a cost by an increment

        Args:
        time: the current timestep
        increment: how much to increase the cost
        """
        if time in self.avg_cost_agents.keys():
            self.avg_cost_agents[time]=list(map(add,self.avg_cost_agents[time],[float(increment),1]))
        else:
            self.avg_cost_agents[time]=[float(increment),1]

    def increment_avg_cost_remote(self,time,increment):
        """
        Updates a cost by an increment

        Args:
        time: the current timestep
        increment: how much to increase the cost
        """
        if time in self.avg_cost_remote.keys():
            self.avg_cost_remote[time]=list(map(add,self.avg_cost_remote[time],[float(increment),1]))
        else:
            self.avg_cost_remote[time]=[float(increment),1]

    def update_success_rate(self,time,success):
        """
        Updates the success rate

        Args:
        time: the current timestep
        success: a boolean
        """
        if time in self.success_rate_agents.keys():
            self.success_rate_agents[time]=list(map(add,self.success_rate_agents[time],[int(success),1]))
        else:
            self.success_rate_agents[time]=[int(success),1]

    def update_success_rate_remote(self,time,success):
        """
        Updates the success rate

        Args:
        time: the current timestep
        success: a boolean
        """
        if time in self.success_rate_remote.keys():
            self.success_rate_remote[time]=list(map(add,self.success_rate_remote[time],[int(success),1]))
        else:
            self.success_rate_remote[time]=[int(success),1]

    def update_comm_counter(self,time):
        """
        Updates the communication counter

        Args:
        time: the current timestep
        """
        if time in self.comm_counter_agents.keys():
            self.comm_counter_agents[time]+=1
        else:
            self.comm_counter_agents[time]=1

    def update_comm_counter_remote(self,time):
        """
        Updates the communication counter

        Args:
        time: the current timestep
        """
        if time in self.comm_counter_remote.keys():
            self.comm_counter_remote[time]+=1
        else:
            self.comm_counter_remote[time]=1

    def __get_avg(self,prop):
        """
        Returns the average of a property
        """
        return [[t,v/float(n)] for (t,(v,n)) in prop.items()] # average

    def get_avg_cost(self):
        """
        Returns the average cost agents had to pay at every timestep

        Returns:
        a list of tuples, indicating a timestep and the average cost across agents
        """
        return self.__get_avg(self.avg_cost_agents)

    def get_avg_cost_remote(self):
        """
        Returns the average cost agents had to pay at every timestep

        Returns:
        a list of tuples, indicating a timestep and the average cost across agents
        """
        return self.__get_avg(self.avg_cost_remote)

    def get_success_rate(self):
        """
        Returns the average success rate when communicating between agents

        Returns:
        a list of tuples, indicating a timestep and the average cost across agents
        """
        return self.__get_avg(self.success_rate_agents)

    def get_success_rate_remote(self):
        """
        Returns the average success rate when communicating to the remote

        Returns:
        a list of tuples, indicating a timestep and the average cost across agents
        """
        return self.__get_avg(self.success_rate_remote)

    def get_comm_counter(self):
        """
        Returns the average success rate when communicating between agents

        Returns:
        a list of tuples, indicating a timestep and the average cost across agents
        """
        return list(self.comm_counter_agents.items())

    def get_comm_counter_remote(self):
        """
        Returns the average success rate when communicating to the remote

        Returns:
        a list of tuples, indicating a timestep and the average cost across agents
        """
        return list(self.comm_counter_remote.items())

    ## Methods

    def parse_costs(self,src,dest,message):
        """
        Determines the privacy costs.

        Args:
        src: the source's ID
        dest: the destination's ID
        message: the labels

        Returns:
        a cost
        """
        f=lambda costs,message: sum([present if label in message else absent for (label,(present,absent)) in costs.items()]) # sums the values in costs, based on the labels in message
        costs={}
        if dest=="remote":                            # agent to remote
            costs={'sensor_id':(self.privacy_unit,    # present
                                2*self.privacy_unit), # absent
                   'time':(0,                     # present
                           2),                     # absent
                   's_type':(0,                   # present
                             2),                  # absent
            }
        elif dest in self.agent_ids:
            if src in self.agent_ids:                  # agent to agent
                costs={'sensor_id':(self.privacy_unit, # present
                                    0),                # absent
                       'value':(0,2),
                       's_type':(0,2)
                }
            elif src in self.sensor_ids:               # sensor to agent
                costs={'sensor_id':(self.privacy_unit, # present
                                    0)                # absent
                }
            elif src == "remote":
                costs={}
            else:
                print("error parsing src")
        else:
            print("error parsing dest")
        return f(costs,message)

    def compute_costs(self,msg,src,dest,distance=1,cost_unit=1,privacy_unit=1):
        """
        Computes the transmission costs for a given message. As a side effect, it updates the values of self.cost and self.privacy_cost

        Args:
        msg: the message, a list of labels
        src: The id of the source
        dest: The id of the destination

        Kwargs:
        distance: the distance between src and dest, defaults to one
        cost_unit: the cost of transmitting a unit of information
        privacy_unit: the privacy cost of transmitting a unit of information

        Returns:
        The total cost, sum of privacy and transmission cost
        """
        cost=self.cost_unit*len(msg)*distance
        self.cost+=cost
        if 'sensor_id' in msg:
            self.privacy_cost+=self.privacy_unit*distance
        return cost+self.parse_costs(src,dest,msg)*distance


    def send_msg(self,src_id=None,dest=None,time=None,msg=None,debug=False):
        """
        Send a new message over the network.

        Update the total cost based on the distance between src and dest.

        Kwargs:
        src_id: The ID of the source.
        dest: The destination (object).
        time: The current timestep, needed for generating statistics
        msg: A dictionary with the contents of the message.
        debug

        Raises:
        TypeError: if src or destination are None, or if msg is not a dictionary
        ValueError: if src and dest are the same object

        Returns:
        a tuple (ans,cost,success) where ans is pair containing a ternary classification (true/false/None) and its confidence, cost is the transmission and privacy cost, and success is the success ratio of transmissions
        """
        ans=None
        cost=0
        reward=0
        success=False
        conf=None
        ## check input
        if src_id==None or dest==None:
            raise TypeError("Source or destination are None")
        if not isinstance(msg,dict):
            raise TypeError("Message is not a dictionary")
        dest_id=dest.get_id()
        choice=[l for (l,v) in msg.items() if v!=None]
        if src_id==dest_id:
            raise ValueError("Error: trying to communicate with one self")
        ## communicate
        if dest_id=="remote":      # sending to the central supervisor, max costs apply
            cost=self.compute_costs(choice,src_id,dest_id)
            self.increment_avg_cost_remote(time,cost) # record statistic
            self.update_comm_counter_remote(time)
            if 'sensor_id' in choice and 'time' in choice and 's_type' in choice:
                ans=dest.send_alarm(src_id,msg['s_type'],msg['time'],msg['sensor_id'],debug=debug) # communicate to remote and get ground truth
                conf=1          # certain
                success=True
            else:
                print_debug(debug,"agent "+str(src_id)+" impossible to transmit to remote: not enough info in "+str(choice)+", need sensor_id, time and s_type")
            self.update_success_rate_remote(time,success)
        elif src_id in self.agent_ids and dest_id in self.agent_ids: # two agents communicate, it goes over the net
            cost=self.compute_costs(choice,src_id,dest_id)
            self.increment_avg_cost(time,cost) # record statistic
            self.update_comm_counter(time)
            if 'value' in choice and 's_type' in choice:
                ans,conf=dest.is_outlier(msg['value'],msg['time'],msg['sensor_id'],msg['s_type'],msg['e_type'],forwarded=src_id,debug=debug)
                print_debug(debug,"Agent "+str(dest_id)+" replying to agent "+str(src_id)+" with "+str(ans))
                success=True
            else:
                print_debug(debug, "agent "+str(src_id)+" impossible to transmit to agent: not enough info in "+str(choice)+", need value and s_type")
            # no privacy loss as agents don't communicate their location
            # self.privacy_cost+=self.privacy_unit
            self.update_success_rate(time,success)
        elif src_id in self.sensor_ids and dest_id in self.agent_ids: # agent and sensor communicate, looking at topology
            distance=self.topology[self.sensor_ids.index(src_id),self.agent_ids.index(dest_id)]
            print_debug(debug,"Distance between sensor "+str(src_id)+" and agent "+str(dest_id)+" is "+str(distance))
            cost=self.compute_costs(choice,src_id,dest_id,distance=distance)
            # do not record statistic as this is a constant, not relevant to study learning performance
            # sensors always send all information
            ans,conf=dest.is_outlier(msg['value'],msg['time'],msg['sensor_id'],msg['s_type'],msg['e_type'],debug=debug)
            success=True
        else:
            print("Warning, trying to send a message between src "+str(src_id)+" and dest "+str(dest_id))
        print_debug(debug,"Transmission between "+str(src_id)+" and "+str(dest_id)+" costed "+str(cost)+" and was"+(" " if success else " not ")+"successful")
        # TODO move to transmitter
        reward=(10 if success else 0)-cost # learning algorithm wants positive rewards, so we invert the costs
        return (ans,conf),cost,reward
