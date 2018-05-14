import unittest
from app.network import *
from app.agent import *
from app.remote import *

class TestNetwork(unittest.TestCase):
    """
    Test suite for class Network
    """

    def __init__(self, *args, **kwargs):
        super(TestNetwork, self).__init__(*args, **kwargs)
        self.cu=1
        self.pu=1
        self.n=Network(["a1","a2"],["s1","s2"],self.cu,self.pu)
        self.time=2
        self.time2=3
        self.inc=0.5
        self.inc2=1
        self.dist=0.5


    def test_topology_default_0(self):
        self.assertEqual(sum(self.n.topology.flatten()),0)

    def test_set_topology_value(self):
        self.n.set_topology("s1","a1",1)
        self.assertEqual(sum(self.n.topology.flatten()),1)

    def test_set_topology_notInList_Exception(self):
        self.assertRaises(ValueError,self.n.set_topology,"a1","s1",1) # swapped src and dest
        self.assertRaises(ValueError,self.n.set_topology,"s1","no",1) # value not in list
        self.assertRaises(ValueError,self.n.set_topology,"no","a1",1) # value not in list

    def test_set_topology_badValue_Exception(self):
        self.assertRaises(TypeError,self.n.set_topology,"s1","a1",None) # not a number
        self.assertRaises(ValueError,self.n.set_topology,"s1","a1","nonumber") # not a number

    def test_avgCost_default_empty(self):
        self.assertEqual(self.n.avg_cost_agents,{})

    def test_incrementAvgCost_correctValue(self):
        self.n.increment_avg_cost(self.time,self.inc)
        self.assertNotEqual(self.n.avg_cost_agents,{}) # not empty
        self.assertEqual(self.n.avg_cost_agents[self.time],[self.inc,1])

    def test_incrementAvgCost_updateCorrect(self):
        ## update an existing key
        self.n.increment_avg_cost(self.time,self.inc)
        self.n.increment_avg_cost(self.time,self.inc)
        self.assertEqual(self.n.avg_cost_agents[self.time],[2*self.inc,2]) # value,counter

    def test_incrementAvgCost_keyIsString_ok(self):
        time="nonumber"
        self.n.increment_avg_cost(time,self.inc)
        self.assertEqual(self.n.avg_cost_agents[time],[self.inc,1]) # works fine
        # update key
        self.n.increment_avg_cost(time,self.inc)
        self.assertEqual(self.n.avg_cost_agents[time],[2*self.inc,2]) # works fine

    def test_incrementAvgCost_incrementIsString_exception(self):
        inc="nonumber"
        self.assertRaises(ValueError,self.n.increment_avg_cost,self.time,inc) # update does not work

    def test_getAvgCost_default_empty(self):
        self.assertEqual(self.n.get_avg_cost(),[]) # empty value

    def test_getAvgCost_initialized_correct(self):
        self.n.increment_avg_cost(self.time,self.inc)
        self.assertEqual(self.n.get_avg_cost(),[[self.time,self.inc]])

    def test_getAvgCost_updated_average(self):
        self.n.increment_avg_cost(self.time,self.inc)
        self.n.increment_avg_cost(self.time,self.inc2)
        self.assertEqual(self.n.get_avg_cost(),[[self.time,(self.inc+self.inc2)/float(2)]]) # average

    def test_getAvgCost_updatedMultiKey_average(self):
        self.n.increment_avg_cost(self.time,self.inc)
        self.n.increment_avg_cost(self.time,self.inc2)
        self.n.increment_avg_cost(self.time2,self.inc2)
        self.assertEqual(self.n.get_avg_cost(),[[self.time,(self.inc+self.inc2)/float(2)],
                                           [self.time2,self.inc2]]) # average

    def test_sendMsg_wrongInput_exception(self):
        a=Agent([])
        r=Remote()
        self.assertRaises(TypeError,self.n.send_msg)
        self.assertRaises(TypeError,self.n.send_msg,"nosrc",a,"t1","msg")

    def test_sendMsg_sameSrcDest_exception(self):
        a=Agent([])
        r=Remote()
        self.assertRaises(ValueError,self.n.send_msg,a.get_id(),a,"t1",{})

    def __send_msg_helper(self,src,dest,msg,c,ncost,nprivcost,cu,pu,init_net=False):
        n=Network([src,dest.get_id()],["s1","s2"],cu,pu)
        if init_net:
            dest.set_network(n)
        ans,cost,success=n.send_msg(src,dest,"t1",msg)
        self.assertEqual(cost,c)
        self.assertEqual(n.cost,ncost)
        self.assertEqual(n.privacy_cost,nprivcost)

    def test_sendMsg_remote_correct(self):
        ## TODO compute correct values with one function
        r=Remote()
        # all info absent
        self.__send_msg_helper("src",r,{"sensor_id":None,"time":None,"e_type":None},2*self.pu+2+2,0,0,self.cu,self.pu)
        self.__send_msg_helper("src",r,{"sensor_id":1,"time":None,"e_type":None},self.cu+self.pu+2+2,self.cu,self.pu,self.cu,self.pu)
        self.__send_msg_helper("src",r,{"sensor_id":None,"time":1,"e_type":None},self.cu+2*self.pu+2,self.cu,0,self.cu,self.pu)
        self.__send_msg_helper("src",r,{"sensor_id":None,"time":None,"e_type":1},self.cu+2*self.pu+2,self.cu,0,self.cu,self.pu)
        self.__send_msg_helper("src",r,{"sensor_id":1,"time":1,"e_type":None},2*self.cu+self.pu+2,2*self.cu,self.pu,self.cu,self.pu)
        self.__send_msg_helper("src",r,{"sensor_id":1,"time":1,"e_type":1},3*self.cu+self.pu,3*self.cu,self.pu,self.cu,self.pu)

    def test_sendMsg_remoteAllInfo_correct(self):
        # all three are present
        a=Agent([])
        r=Remote()
        time="t1"
        ans,cost,success=self.n.send_msg(a,r,time,{"sensor_id":1,"time":time,"e_type":1})
        ## check that alarm is sent
        self.assertEqual(r.get_alarms(),[[time,a,1,1]])
        ## check stats
        self.assertEqual(self.n.avg_cost_remote[time],[cost,1])

    def test_sendMsg_agent2agent_correct(self):
        src="a1"
        dest=Agent([])
        # all info absent
        self.__send_msg_helper(src,dest,{"sensor_id":None,"value":None,"s_type":None,"time":None,"e_type":None},2+2,0,0,self.cu,self.pu,init_net=True)
        # sensor_id is present
        self.__send_msg_helper(src,dest,{"sensor_id":1,"value":None,"s_type":None,"time":None,"e_type":None},self.cu+self.pu+2+2,self.cu,self.pu,self.cu,self.pu,init_net=True)
        # time is present
        self.__send_msg_helper(src,dest,{"sensor_id":None,"value":1,"s_type":None,"time":None,"e_type":None},self.cu+2,self.cu,0,self.cu,self.pu,init_net=True)
        # e_type is present
        self.__send_msg_helper(src,dest,{"sensor_id":None,"value":None,"s_type":1,"time":None,"e_type":None},self.cu+2,self.cu,0,self.cu,self.pu,init_net=True)
        # all present
        self.__send_msg_helper(src,dest,{"sensor_id":1,"value":1,"s_type":1,"time":None,"e_type":None},3*self.cu+self.pu,3*self.cu,self.pu,self.cu,self.pu,init_net=True)

    def test_sendMsg_sensor2agent_correct(self):
        src="s1"
        dest=Agent([])
        msg={"value":1,"time":2,"sensor_id":None,"s_type":None,"e_type":None}
        n=Network(["a1",dest.get_id()],["s1","s2"],self.cu,self.pu)
        dest.set_network(n)
        n.set_topology(src,dest.get_id(),self.dist)
        ans,cost,success=n.send_msg(src,dest,"t1",msg)
        self.assertEqual(cost,self.cu*2*self.dist)
        self.assertFalse(ans[0])
        self.assertEqual(n.cost,self.cu*self.dist*2)
        self.assertEqual(n.privacy_cost,0)
        ## with sensor id
        msg={"value":1,"time":2,"sensor_id":1,"s_type":None,"e_type":None}
        n=Network(["a1",dest.get_id()],["s1","s2"],self.cu,self.pu)
        dest.set_network(n)
        n.set_topology(src,dest.get_id(),self.dist)
        ans,cost,success=n.send_msg(src,dest,"t1",msg)
        self.assertEqual(cost,self.cu*3*self.dist+self.pu*self.dist)
        self.assertFalse(ans[0])
        self.assertEqual(n.cost,self.cu*self.dist*3)
        self.assertEqual(n.privacy_cost,self.pu*self.dist)

    def test_computeCosts_privCost(self):
        ## internal vars are assigned properly
        msg=["value","time","sensor_id","s_type","e_type"]
        self.n.compute_costs(msg,"s1","a1",distance=self.dist)
        self.assertEqual(self.n.cost,len(msg)*self.cu*self.dist)
        self.assertEqual(self.n.privacy_cost,self.pu*self.dist)

    def test_computeCosts_noPrivCost(self):
        ## no privacy cost
        msg=["value","time","s_type","e_type"]
        self.n.compute_costs(msg,"s1","a1",distance=self.dist)
        self.assertEqual(self.n.cost,len(msg)*self.cu*self.dist)
        self.assertEqual(self.n.privacy_cost,0)
