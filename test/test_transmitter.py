import numpy as np
import unittest
from app.transmitter import *
from app.agent import *
from app.remote import *
from app.network import *

default_actions=["value","time","s_type","e_type","sensor_id"]

class TestTransmitter(unittest.TestCase):
    """
    Test suite for class Transmitter
    """

    def test_qlearn_getq(self):
        q=QLearn(default_actions)
        index=np.random.randint(0,len(default_actions))
        q.q[(1,default_actions[index])]=10
        ret=q.getQ(1,default_actions[index])
        self.assertEqual(ret,10) # correct value
        ret=q.getQ(2,default_actions[index])
        self.assertEqual(ret,0.0) # default value

    def test_qlearn_learn(self):
        q=QLearn(default_actions)
        index=np.random.randint(0,len(default_actions))
        ## check if it learns correctly
        q.learn(1,default_actions[index],10,1)
        ret=q.getQ(1,default_actions[index])
        self.assertGreater(ret,0) # learned a value
        self.assertEqual(ret,10) # learned the correct value
        ## check that learned values are reinforced correctly
        q.learn(1,default_actions[index],10,1)
        ret=q.getQ(1,default_actions[index])
        self.assertGreater(ret,10) # reinforced value
        ## check that not learned things are 0
        ret=q.getQ(2,default_actions[index])
        self.assertEqual(ret,0) # nothing learned
        ret=q.getQ(2,default_actions[index])
        ## check type errors
        self.assertRaises(TypeError,q.learn,1,default_actions[index],"asd",1)

    def test_qlearn_chooseaction(self):
        q=QLearn(default_actions,epsilon=0.0) # no randomness
        index=np.random.randint(0,len(default_actions))
        q.learn(1,default_actions[index],10,1)
        act,val=q.chooseAction(1,return_q=True)
        self.assertEqual(act,default_actions[index]) # only action learned
        correct_ans=[0]*len(default_actions)
        correct_ans[index]=10
        self.assertEqual(val,correct_ans) # correct qvalues

    def test_transmitter_learn(self):
        '''
        Test if the transmitter is able to select the correct actions
        '''
        acts=["a","b","c","d"]
        rews={"a":1,"b":-1,"c":3,"d":-1}
        t=Transmitter(acts)
        combs=itertools.product([False, True], repeat=len(acts)) # all possible combinations of actions
        for l in combs:
            ans=[a for (a,v) in zip(acts,l) if v] # list of choosen actions
            rew=sum([rews[l] for l in ans])       # total reward
            t.last_action=l
            t.learn(rew)
        self.assertEqual(t.choose_action(),["a","c"])

    def test_calibration(self):
        """
        Test that a calibrated agent keeps being calibrated
        """
        #init
        rem=Remote()
        ns=[Agent([],remote=rem,learn_from_neighbors=True) for _ in range(10)] # allow them to learn from neighbors for making init easier
        a=Agent(ns,remote=rem,classify=True,learn=True,calibrated=True)
        net=Network([n.get_id() for n in ns]+[a.get_id()],["s1"],1,1) # add a fake sensor, we don't use it
        a.set_network(net)
        for n in ns:
            n.set_network(net)
        # ask what to send
        choice=dict(zip(Sensor.protocol,[None]*len(Sensor.protocol)))
        labels=a.t_agent.choose_action()
        self.assertEqual(labels,["value","s_type"]) # correctly calibrated
        choice.update(dict(zip(labels,[1]*len(labels))))
        print(choice)
        # perform communication and train transmitter with rewards
        for i in range(500):
            for n in ns:
                ans,trans_cost,reward=net.send_msg(src_id=a.get_id(),dest=n,time=1,msg=choice)
                print("reward "+str(reward))
                a.t_agent.learn(reward)
        labels=a.t_agent.choose_action()
        self.assertEqual(labels,["value","s_type"]) # still correctly calibrated
