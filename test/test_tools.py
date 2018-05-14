import numpy as np
import unittest
from app.tools import *
from app.sensor import *
from app.agent import *
from app.remote import *

class TestTools(unittest.TestCase):
    """
    Test suite for class Tools
    """

    def __init__(self, *args, **kwargs):
        super(TestTools, self).__init__(*args, **kwargs)
        self.norm_func=abs
        self.event_func=lambda x: abs(x)+1


    def test_initEvents_noEvents_empty(self):
        s=Sensor(self.norm_func,self.event_func,[],1,1)
        ## zero events, empty output
        g,e=init_events([s],0,10)
        self.assertEqual(g,[])
        self.assertEqual(e,{})

    def test_initEventsProb_noEvents_empty(self):
        s=Sensor(self.norm_func,self.event_func,[],1,1)
        ## zero events, empty output
        g,e=init_events_prob([s],10,0)
        self.assertEqual(g,[])
        self.assertEqual(e,{})

    def test_initEvents_events_lenghtOk(self):
        ## output length
        s=Sensor(self.norm_func,self.event_func,[],1,1)
        n=10
        g,e=init_events([s],n,n)
        self.assertEqual(len(g),len(e))
        # some events might be duplicated, so the total number could be lower than n
        self.assertGreaterEqual(len(g),0)
        self.assertLessEqual(len(g),n)

    def test_initEventsProb_events_lenghtOk(self):
        ## output length
        s=Sensor(self.norm_func,self.event_func,[],1,1)
        n=10
        g,e=init_events_prob([s],n,1)
        self.assertEqual(len(g),len(e))
        # some events might be duplicated, so the total number could be lower than n
        self.assertGreaterEqual(len(g),0)
        self.assertLessEqual(len(g),n)

    def test_initEvents_manyEvents_truncated(self):
        s=Sensor(self.norm_func,self.event_func,[],1,1)
        n=20
        et=10
        g,e=init_events([s],n,et)
        self.assertEqual(len(g),len(e))
        self.assertLessEqual(len(g),et)

    def test_initEvents_events_outputOk(self):
        n=50
        etime=20
        s=Sensor(self.norm_func,self.event_func,[],1,1)
        s2=Sensor(self.norm_func,self.event_func,[],1,1)
        s3=Sensor(self.norm_func,self.event_func,[],1,1)
        slist=[s,s2,s3]
        locs=[a.get_id() for a in slist]
        g,e=init_events(slist,n,etime)
        for a,b,c in g:
            self.assertIn(a,[1]) # event type
            self.assertIn(b,range(etime)) # time
            self.assertIn(c,locs) # sensor id
        self.assertEqual(len(set(g)),len(g)) # all unique
        for i in e.keys():
            self.assertIn(i,range(etime)) # time
            self.assertIn(i,list(zip(*g))[1]) # consistent with ground truth
            self.assertEqual(len(np.unique(e[i])),len(e[i])) # all unique
            for j in e[i]:
                self.assertIn(j,range(len(slist))) # agent index

    def test_initEventsProb_events_outputOk(self):
        p=0.8
        etime=20
        s=Sensor(self.norm_func,self.event_func,[],1,1)
        s2=Sensor(self.norm_func,self.event_func,[],1,1)
        s3=Sensor(self.norm_func,self.event_func,[],1,1)
        slist=[s,s2,s3]
        locs=[a.get_id() for a in slist]
        g,e=init_events_prob(slist,etime,p)
        for a,b,c in g:
            self.assertIn(a,[1]) # event type
            self.assertIn(b,range(etime)) # time
            self.assertIn(c,locs) # sensor id
        self.assertEqual(len(set(g)),len(g)) # all unique
        for i in e.keys():
            self.assertIn(i,range(etime)) # time
            self.assertIn(i,list(zip(*g))[1]) # consistent with ground truth
            self.assertEqual(len(np.unique(e[i])),len(e[i])) # all unique
            for j in e[i]:
                self.assertIn(j,range(len(slist))) # agent index

    def test_initCentralizedTopology_default_netOk(self):
        a=Agent([])
        s1=Sensor(self.norm_func,self.event_func,[a],1,1)
        s2=Sensor(self.norm_func,self.event_func,[a],1,1)
        n=init_centralized_topology([a],[s1,s2],1,1)
        ## network initialization
        self.assertEqual(type(n),Network)
        self.assertIn(a.get_id(),n.agent_ids) # agent is registered
        for s in [s1,s2]:
            self.assertIn(s.get_id(),n.sensor_ids) # sensors are registered

    def test_initCentralizedTopology_default_topologyOk(self):
        a=Agent([])
        s1=Sensor(self.norm_func,self.event_func,[a],1,1)
        s2=Sensor(self.norm_func,self.event_func,[a],1,1)
        n=init_centralized_topology([a],[s1,s2],1,1)
        ## topology initialization
        self.assertEqual(n.topology.shape,(2,1))
        self.assertEqual(n.topology[0,0],1) # distance is correct

    def test_initDecentralizedTopology_default_netOk(self):
        a1=Agent([])
        a2=Agent([])
        s1=Sensor(self.norm_func,self.event_func,[a1],1,1)
        s2=Sensor(self.norm_func,self.event_func,[a2],1,1)
        n=init_decentralized_topology([a1,a2],[s1,s2],1,1)
        ## network initialization
        self.assertEqual(type(n),Network)
        for a in [a1,a2]:
            self.assertIn(a.get_id(),n.agent_ids) # agents are registered
        for s in [s1,s2]:
            self.assertIn(s.get_id(),n.sensor_ids) # sensors are registered

    def test_initDecentralizedTopology_default_topologyOk(self):
        a1=Agent([])
        a2=Agent([])
        s1=Sensor(self.norm_func,self.event_func,[a1],1,1)
        s2=Sensor(self.norm_func,self.event_func,[a2],1,1)
        n=init_decentralized_topology([a1,a2],[s1,s2],1,1)
        ## topology initialization
        self.assertEqual(n.topology.shape,(2,2))
        self.assertEqual(n.topology[0,0],0)
