import unittest
from app.agent import *
from app.network import *
from app.remote import *

class TestSensor(unittest.TestCase):
    """
    Test suite for class Sensor
    """

    def __init__(self, *args, **kwargs):
        super(TestSensor, self).__init__(*args, **kwargs)
        f1=lambda x:x           # mean is 0
        self.m1=0
        f2=lambda x: abs(x)     # mean is 0.5
        self.m2=0.5
        a=Agent([])
        self.s=Sensor(f1,f2,[a],1,1)
        n=Network([a.get_id()],[self.s.get_id()],1,1)
        r=Remote()
        self.s.set_network(n)
        a.set_network(n)
        a.set_remote(r)
        self.assertNotEqual(a.dic,{})

    def test_measure_noevent_m1(self):
        m=0
        for _ in range(1000):
            m+=self.s.measure("noevent")
        m/=1000.0
        self.assertLess(abs(m),self.m1+0.1) # mean is 0

    def test_measure_event_m2(self):
        m=0
        for _ in range(1000):
            m+=self.s.measure("event",event=True)
        m/=1000.0
        self.assertGreater(m,self.m2-0.2)  # mean is 0.5
