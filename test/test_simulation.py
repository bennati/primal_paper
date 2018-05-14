import numpy as np
import unittest
from app.simulation import *
import functools
from multiprocessing import Pool

class TestSimulation(unittest.TestCase):
    """
    Test suite for class Simulation
    """

    def __init__(self, *args, **kwargs):
        super(TestSimulation, self).__init__(*args, **kwargs)
        self.num_sensors=10
        self.num_agents=5
        self.prob=0.5       # two for each agent
        self.end_time=4
        self.cu=1
        self.pu=1
        self.ns=0.4
        n_c=Network(["a1"],["s1"],self.cu,self.pu)
        self.stoa_c=n_c.compute_costs(Sensor.protocol, # everything is transmitted
                                      "s1","a1",cost_unit=self.cu,privacy_unit=self.pu,distance=1)
        self.ator_c=n_c.compute_costs(Sensor.protocol, # everything is transmitted
                                      "a1","remote",cost_unit=self.cu,privacy_unit=self.pu)
        n_d=Network(["a1","a2"],["s1"],self.cu,self.pu)
        self.stoa_d=0 # in decentralized all sensors are directly connected to the agents, no communication costs
        self.atoa_d=n_d.compute_costs(Sensor.protocol, # everything is transmitted
                                      "a2","a1",cost_unit=self.cu,privacy_unit=self.pu,distance=1)
        self.ator_d=n_d.compute_costs(Sensor.protocol, # everything is transmitted
                                      "a1","remote",cost_unit=self.cu,privacy_unit=self.pu)
        #hierarchical
        self.stoa_h=self.stoa_c
        self.atoa_h=self.atoa_d
        self.ator_h=self.ator_d


    def test_centralizedInput_wrongType_exception(self):
        ## wrong type
        self.assertRaises(TypeError,run_centralized,num_sensors="nonumber",end_time=1,num_rep=1)
        self.assertRaises(TypeError,run_centralized,num_sensors=1,end_time="nonumber",num_rep=1)
        self.assertRaises(TypeError,run_centralized,num_sensors=1,end_time=1,num_rep="nonumber")
        self.assertRaises(TypeError,run_centralized,num_sensors=1,end_time=1,num_rep=1,p="nonumber") # casted to int

    def test_initTopology_init_correct(self):
        remote=Remote()
        net,sensors=init_topology(self.num_sensors,self.num_agents,remote,ns=self.ns)
        print("n "+str(net))
        self.assertEqual(type(net),Network)
        self.assertEqual(len(sensors),self.num_sensors)
        self.assertEqual(net.topology.shape,(self.num_sensors,self.num_agents))
        ## check initialization was correct
        for s in sensors:
            self.assertTrue(s.net) # net is initialized
            for a in s.get_agents():
                self.assertTrue(a.net) # net is initialized
                ## neighbors
                self.assertEqual(len(a.ns),int(self.ns*self.num_agents)) # neighbors are initialized

    ### Centralized
    def test_runCentralized_events_scoreNotZero(self):
        ## if many events score is not 0
        asd=run_centralized(self.num_sensors,self.end_time,20,p=self.prob,cost_unit=self.cu,privacy_unit=self.pu,learn=False,classify=False)
        self.assertNotEqual(last_val(asd[0][0]),0) # final precision
        self.assertNotEqual(last_val(asd[0][1]),0) # final recall
        self.assertNotEqual(last_val(asd[0][3]),0) # final cost
        self.assertNotEqual(last_val(asd[0][4]),0) # final privacy

    def test_runCentralized_noInt_noEvents_scoreZero(self):
        ## check the costs, no events
        asd=run_centralized(self.num_sensors,self.end_time,1,p=0,cost_unit=self.cu,privacy_unit=self.pu,learn=False,classify=False)
        ## if 0 events score is 0
        self.assertEqual(asd[0][0],{}) # final precision
        self.assertEqual(asd[0][1],{}) # final recall

    def test_computeCostsCen_noInt_NoEvents_transmission(self):
        asd=run_centralized(self.num_sensors,self.end_time,1,p=0,cost_unit=self.cu,privacy_unit=self.pu,learn=False,classify=False)

        ## without learning and classification, everything is transmitted
        self.assertEqual(self.stoa_c,len(Sensor.protocol)*self.cu+1*self.pu)
        self.assertEqual(self.ator_c,len(Sensor.protocol)*self.cu+1*self.pu)
        self.assertEqual(last_val(asd[0][3])+last_val(asd[0][4]),   # cost and privacy cost
                         (self.stoa_c*self.num_sensors # cost of sensors
                         +self.ator_c*self.num_sensors)     # cost of agents reporting the event
                         *self.end_time)

    def test_computeCostsCen_noInt_Events_scoreOne(self):
        asd=run_centralized(self.num_sensors,self.end_time,1,p=self.prob,cost_unit=self.cu,privacy_unit=self.pu,learn=False,classify=False)
        self.assertEqual(last_val(asd[0][1]),1) # everything is transmitted so the recall is 1

    def test_computeCostsCen_Int_noEvents_score(self):
        ## if learning and classification are enabled, the cost gets reduced
        asd=run_centralized(self.num_sensors,self.end_time,1,p=0,cost_unit=self.cu,privacy_unit=self.pu,learn=True,classify=True)
        self.assertLess(last_val(asd[0][3])+last_val(asd[0][4]),   # cost and privacy cost
                         (self.stoa_c*self.num_sensors # cost of sensors
                         +self.ator_c*self.num_sensors)     # cost of agents reporting the event
                         *self.end_time)

    def test_computeCostsCen_Int_Events_score(self):
        ret=[]
        for _ in range(10):
            asd=run_centralized(self.num_sensors,self.end_time,1,p=self.prob,cost_unit=self.cu,privacy_unit=self.pu,learn=True,classify=True)
            ret.append(last_val(asd[0][0]))
        self.assertLess(np.asarray(ret).mean(),1) # with learning and classification enabled, mistakes are produced


    ### Decentralized
    def test_runDecentralized_events_scoreNotZero(self):
        ## if many events score is not 0
        asd=run_decentralized(self.num_sensors,self.end_time,20,p=self.prob,ns=self.ns,cost_unit=self.cu,privacy_unit=self.pu,learn=False,classify=False)
        self.assertNotEqual(last_val(asd[0][0]),0) # final precision
        self.assertNotEqual(last_val(asd[0][1]),0) # final recall
        self.assertNotEqual(last_val(asd[0][3]),0) # final cost
        self.assertNotEqual(last_val(asd[0][4]),0) # final privacy

    def test_runDecentralized_noEvents_scoreZero(self):
        asd=run_decentralized(self.num_sensors,self.end_time,1,p=0,cost_unit=self.cu,privacy_unit=self.pu,learn=False,classify=False,ns=self.ns)
        ## if 0 events score is 0
        self.assertEqual(asd[0][0],{}) # final precision
        self.assertEqual(asd[0][1],{}) # final recall

    def test_computeCostsDec_noInt_NoEvents_transmission(self):
        asd=run_decentralized(self.num_sensors,self.end_time,1,p=0,cost_unit=self.cu,privacy_unit=self.pu,learn=False,classify=False,ns=self.ns)
        ## without learning and classification, everything is transmitted
        num_agents=self.num_sensors  # by default
        self.assertEqual(self.atoa_d,len(Sensor.protocol)*self.cu+1*self.pu)
        self.assertEqual(self.ator_d,len(Sensor.protocol)*self.cu+1*self.pu)
        self.assertEqual(last_val(asd[0][3])+last_val(asd[0][4]),   # cost and privacy cost
                         self.num_sensors*(
                             self.stoa_d
                             +self.ator_d     # cost of agents reporting the event to remote
                             +self.atoa_d*int(self.ns*num_agents)) # agents communicating between each other
                         *self.end_time)

    def test_computeCostsDec_noInt_Events_scoreOne(self):
        asd=run_decentralized(self.num_sensors,self.end_time,1,p=self.prob,cost_unit=self.cu,privacy_unit=self.pu,learn=False,classify=False,ns=self.ns)
        self.assertEqual(last_val(asd[0][1]),1) # everything is transmitted so the recall is 1

    def test_computeCostsDec_Int_noEvents_score(self):
        ## if learning and classification are enabled, the cost gets reduced
        num_agents=self.num_sensors  # by default
        asd=run_decentralized(self.num_sensors,self.end_time,1,p=0,ns=self.ns,cost_unit=self.cu,privacy_unit=self.pu,learn=True,classify=True)
        self.assertLess(last_val(asd[0][3])+last_val(asd[0][4]),   # cost and privacy cost
                         self.num_sensors*(
                             self.stoa_d
                             +self.ator_d     # cost of agents reporting the event to remote
                             +self.atoa_d*int(self.ns*num_agents)) # agents communicating between each other
                         *self.end_time)

    def test_computeCostsDec_Int_Events_score(self):
        ret=[]
        for _ in range(10):
            asd=run_decentralized(self.num_sensors,self.end_time,1,p=self.prob,ns=self.ns,cost_unit=self.cu,privacy_unit=self.pu,learn=True,classify=True)
            ret.append(last_val(asd[0][0]))
        self.assertLess(np.asarray(ret).mean(),1) # with learning and classification enabled, mistakes are produced

    ### hierarchical
    def test_runHierarchical_events_scoreNotZero(self):
        ## if many events score is not 0
        asd=run_hierarchical(self.num_sensors,self.num_agents,self.end_time,20,p=self.prob,ns=self.ns,cost_unit=self.cu,privacy_unit=self.pu,learn=False,classify=False)
        self.assertNotEqual(last_val(asd[0][0]),0) # final precision
        self.assertNotEqual(last_val(asd[0][1]),0) # final recall
        self.assertNotEqual(last_val(asd[0][3]),0) # final cost
        self.assertNotEqual(last_val(asd[0][4]),0) # final privacy

    def test_runHierarchical_noEvents_scoreZero(self):
        asd=run_hierarchical(self.num_sensors,self.num_agents,self.end_time,1,p=0,cost_unit=self.cu,privacy_unit=self.pu,learn=False,classify=False,ns=self.ns)
        ## if 0 events score is 0
        self.assertEqual(asd[0][0],{}) # final precision
        self.assertEqual(asd[0][1],{}) # final recall

    def test_computeCostsHier_noInt_NoEvents_transmission(self):
        asd=run_hierarchical(self.num_sensors,self.num_agents,self.end_time,1,p=0,cost_unit=self.cu,privacy_unit=self.pu,learn=False,classify=False,ns=self.ns)
        ## without learning and classification, everything is transmitted
        self.assertEqual(self.atoa_h,len(Sensor.protocol)*self.cu+1*self.pu)
        self.assertEqual(self.ator_h,len(Sensor.protocol)*self.cu+1*self.pu)
        self.assertEqual(last_val(asd[0][3])+last_val(asd[0][4]),   # cost and privacy cost
                         self.num_sensors*(
                             self.stoa_h
                             +self.ator_h     # cost of agents reporting the event to remote
                             +self.atoa_h*int(self.ns*self.num_agents)) # agents communicating between each other
                         *self.end_time)

    def test_computeCostsHier_noInt_Events_scoreOne(self):
        asd=run_hierarchical(num_sensors=self.num_sensors,num_agents=self.num_agents,end_time=self.end_time,num_rep=1,ns=self.ns,p=self.prob,cost_unit=self.cu,privacy_unit=self.pu,learn=False,classify=False,debug=True)
        print(asd)
        self.assertEqual(last_val(asd[0][1]),1) # everything is transmitted so the recall is 1

    def test_computeCostsHier_Int_noEvents_score(self):
        ## if learning and classification are enabled, the cost gets reduced
        asd=run_hierarchical(self.num_sensors,self.num_agents,self.end_time,1,p=0,ns=self.ns,cost_unit=self.cu,privacy_unit=self.pu,learn=True,classify=True)
        self.assertLess(last_val(asd[0][3])+last_val(asd[0][4]),   # cost and privacy cost
                         self.num_sensors*(
                             self.stoa_h
                             +self.ator_h     # cost of agents reporting the event to remote
                             +self.atoa_h*int(self.ns*self.num_agents)) # agents communicating between each other
                         *self.end_time)

    def test_computeCostsHier_Int_Events_score(self):
        ret=[]
        for _ in range(10):
            asd=run_hierarchical(self.num_sensors,self.num_agents,self.end_time,1,ns=self.ns,p=self.prob,cost_unit=self.cu,privacy_unit=self.pu,learn=True,classify=True)
            ret.append(last_val(asd[0][0]))
        self.assertLess(np.asarray(ret).mean(),1) # with learning and classification enabled, mistakes are produced

    def test_simulate_oneSensor_oneRep(self):
        """
        tests output format
        """
        num_rep=1
        n_sens=[5]
        ans=run_decentralized(4,2,num_rep,p=0.5,ns=0.8)
        correct=np.asarray(ans).shape+(1,)
        ans=simulate_cen(n_sens,2,1,p=0.5)
        self.assertEqual(np.asarray(ans).shape,correct)
        ans=simulate_decen(n_sens,2,1,p=0.5,ns=0.8)
        self.assertEqual(np.asarray(ans).shape,correct)
        ans=simulate_hier(n_sens,2,2,1,p=0.5,ns=0.8)
        self.assertEqual(np.asarray(ans).shape,correct)

    def test_simulate_oneSensor_manyRep(self):
        """
        tests output format
        """
        num_rep=3
        n_sens=[5]
        ans=run_decentralized(4,2,num_rep,p=0.5,ns=0.8)
        correct=np.asarray(ans).shape+(1,)
        ans=simulate_cen(n_sens,2,1,p=0.5)
        self.assertEqual(np.asarray(ans).shape,correct)
        ans=simulate_decen(n_sens,2,1,p=0.5,ns=0.8)
        self.assertEqual(np.asarray(ans).shape,correct)
        ans=simulate_hier(n_sens,2,2,1,p=0.5,ns=0.8)
        self.assertEqual(np.asarray(ans).shape,correct)

    def test_simulate_moreSensors_oneRep(self):
        """
        tests output format
        """
        num_rep=1
        n_sens=[5,6,7]
        ans=run_decentralized(4,2,num_rep,p=0.5,ns=0.8)
        correct=np.asarray(ans).shape+(len(n_sens),)
        ans=simulate_cen(n_sens,2,1,p=0.5)
        self.assertEqual(np.asarray(ans).shape,correct)
        ans=simulate_decen(n_sens,2,1,p=0.5,ns=0.8)
        self.assertEqual(np.asarray(ans).shape,correct)
        ans=simulate_hier(n_sens,2,2,1,p=0.5,ns=0.8)
        self.assertEqual(np.asarray(ans).shape,correct)

    def test_simulate_moreSensors_manyRep(self):
        """
        tests output format
        """
        num_rep=3
        n_sens=[5,6,7]
        ans=run_decentralized(4,2,num_rep,p=0.5,ns=0.8)
        correct=np.asarray(ans).shape+(len(n_sens),)
        ans=simulate_cen(n_sens,2,1,p=0.5)
        self.assertEqual(np.asarray(ans).shape,correct)
        ans=simulate_decen(n_sens,2,1,p=0.5,ns=0.8)
        self.assertEqual(np.asarray(ans).shape,correct)
        ans=simulate_hier(n_sens,2,2,1,p=0.5,ns=0.8)
        self.assertEqual(np.asarray(ans).shape,correct)
