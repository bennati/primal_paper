import numpy as np
import unittest
from app.agent import *

class TestAgent(unittest.TestCase):
    """
    Test suite for class Agent
    """


    def __init__(self, *args, **kwargs):
        super(TestAgent, self).__init__(*args, **kwargs)
        self.a=Agent([])
        self.a.enable_classification=True

    def test_parseAns_majority(self):
        ## check majority vote
        ret=self.a.parse_answers([(True,1),(False,1),(True,1),(None,1)])
        self.assertTrue(ret[0])
        self.assertEqual(ret[1],1)
        ret=self.a.parse_answers([(False,1),(False,1),(True,1),(None,1)])
        self.assertFalse(ret[0])
        self.assertEqual(ret[1],1)
        ret=self.a.parse_answers([(True,1),(True,2),(True,3),(None,1)])
        self.assertTrue(ret[0])
        self.assertEqual(ret[1],2) # average

    def test_parseAns_none_none(self):
        ## check None handling
        ret=self.a.parse_answers([(None,1)])
        self.assertEqual(ret[0],None)

    def test_parseAns_wrongInput(self):
        ## check input format
        ret=self.a.parse_answers([])
        self.assertEqual(ret[0],None)
        self.assertRaises(TypeError,self.a.parse_answers,[("asd",1)]) # cannot be cast to bool

    def test_initClassifier_default_empty(self):
        self.assertEqual(self.a.dic,{})

    def test_initClassifier_init_correct(self):
        key1=1
        key2=2
        ## check that dictionary is added correctly
        self.a.init_classifier(key1,"asd")
        self.assertNotEqual(self.a.dic,{})
        self.assertNotEqual(self.a.dic[key1],{})     # right key
        self.assertEqual(len(self.a.dic[key1]),1)
        self.assertEqual(self.a.dic.get(key2),None) # wrong key
        self.assertIsInstance(self.a.dic[key1]["asd"],Classifier)

    def test_initClassifier_extend_correct(self):
        key=1
        ## check that dictionary is extended correctly
        self.a.init_classifier(key,"asd")
        self.a.init_classifier(key,"asd2")
        self.assertEqual(len(self.a.dic[key]),2)

    def test_classify_wrongKey_exception(self):
        self.assertRaises(KeyError,self.a.classify,"UNKNOWN KEY",10)

    def test_classify_classificationDefault(self):
        value=10
        stype=1
        sid="asd3"
        self.a.init_classifier(stype,sid)
        self.a.register_source(stype,sid)
        self.assertEqual(self.a.classify(stype,0)[0],None)
        self.a.learn(0,False,stype,sid)
        self.a.learn(0.1,False,stype,sid)
        self.assertTrue(self.a.classify(stype,value)[0])

    def test_classify_disableClassification_alwaysTrue(self):
        # disable classification
        self.a.set_classify(False)
        count=0
        end=100
        stype=1
        for _ in range(end):
            ans=self.a.classify(stype,np.random.randint(0,100))
            if ans:
                count+=1
        self.assertEqual(count,end)

    def test_defineProtocol_unknownLabels_notSelected(self):
        keys=["a","b"]
        vals=[10,20]
        self.a.enable_learning=True
        ans=self.a._Agent__define_protocol(zip(keys,vals))
        for k in ans.keys():
            self.assertIn(k,keys)
        self.assertEqual(list(ans.values()),[None,None])

    def test_defineProtocol_knownLabels_selected(self):
        keys=[Sensor.protocol[1],"b"]
        vals=[10,20]
        count={}
        self.a.enable_learning=True
        for _ in range(100):    # repeat as it is random
            ans=self.a._Agent__define_protocol(zip(keys,vals))
            for k in ans.keys():
                self.assertIn(k,keys)
            count,ctr=dsum([count, ans])
        self.assertGreater(count[Sensor.protocol[1]],0) # value is considered
        self.assertNotIn("b",count.keys())                  # value is not considered

    def test_defineProtocol_learningDisabled_everythingSelected(self):                ## only known labels are selected
        ## if learning is disabled, all values are selected
        keys=[Sensor.protocol[1],"b"]
        vals=[10,20]
        self.a.set_learn(False)
        ans=self.a._Agent__define_protocol(zip(keys,vals)) # deterministic
        for k in ans.keys():
            self.assertIn(k,keys)
        self.assertGreater(ans[Sensor.protocol[1]],0) # value is considered
        self.assertGreater(ans["b"],0) # value is considered anyways
