import unittest
from app.remote import *
from app.sensor import *

class TestRemote(unittest.TestCase):
    """
    Test suite for class Remote
    """

    def __init__(self, *args, **kwargs):
        super(TestRemote, self).__init__(*args, **kwargs)
        self.r=Remote(supervised=True)
        self.e_type=1
        self.t=1
        self.loc=2
        gt=(self.e_type,self.t,self.loc)
        self.r.set_ground_truth([gt])

    def test_supervised_trueAlarm_true(self):
        ans=self.r.send_alarm(99,self.e_type,self.t,self.loc)
        self.assertTrue(ans)

    def test_supervised_falseAlarm_false(self):
        ans=self.r.send_alarm(99,self.e_type,self.t+2,self.loc)
        self.assertFalse(ans)

    def test_unsupervised_trueAlarm_none(self):
        # disable supervision, returns always None
        self.r.set_supervised(False)
        ans=self.r.send_alarm(99,self.e_type,self.t,self.loc)
        self.assertEqual(ans,None)

    def test_unsupervised_FalseAlarm_none(self):
        # disable supervision, returns always None
        self.r.set_supervised(False)
        ans=self.r.send_alarm(99,self.e_type,self.t+2,self.loc)
        self.assertEqual(ans,None)

    def test_computeScore_init_zero(self):
        rem=Remote()
        s1=Sensor(abs,lambda x: abs(x)+1,[],1,1)
        s2=Sensor(abs,lambda x: abs(x)+1,[],1,1)
        s3=Sensor(abs,lambda x: abs(x)+1,[],1,1)
        gt,events=init_events([s1,s2,s3],5,10)
        ## without running the simulation, score is 0
        rem.set_ground_truth(gt)
        score=(0,0)
        score=score+(compute_f1(*score),)
        self.assertEqual(rem.compute_score(),score) # 1 because no event has been reported (all false negatives)

    def test_computeScore_correctAns_one(self):
        rem=Remote()
        s1=Sensor(abs,lambda x: abs(x)+1,[],1,1)
        s2=Sensor(abs,lambda x: abs(x)+1,[],1,1)
        s3=Sensor(abs,lambda x: abs(x)+1,[],1,1)
        gt,events=init_events([s1,s2,s3],5,10)
        ## score of a correct answer is 1
        rem.set_ground_truth(gt)
        correct=[(t,10,e,l) for (e,t,l) in gt]
        rem.set_alarms(correct)
        self.assertEqual(rem.compute_score(),(1,1,1))

    def test_computeScore_divideByZero_none(self):
        rem=Remote()
        ## divide by zero
        gt=[]
        rem.set_ground_truth(gt)
        self.assertEqual(rem.compute_score(),(np.nan,np.nan,np.nan))

    def test_computeScore_falsePositive_one(self):
        rem=Remote()
        s1=Sensor(abs,lambda x: abs(x)+1,[],1,1)
        s2=Sensor(abs,lambda x: abs(x)+1,[],1,1)
        s3=Sensor(abs,lambda x: abs(x)+1,[],1,1)
        gt,events=init_events([s1,s2,s3],5,10)
        rem.set_ground_truth(gt)
        ## a correct answer and a false positive
        correct=[(t,10,e,l) for (e,t,l) in gt]+[(1,10,3,999)]
        print(correct)
        rem.set_alarms(correct)
        score=compute_prec_recall(len(gt),1,0)
        score=score+(compute_f1(*score),)
        self.assertEqual(rem.compute_score(),score)
