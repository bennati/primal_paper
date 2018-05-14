import random
import itertools

class QLearn:
    '''
    Taken from https://github.com/studywolf/blog/blob/master/RL/Cat%20vs%20Mouse%20exploration/qlearn_mod_random.py

    Args:
    actions: a list of actions that can be executed
    epsilon: probability of choosing a random action
    alpha: learning rate
    gamma: discount factor
    '''
    def __init__(self, actions, epsilon=0.1, alpha=0.2, gamma=0.9):
        """
        Initializes the QLearning algorithm
        """
        ## The Qtable
        self.q = {}
        ## Error parameter
        self.epsilon = epsilon
        ## Learning parameter
        self.alpha = alpha
        ## Discount parameter
        self.gamma = gamma
        ## The actions that can be learned (cols of Qtable)
        self.actions = actions

    def getQ(self, state, action):
        """
        Returns the Qvalue for a given state and action

        Args:
        state: the state
        action: the action

        Returns:
        The content of the cell in the Qtable
        """
        return self.q.get((state, str(action)), 0.0)
        # return self.q.get((state, action), 1.0)

    def learnQ(self, state, action, reward, value):
        """
        Update the Qtable

        Args:
        state: the current state
        action: the action selected
        reward: the reward obtained
        value: the new value to insert in the table
        """
        ## TODO
        oldv = self.q.get((state, str(action)), None)
        if oldv is None:
            self.q[(state, str(action))] = reward
        else:
            self.q[(state, str(action))] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        """
        Pick one action

        Args:
        state: the current state

        Kwargs:
        return_q: If true return the value in the Qtable

        Returns:
        The action or a pair (action,qvalue) if return_q in True
        """
        q = [self.getQ(state, a) for a in self.actions]
        explored=[a for (s,a) in self.q.keys() if s==state]
        if random.random() < self.epsilon and len(explored)<len(self.actions):
            #pick a random action that has not been tried before
            unexplored=[a for a in self.actions if (state,str(a)) not in self.q.keys()]
            action=random.choice(unexplored)
            assert(len(unexplored)+len(explored)==len(self.actions))
        else:
            maxQ=max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]

        if return_q: # if they want it, give it!
            return action, q
        return action

    # def chooseAction(self, state, return_q=False):
    #     """
    #     Pick one action

    #     Args:
    #     state: the current state

    #     Kwargs:
    #     return_q: If true return the value in the Qtable

    #     Returns:
    #     The action or a pair (action,qvalue) if return_q in True
    #     """
    #     q = [self.getQ(state, a) for a in self.actions]
    #     maxQ = max(q)

    #     if random.random() < self.epsilon:
    #         #action = random.choice(self.actions)
    #         minQ = min(q); mag = max(abs(minQ), abs(maxQ))
    #         q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))] # add random values to all the actions, recalculate maxQ
    #         maxQ = max(q)

    #     count = q.count(maxQ)
    #     if count > 1:
    #         best = [i for i in range(len(self.actions)) if q[i] == maxQ]
    #         i = random.choice(best)
    #     else:
    #         i = q.index(maxQ)

    #     action = self.actions[i]

    #     if return_q: # if they want it, give it!
    #         return action, q
    #     return action

    def learn(self, state1, action1, reward, state2):
        """
        Learn from experience

        Args:
        state1: the previous state
        action1: the action previously selected
        reward: the reward obtained
        state2: the current state
        """
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

class Transmitter():
    '''
    The class transmitter learns what type of information is good to send.

    Args:
    actions: a list of names identifying information the transmitter can send. The transmitter will learn what subset of actions to send. Example ["value","time","s_type","e_type","sensor_id"]
    '''
    def __init__(self,actions,eps=0.5):
        """
        Initialize the transmitter and the QLearning algorithm
        """
        assert(type(actions) is list)
        assert(len(actions)>1)
        ## The values that can be transmitted
        self.actions=actions
        ## all possible combinations of actions
        self.combs = list(itertools.product([False, True], repeat=len(actions))) # all possible combinations of actions
        ## The Qlearning algorithm
        self.Q=QLearn(self.combs,epsilon=eps,gamma=0.0)
        ## The last action executed
        self.last_action=[True]*len(self.actions) # default action

    def learn(self,reward):
        """
        Learn from experience

        Args:
        reward: the reward obtained
        """
        self.Q.learn(1,self.last_action,reward,1) # there is only one state

    def choose_action(self):
        """
        Pick one action
        """
        a=self.Q.chooseAction(1)
        self.last_action=a
        return [l for (l,b) in zip(self.actions,a) if b] # return labels of active infomation
