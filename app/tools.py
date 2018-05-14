from app.network import *
from collections import defaultdict
import pickle

def print_debug(debug,str):
    if debug:
        print(str)

def last_val(dic):
    return dic[max(dic.keys())]

def init_events(sensors,n_events,end_time):
    """
    Args:
    sensors: a list of sensors
    n_events: how many events will happen
    end_time: how long will the simulation last

    Returns: a pair (ground_truth,events) where
    ground truth is a list of triplets (event_type,time,sensor_id) and
    events is a dictionary where the keys are times and the values are the sensor index in the vector
    """
    assert(type(sensors)==list and len(sensors)>0)
    ground_truth=[]
    events={}
    if n_events:
        random_times=np.random.randint(0,high=end_time,size=int(n_events)) # a list of random times
        random_locations=np.random.randint(0,high=len(sensors),size=int(n_events)) # a list of indeces of random sensors
        s_types=[sensors[i].s_type for i in random_locations]                # the event types associated with the random sensors
        loc_ids=[sensors[i].get_id() for i in random_locations]              # the ID associated with the random sensors
        ## Set the ground truth
        ground_truth=list(zip(s_types,random_times,loc_ids))
        ground_truth=list(set(ground_truth)) # remove duplicates, it could be shorter than before as two random numbers could be equal
        events={} # time and location at which an event is happening, remove duplicates
        for (t,l) in zip(random_times,random_locations):
            if t in events:
                if l not in events[t]: # not a duplicate
                    events[t].append(l)
            else:
                events[t]=[l]
    return (ground_truth,events)

def init_events_prob(sensors,end_time,prob):
    """
    Args:
    sensors: a list of sensors
    end_time: how long will the simulation last
    prob: the probability that each sensor will measure an event

    Returns: a pair (ground_truth,events) where
    ground truth is a list of triplets (event_type,time,sensor_id) and
    events is a dictionary where the keys are times and the values are the sensor index in the vector
    """
    assert(type(sensors)==list and len(sensors)>0)
    ground_truth=[]
    events={}
    if prob:
        for i in range(len(sensors)):
            s=sensors[i]
            l=s.get_id()
            for t in range(end_time):
                if np.random.uniform() < prob:
                    ground_truth.append((s.s_type,t,l))
                    if t in events:
                        if l not in events[t]: # not a duplicate, it should never happen but whatever...
                            events[t].append(i)
                        else:
                            print("Warning! Duplicate!? something must be wrong...")
                    else:
                        events[t]=[i]
    return (ground_truth,events)

def init_centralized_topology(agent,sensors,cost_unit,privacy_unit):
    """
    Initialize the network to be centralized: only one agent connected to all sensors.

    Args:
    agent: A list containing only one agent.
    sensors: A list containing sensors.
    cost_unit: To be given to the network
    privacy_unit: To be given to the network

    Returns: A new Network object
    """
    try:
        assert(len(agent)==1)
    except AssertionError:
        print("Error: Only one agent is required to initialize a centralized network")
    agent_id=agent[0].get_id()
    net=Network([agent_id],[s.get_id() for s in sensors],cost_unit,privacy_unit)
    for s in sensors:
        net.set_topology(s.get_id(),agent_id,1) # set all distances to 1
    return net

def init_decentralized_topology(agents,sensors,cost_unit,privacy_unit):
    """
    Initialize the network to be decentralized: one agent connected to each sensor.

    Args:
    agents: A list containing agents.
    sensors: A list containing sensors.
    cost_unit: To be given to the network
    privacy_unit: To be given to the network

    Returns: A new Network object
    """
    try:
        assert(len(agents)==len(sensors))
    except AssertionError:
        print("Error: Number of agents and sensors must be the same to initialize a decentralized network")
    net=Network([a.get_id() for a in agents],[s.get_id() for s in sensors],cost_unit,privacy_unit)
    for a in agents:
        for s in sensors:
            net.set_topology(s.get_id(),a.get_id(),0) # set all distances to 0
    return net

# def dsum(*dicts):
#     """
#     sums many dictionarys over their keys
#     """
#     ret = defaultdict(float)
#     for d in dicts:
#         for k, v in d.items():
#             ret[k] += (v or 0)  # deal with NoneType
#     return dict(ret)

# def ddiv(dic,num):
#     """
#     divides the values of a dictionary by a number
#     """
#     for (k,v) in dic.items():
#         dic[k]=v/float(num)
#     return dic

def dsum(dicts):
    """
    sums many dictionarys over their keys.
    Warning: The output vector will NOT contain keys for which all values are None or nan
    """
    ret = defaultdict(float)
    counts=defaultdict(float)
    for d in dicts:
        for k, v in d.items():
            if v!=None and not np.isnan(v): # valid value
                ret[k] += v  # deal with NoneType
                counts[k]+=1
    return dict(ret),dict(counts)

def dsub(d1,d2):
    """
    subtracts two dictionarys over their keys.
    Warning: The output vector will NOT contain keys for which all values are None or nan
    """
    ret=defaultdict(float)
    for k, v in d1.items():
        if v!=None and not np.isnan(v): # valid value
            ret[k]=d1[k]
    for k, v in d2.items():
        if v!=None and not np.isnan(v): # valid value
            ret[k] -= np.asarray(v)  # deal with NoneType
    return dict(ret)

def dmul(dic,counts):
    """
    multiplies the values of a dictionary by a number
    """
    ret=defaultdict(float)
    for (k,v) in dic.items():
        ret[k]=v*counts[k]
    return dict(ret)

def dmul_const(dic,const):
    """
    multiplies the values of a dictionary by a constant
    """
    ret=defaultdict(float)
    for (k,v) in dic.items():
        ret[k]=v*const
    return dict(ret)

def ddiv(dic,counts):
    """
    divides the values of a dictionary by a number
    """
    ret=defaultdict(float)
    for (k,v) in dic.items():
        ret[k]=v/counts[k]
    return dict(ret)

def dsqrt(dic):
    """
    computes the squared root of the values of a dictionary
    """
    ret=defaultdict(float)
    for (k,v) in dic.items():
        ret[k]=np.sqrt(v)
    return dict(ret)

def davg(dicts):
    """
    averages many dictionaries

    Args:
    dicts: a list of dictionaries

    Returns: A dictionary
    """
    ret,counts=dsum(dicts)
    return ddiv(ret,counts)


def reshape_output(ans):
    """
    Args:
    ans: a map object of lists shaped as the output of run_(de)centralized. A list of 4 dictionaries

    Returns:
    A list of 4 dictionaries, containing the sum of the dictionaries in the respective position in the input

    Example:
    a1=(d11,...,d14)
    ...
    ak=(dk1,...,dk4)

    ans=reshape_output(map(a1,...,ak))
    ans=(d11+...+dk1,...,d14+...+dk4)
    """
    return [davg(a) for a in list(zip(*ans))]

def compute_prec_recall(tp,fp,fn):
    precision=(tp/float(tp+fp) if tp+fp>0 else 0)
    recall=(tp/float(tp+fn) if tp+fn>0 else 0)
    return precision,recall

def compute_f1(precision,recall):
    return 2*((precision*recall)/(precision+recall)) if (not np.isnan(precision) and not np.isnan(recall) and precision+recall>0) else np.nan

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def cumulative_to_delta(v):
    return [b-a for a,b in pairwise([0]+list(v))]

def save_obj(directory,obj, name ):
    with open(directory+'/obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, 0)  # text format

def load_obj(directory,name ):
    with open(directory+'/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
