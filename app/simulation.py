import numpy as np
from app.remote import *
from app.agent import *
from app.remote import *
from multiprocessing import Pool
import functools

# default_norm_function=abs
# default_event_function=lambda x: abs(x)+1
default_norm_function=lambda x: x
default_event_function=lambda x: x+1


def simulate(fct,num_sensors,**kwargs):
    """
    Run a simulation and return the results
    """
    # Centralized
    part_fun_cen=functools.partial(fct,**kwargs)
    if __name__ == '__main__' and num_rep>1:
        pool=Pool()
        print("starting processes")
        ans=pool.map(part_fun_cen,num_sensors)
    else:
        print("starting process")
        ans=list(map(part_fun_cen,num_sensors))
    if len(num_sensors)>1:
        ans=list(zip(*ans))
        ans=[list(zip(*ans[0])),list(zip(*ans[1])),list(zip(*ans[2]))]
    else:
        ans=[[(a,) for a in ans[0][0]]]+[[(a,) for a in ans[0][1]]]+[[(a,) for a in ans[0][2]]]
    return ans

def simulate_cen(num_sensors,end_time,num_rep,p=0,ns=0,norm_func=default_norm_function,event_func=default_event_function,learn=False,classify=False,debug=False,calibrated=False,comm=None,comm_rem=None,supervised=False,pretrain=False,learn_from_neighbors=False,single=False):
    return simulate(fct=run_centralized,**locals())

def simulate_decen(num_sensors,end_time,num_rep,ns=0,p=0,norm_func=default_norm_function,event_func=default_event_function,learn=False,classify=False,debug=False,calibrated=False,comm=None,comm_rem=None,supervised=False,pretrain=False,learn_from_neighbors=False,single=False):
    return simulate(fct=run_decentralized,**locals())

def simulate_hier(num_sensors,num_agents,end_time,num_rep,ns=0,p=0,norm_func=default_norm_function,event_func=default_event_function,learn=False,classify=False,debug=False,calibrated=False,comm=None,comm_rem=None,supervised=False,pretrain=False,learn_from_neighbors=False,single=False):
    return simulate(fct=run_hierarchical,**locals())

def disable_training(sensors,debug=False):
    print_debug(debug,"Disabling training for all agents")
    for s in sensors:
        for a in s.get_agents():
            a.set_train_classifier(False)

def run_simulation(sensors,remote,net,end_time,p,debug=False,exogenous_events={}):
    """
    Initializes the events and triggers the measurements for all sensors at every timestep.
    If remote is initialized with supervised, it detects false negatives and communicates them to the appropriate agents.

    Args:
    sensors: a list of Sensor objects
    remote: a Remote object
    net: the network, needed to compute the costs
    end_time: the timeout
    p: probability of an event happening, if p>1 it is interpreted as the number of events

    Kwargs:
    exogenous_events: a dictionary with timesteps as keys and a list of functions as value. Each function is executed at the given timestep

    Returns: a triplet (score,cost,privacy) of dictionaries whose keys are timesteps
    """
    if p<=1:                    # probability
        print("event probability: "+str(p))
        (ground_truth,events)=init_events_prob(sensors,end_time,p)
    else:                       # number of events
        print("number of events: "+str(p))
        (ground_truth,events)=init_events(sensors,p,end_time)
    remote.set_ground_truth(ground_truth)
    prec={}
    rec={}
    f1={}
    cost={}
    privacy={}
    for t in range(end_time):
        print_debug(debug,"-------------------- Time step: "+str(t)+" --------------------")
        flags=[False]*len(sensors)
        if t in events.keys():
            for i in events[t]: # all agents perceiving an event at this time
                flags[i]=True # set the flag true for this agent
        if t in exogenous_events.keys():
            for f in exogenous_events[t]:
                f(sensors)      # call the function
        for (s,f) in zip(sensors,flags):
            if f:
                print_debug(debug,"++++++++++ Event is happening")
            else:
                print_debug(debug,"++++++++++ Normal measurement")
            m=s.measure(t,event=f,debug=debug)
            # give feedback for false negatives
            if remote.get_supervised() and f:               # it is a positive
                for agent in s.get_agents(): # all agents associated with this sensor
                    if [t,agent.get_id(),s.get_etype(),s.get_id()] not in remote.get_alarms(): # false negative
                        agent.process_feedback(t,m,True,s.get_type(),s.get_etype(),s.get_id(),debug,requested=False) # it was positive
        # record measures
        prec[t],rec[t],f1[t]=remote.compute_partial_score(t)
        cost[t]=net.get_cost()
        privacy[t]=net.get_privacy_cost()
    # compute score
    prec_tot,rec_tot,f1_tot=remote.compute_score()
    if not np.isnan(prec_tot):
        assert(prec_tot==prec[max(list(prec.keys()))])
    if not np.isnan(f1_tot):
        assert(f1_tot==f1[max(list(f1.keys()))])
    return prec,rec,f1,cost,privacy

def init_topology_sensors(num_sensors,agents,norm_func,event_func):
    sensors=[]
    if len(agents)==1:          # centralized
        sensors=[Sensor(norm_func,event_func,agents,1,1) for _ in range(num_sensors)] # assign each sensor to the same agent
    elif len(agents)==num_sensors: # decentralized
        sensors=[Sensor(norm_func,event_func,[a],1,1) for a in agents] # assign each sensor to one agent
    else:                                                              # partition the sensors
        l=range(num_sensors)
        n=len(agents)
        part=[l[i::n] for i in range(n)] # many sensors for each agent
        sensors=[]
        for a,s in zip(agents,part):
            sensors.extend([Sensor(norm_func,event_func,[a],1,1) for _ in s]) # assign one agent to many sensors
    return sensors

def init_topology(num_sensors,num_agents,remote,ns=0,norm_func=default_norm_function,event_func=default_event_function,cost_unit=1,privacy_unit=1,comm=None,comm_rem=None,pretrain=False,**kwargs):
    """
    Creates a network with a given topology.

    Args:
    num_sensors: The number of sensors in the network.
    num_agents: The number of agents
    remote: the remote object

    Kwargs:
    ns: The size of the neighborhood of each agent (percentage of population). Neighbors are selected randomly.
    norm_func: The function that describes normal events.
    event_func: The function that describes abnormal events (outliers).
    cost_unit: the cost of one transmission. It defaults to 1.
    privacy_unit: the privacy cost of one transmission. It defaults to 1.
    comm: the function that determines when to communicate to agents.
    comm_rem: the function that determines when to communicate to remote.
    pretrain: Either a boolean or a number. If boolean all agents receive that value as parameter, if a number only that specific number of agents receive True as parameter.
    train: see Agent
    learn: see Agent
    calibrated: see Agent
    classify: see Agent
    learn_from_neighbors: see Agent
    train_supervised: see Agent

    Returns: A pair (net,sensors)
    """
    ################################
    # testing centralized topology #
    ################################
    ## booleans are integers: True==1, False==0
    if pretrain<=1:
        pretrain=int(num_agents*pretrain)
    elif pretrain>num_agents:
        print("Warning: pretrain is greater than the numbe of agents. Initializing all agents with True")
        pretrain=num_agents
    params=[True]*pretrain
    params+=[False]*(num_agents-pretrain)
    np.random.shuffle(params)
    agents=[Agent([],remote=remote,pretrain=pt,**kwargs) for _,pt in zip(range(num_agents),params)]
    if comm!=None:              # set the communication function
        for a in agents:
            a.set_comm(comm)
    if comm_rem!=None:              # set the communication function
        for a in agents:
            a.set_comm_rem(comm_rem)
    if num_agents>1:       # assign neighbors
        for a in agents:
            neighs=[n for n in agents if n!= a]
            assert(len(neighs)==len(agents)-1)
            np.random.shuffle(neighs)
            neighs=neighs[:int(ns*num_agents)] # reduce
            a.set_neighbors(neighs) # assign neighbors
    sensors=init_topology_sensors(num_sensors,agents,norm_func,event_func)
    net=Network([a.get_id() for a in agents],[s.get_id() for s in sensors],cost_unit,privacy_unit)

    # give the network to agents and sensors
    for a in agents:
        a.set_network(net)
    for s in sensors:
        s.set_network(net)
    return net,sensors

var_names=["prec","rec","f1","cost","priv","avg_cost","avg_cost_rem","sr","sr_rem","cc","cc_rem"]

def return_dictionary_str(i):
    """
    Dictionary for the return format of run_topology.
    Converts an index in the return vector to a string.
    """
    dic=dict(zip(range(len(var_names)),var_names))
    return dic[i]

def return_dictionary_idx(s):
    """
    Dictionary for the return format of run_topology.
    Converts a string to an index in the return vector.
    """
    dic=dict(zip(var_names,range(len(var_names))))
    return dic[s]

def run_topology(num_sensors,num_agents,end_time,num_rep,default_dist,ns=0,p=None,norm_func=default_norm_function,event_func=default_event_function,cost_unit=1,privacy_unit=1,debug=False,comm=None,comm_rem=None,supervised=False,exogenous_events={},**kwargs):
    """
    Runs a simulation with a given topology.

    Args:
    num_sensors: The number of sensors in the network.
    num_agents: The number of agents in the network.
    end_time: The number of timestep to simulate.
    num_rep: The number of repetitions.
    default_dist: the default distance between agents and sensors (for topology initialization).

    Kwargs:
    ns: The size of the neighborhood of each agent (in % over the number of agents)
    p: The probability that an event will happen during the simulation (for each sensor). It defaults to 0.
    norm_func: The function that describes normal events.
    event_func: The function that describes abnormal events (outliers).
    cost_unit: the cost of one transmission. It defaults to 1.
    privacy_unit: the privacy cost of one transmission. It defaults to 1.
    supervised: If to give feedback to the agents or not
    comm: the function that determines when to communicate
    comm_rem: the function that determines when to communicate to remote.
    learn: see Agent
    calibrated: see Agent
    classify: see Agent
    train_supervised: see Agent
    pretrain: see Agent
    train: see Agent
    learn_from_neighbors: see Agent

    Returns: A list of dictionaries (prec,rec,f1,cost,privacy_cost,avg_cost,avg_cost_remote,success_rate,success_rate_remote)
    - prec contains the precision of the classifier
    - rec contains the recall of the classifier
    - f1 is the F1 measure of performance
    - cost contains the total transmission cost
    - priv contains the total transmission privacy cost
    - avg_cost contains the average cost of a transmission from an agent to another agent
    - avg_cost_remote contains the average cost of a transmission from an agent to remote
    - success_rate contains the fraction of transmissions between agents that are successful (always 1 if learning is disabled)
    - success_rate_remote contains the fraction of transmissions to remote that are successful (always 1 if learning is disabled)

    And the respective standard deviations
    """
    np.random.seed()
    ret=[None,None,None,None,None,None,None,None,None,None,None]
    if p==None:
        p=1
    for _ in range(num_rep):
        remote=Remote(supervised=supervised)
        net,sensors=init_topology(num_sensors,num_agents,remote,ns,norm_func,event_func,cost_unit,privacy_unit,comm=comm,comm_rem=comm_rem,**kwargs)
        net.reset_topology(default_dist)   # all connected locally
        prec,rec,f1,cost,priv=run_simulation(sensors,remote,net,end_time,p,debug=debug,exogenous_events=exogenous_events)
        # keep track of the counts as the dict avg_cost might not record all timesteps
        ret=[[b] if a==None else a+[b] for (a,b) in zip(ret,[prec,rec,f1,cost,priv,dict(net.get_avg_cost()),dict(net.get_avg_cost_remote()),dict(net.get_success_rate()),dict(net.get_success_rate_remote()),dict(net.get_comm_counter()),dict(net.get_comm_counter_remote())])]
        # ret+=np.array([(score or 0), # a valid number
        #                net.get_cost(),net.get_privacy_cost()])
        # avg_cost=dsum(avg_cost,dict(net.get_avg_cost()))
    avg=[davg(l) for l in ret]
    if num_rep>1:
        # compute the standard deviation
        sd=[]
        for (a,v) in zip(avg,ret):
            diff=[dsub(i,a) for i in v]
            sq=[dmul(i,i) for i in diff]
            sqavg=davg(sq)
            sd.append(dsqrt(sqavg))
        ci=[dmul_const(s,1.96/np.sqrt(num_rep)) for s in sd]
        for s,c in zip(sd,ci):
            for (k1,v1),(k2,v2) in zip(s.items(),c.items()):
                assert(abs(v1*1.96/np.sqrt(num_rep)-v2)<0.01)
    else:
        sd=[None]*len(avg)
        ci=[None]*len(avg)
    return [[dict_2_array(j,end_time) for j in avg],[dict_2_array(j,end_time) for j in sd],[dict_2_array(j,end_time) for j in ci]]

def dict_2_array(d,end_time):
    if d==None:
        ret=[np.nan]*end_time
    else:
        ret=[d[i] if i in d.keys() else np.nan for i in range(end_time)]
    return ret

def run_centralized(num_sensors,end_time,num_rep,p=None,ns=0,norm_func=default_norm_function,event_func=default_event_function,cost_unit=1,privacy_unit=1,supervised=False,debug=False,learn=False,classify=False,calibrated=False,comm=None,comm_rem=None,pretrain=False,learn_from_neighbors=False,single=False,exogenous_events={}):
    return run_topology(num_agents=1,default_dist=1,**locals())

def run_decentralized(num_sensors,end_time,num_rep,ns=0,p=None,norm_func=default_norm_function,event_func=default_event_function,cost_unit=1,privacy_unit=1,supervised=False,learn=False,classify=False,debug=False,calibrated=False,comm=None,comm_rem=None,pretrain=False,learn_from_neighbors=False,single=False,exogenous_events={}):
    return run_topology(num_agents=num_sensors,default_dist=0,**locals())

def run_hierarchical(num_sensors,num_agents,end_time,num_rep,ns=0,p=None,norm_func=default_norm_function,event_func=default_event_function,cost_unit=1,privacy_unit=1,supervised=False,learn=False,classify=False,debug=False,calibrated=False,comm=None,comm_rem=None,pretrain=False,learn_from_neighbors=False,single=False,exogenous_events={}):
    locs=locals().copy()
    if num_agents<=1 and num_agents>0: # percentual
        locs.update({"num_agents":int(num_sensors*num_agents)})
    return run_topology(default_dist=1,**locs)
