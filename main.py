from app.simulation import *
import functools
from multiprocessing import Pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import numpy as np

default_values_n=[10,50]
default_values_ns=[0.0,0.2,0.5]
default_values_p=[0.2,0.5]
default_values_nevents=[100,400]
default_const_n=50
default_const_ns=0.2
default_const_p=0.2
default_const_nevents=100
default_treatments=[("No classification",{"classify":False}),
                    ("Classification",{"classify":True,"supervised":True}), # classification
                # ("_learn",{"learn":True,"classify":True}), # learning
                    ("Learning",{"learn":True,"classify":True,"calibrated":True,"supervised":True}), # learning and calibrated
                    ("Learning ",{"learn":True,"classify":True,"calibrated":True,"supervised":True,"single":True})] # SINGLE


lab_ns="Neighborhood size (fraction of population)"
lab_n="Number of sensors"
lab_p="Probability"
lab_time="Time"
lab_alg="Algorithm"
lab_cost="Cost"
lab_priv="Privacy cost"
lab_cc="No. of messages"
lab_ccrem="No. of messages superv."
lab_prec="Precision"
lab_f1="F-Measure"
lab_rec="Recall"

def plot_line(x,y,sd,sty1,sty2,ax,ylim,linewidth):
    if not all(np.isnan(y)):
        y=y if ylim else cumulative_to_delta(y)
        if sd!=None:
            s=sd if ylim else cumulative_to_delta(sd)
        else:
            s=None
        try:
            ax.plot(x,y,color=sty1,linestyle=sty2,linewidth=linewidth)
        except:
            print("----------------------------------------")
            ax.plot(x,y,color=sty2,linestyle=sty1,linewidth=linewidth)
        if s!=None:
            try:
                ax.fill_between(x,np.asarray(y)-np.asarray(s),np.asarray(y)+np.asarray(s),alpha=0.2,linestyle=sty2,facecolor=sty1)
            except:
                ax.fill_between(x,np.asarray(y)-np.asarray(s),np.asarray(y)+np.asarray(s),alpha=0.2,linestyle=sty1,facecolor=sty2)

def plot(support,curves,curve_names,tit="",xlab="Timesteps",ylab="Y",Z_names=None,filename="out.pdf",ylim=None,style1=['b','r','g','y','c','k'],artist1=None,style2=['-','--',':','-.'],artist2=None,num_cols=None,font_size=16):
    """
    Args:
    curves: a list of pairs containing a line to be plotted and its standard deviation
    """
    fig, ax1 = plt.subplots()
    fig.suptitle(tit)
    plt.xlabel(xlab,fontsize=font_size)
    plt.ylabel(ylab,fontsize=font_size)
    ax1.tick_params(labelsize=font_size)
    ax1.set_ymargin(0.1)        # increase the margin on top of the graph, allows to see points that have a maximum value
    if np.asarray(ylim).shape==(2,):
        ax1.set_ylim(ylim)
    # elif ylim is True:          # not specified: compute
    #     ymin=None
    #     ymax=None
    #     for (line,sd) in curves:
    #         # lo=np.min(np.asarray(line)-np.asarray(sd))
    #         hi=np.max(np.asarray(line)+np.asarray(sd))
    #         # if not np.isnan(lo) and (ymin is None or ymin > lo):
    #         #     ymin=lo
    #         if not np.isnan(hi) and (ymax is None or ymax < hi):
    #             ymax=hi
    #     # if ymin is not None and ymax is not None:
    #     if ymax is not None:
    #         ax1.set_ylim([0,ymax+0.1*(ymax)])
    x=np.array(support)
    if x.shape==():             # a number
        x=list(range(support))
    else:
        x=list(x)               # a list
    legend_names=[]
    for (line,sd),sty,name in zip(curves,style2,curve_names): # for every curve
            legend_names.append(name)
            if Z_names:                       # if we have more than one neighborhood size
                for z in range(len(Z_names)):
                    plot_line(x,line[z],sd[z],style1[z],sty,ax1,ylim,linewidth=2)
            else:
                plot_line(x,line,sd,style1[0],sty,ax1,ylim,linewidth=2)
    # Shrink current axis's height by 10% on the bottom
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.2,
                      box.width, box.height * 0.8])
    # Put a legend below current axis
    if num_cols==None:
        ncol=len(curves)
    else:
        ncol=num_cols
    # legend=plt.legend(artist2[:len(curves)],legend_names,loc='upper center', bbox_to_anchor=(0.5, -0.15),
                      # fancybox=True, shadow=True, ncol=ncol,fontsize=font_size)
    if Z_names and len(Z_names)>1:
        ax1.add_artist(plt.legend(artist1[:len(Z_names)],Z_names,loc='upper center', bbox_to_anchor=(0.5, -0.25),fancybox=True, shadow=True, ncol=len(Z_names),fontsize=font_size))
    if len(curves)>1:
        ax1.add_artist(plt.legend(artist2[:len(curves)],legend_names,loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True, ncol=ncol,fontsize=font_size))
    fig.savefig(filename,format='pdf')
    plt.close(fig)

def plot_styles(use_cmap=True,**kwargs):
    """
    Args:
    curves: a list of pairs containing a line to be plotted and its standard deviation
    """
    x = np.arange(len(kwargs["curves"]))
    ys = [i+x+(i*x)**2 for i in range(len(kwargs["curves"]))]
    if use_cmap:
        cx1 = plt.get_cmap('cubehelix_r')
        cols=[cx1(0.5)]
    else:
        cols = cm.rainbow(np.linspace(0, 1, len(ys)))
    lstyles=['-','--',':','-.']
    #Create custom artists
    colorArtists = [plt.Line2D((0,1),(0,0), color=c) for c in cols]
    artists=[plt.Line2D((0,1),(0,0), color='k', marker='', linestyle=sty) for sty in lstyles]

    plot(style1=cols,artist1=colorArtists,style2=lstyles,artist2=artists,**kwargs)

def plot_colors(use_cmap=True,**kwargs):
    """
    Args:
    curves: a list of pairs containing a line to be plotted and its standard deviation
    Inverts line styles and colors
    """
    x = np.arange(len(kwargs["curves"]))
    ys = [i+x+(i*x)**2 for i in range(len(kwargs["curves"]))]
    if use_cmap:
        cx1 = plt.get_cmap('cubehelix_r')
        cols=[cx1(float(i+1)/(len(ys)+1)) for i in range(len(ys))]
    else:
        cols = cm.rainbow(np.linspace(0, 1, len(ys)))
    #cols=['b','r','g','y','c','k']
    lstyles=['-','--',':','-.']
    #Create custom artists
    colorArtists = [plt.Line2D((0,1),(0,0), color=c) for c in cols]
    artists=[plt.Line2D((0,1),(0,0), color='k', marker='', linestyle=sty) for sty in lstyles]
    plot(style1=lstyles,artist1=artists,style2=cols,artist2=colorArtists,**kwargs)

def plot_histogram(values,curves,curve_names,tit="title",xlab="Algorithm",zlab="Value",ylab="Y",filename="out.pdf",font_size=16,use_cmap=True):
    vec_means=np.array([[v.mean() for v in v1] for v1,v2 in curves])
    vec_not_nan=[v for v in range(len(vec_means)) if not all(np.isnan(vec_means[v]))]
    vec_stds=np.array([[compute_combined_data_variance(v) for v in v2] for v1,v2 in curves])
#    vec_cis=np.array([[1.96*compute_combined_data_variance(v)/np.sqrt(len(v)) for v in v2] for v1,v2 in curves])
    ind = np.arange(len(vec_not_nan))  # the x locations for the groups
    width = 1/float(len(values)+1)       # the width of the bars
    if use_cmap:
        cx1 = plt.get_cmap('cubehelix_r')
        cols=[cx1(float(i+1)/(vec_means.shape[1]+1)) for i in range(vec_means.shape[1])]
    else:
        cols=['b','r','g','y','c','k']
    fig, ax = plt.subplots()
    fig.suptitle(tit,fontsize=font_size)
    plt.xlabel(xlab,fontsize=font_size)
    plt.ylabel(ylab,fontsize=font_size)
    ## compute ylim
    hi=np.nanmax(np.asarray(vec_means[vec_not_nan])+np.asarray(vec_stds[vec_not_nan]))
    print(str(hi))
    if hi is not None and not np.isnan(hi):
        ylim=int(hi+0.1*(hi))
        print(ylim)
        if ylim==hi:
            ylim+=1
        ax.set_ylim([0,ylim])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0,
                     box.width, box.height * 0.9])
    plt.subplots_adjust(bottom=0.2)
    rects=[ax.bar(ind+offset*width,means , width, yerr=stds,color=cols[offset],error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2)) for means,stds,offset in zip(vec_means[vec_not_nan].T,vec_stds[vec_not_nan].T,range(vec_means.shape[1]))]
    title = matplotlib.patches.Rectangle((0, 0), 0.1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    handles=[title]; handles.extend(rects)
    labels=[zlab]; labels.extend(values)
    ax.legend(handles, labels,loc='upper center', bbox_to_anchor=(0.5, 1.15),
                      fancybox=True, shadow=True, ncol=len(rects)+1,fontsize=font_size)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(np.array(curve_names)[vec_not_nan],rotation=-45,ha="center")
    ax.tick_params(labelsize=font_size)
    fig.savefig(filename,format='pdf')
    plt.close(fig)

def comm_outlier(obj,outlier,confidence):
    return outlier

def comm_outlier_or_margin(obj,outlier,confidence):
    return outlier or confidence==None or (abs(confidence)<=obj.thresh)

def comm_always(obj,outlier,confidence):
    return True

def faulkner11(num_sensors,end_time,num_rep,ns,p,**kwargs):
    """
    Each agent transmits binary message M_{s,t} to the remote. Messages are sent only if classification is positive.
    Each message contains the location.
    Agents do not talk to each other. Each agent has its own classifier.
    Training is done offline.
    """
    dic={"pretrain":True,"classify":True,"comm_rem":comm_outlier,
         "num_sensors":num_sensors,
         "end_time":end_time,
         "num_rep":num_rep,
         "ns":0, # agents do not communicate
         "p":p}
    dic.update(**kwargs)
    print(str(dic))
    return simulate_decen(**dic)

def faulkner13(num_sensors,end_time,num_rep,ns,p,**kwargs):
    """
    Each agent transmits binary message M_{s,t} to the remote. Messages are sent only if classification is positive.
    Unless we consider in-network aggregation, location is always transmitted
    There is only one classifier at the central controller.
    Training is done offline.
    """
    dic={"pretrain":True,"classify":True,"comm_rem":comm_outlier,
         "num_sensors":num_sensors,
         "end_time":end_time,
         "num_rep":num_rep,
         "ns":ns,
         "p":p}
    dic.update(**kwargs)
    print(str(dic))
    return simulate_cen(**dic)


def zhang09(num_sensors,end_time,num_rep,ns,p,**kwargs):
    """
    Each agent transmits binary message M_{s,t} to the remote. Messages are sent only if classification is positive.
    Each agent broadcasts the learned parameters to its neighbors when an outlier is detected, privacy is respected.
    Each agent has a classifier.
    There is no fusion center, but we assume someone will have to be notified when an event is happening.
    Training is done online.
    """
    dic={"learn":True,"classify":True,"pretrain":True,"calibrated":True, # privacy
         "comm":comm_outlier_or_margin, # communicate if is an outlier or support vector
         "comm_rem":comm_outlier,
         "num_sensors":num_sensors,
         "end_time":end_time,
         "num_rep":num_rep,
         "ns":ns,
         "p":p}
    dic.update(**kwargs)
    print(str(dic))
    return simulate_decen(**dic)


def zhang12(num_sensors,end_time,num_rep,ns,p,**kwargs):
    """
    All agents send their measurement to their neighbors if they classify it as outlier.
    There is no central controller but we assume they will have to notify someone.
    Training is done offline.
    """
    dic={"pretrain":True,"classify":True,"comm":comm_outlier, # communicate if is an outlier or support vector
         "num_sensors":num_sensors,
         "end_time":end_time,
         "num_rep":num_rep,
         "ns":ns,
         "p":p}
    dic.update(**kwargs)
    print(str(dic))
    return simulate_decen(**dic)


def ruan08(num_sensors,end_time,num_rep,ns,p,**kwargs):
    """
    Agents are trying to reach consensus, so they all send their classification at every timestep to each other.
    Since every sensor measures the same, privacy does not exist.
    Training is done online.
    """
    dic={"pretrain":True,"classify":True,"comm":comm_always,"comm_rem":comm_outlier,
         "num_sensors":num_sensors,
         "end_time":end_time,
         "num_rep":num_rep,
         "ns":1, # fully connected
         "p":p}
    dic.update(**kwargs)
    print(str(dic))
    return simulate_decen(**dic)


def bahrepour09(num_sensors,end_time,num_rep,ns,p,**kwargs):
    """
    The network is hierarchical: the first level reads from sensor and then communicates to the second level.
    Communication at the first level happens at every timestep. Communication at the secord level happens only if an outlier is classified.
    Agents do not communicate with neighbors.
    Training is done offline.
    """
    num_agents=list(np.asarray([10])*0.5)
    dic={"classify":True,"pretrain":True, # offline training
         "comm_rem":comm_outlier,         # send only outliers
         "num_agents":int(num_agents[0]), # TODO fix
         "num_sensors":num_sensors,
         "end_time":end_time,
         "num_rep":num_rep,
         "ns":0,  # no neighbors
         "p":p}
    dic.update(**kwargs)
    print(str(dic))
    return simulate_hier(**dic)

def bahrepour10(num_sensors,end_time,num_rep,ns,p,**kwargs):
    """
    Neighborhoods aggregate the individual judgements (at every timestep) and compute the reputation table.
    At every timestep the table containing all measurements and votes are sent to the fusion center which computes the consensus.
    Training is done offline.
    """
    dic={"classify":True,"pretrain":True,"comm":comm_always,"comm_rem":comm_always, # communicate always
         "num_sensors":num_sensors,
         "end_time":end_time,
         "num_rep":num_rep,
         "ns":ns,
         "p":p}
    dic.update(**kwargs)
    print(str(dic))
    return simulate_decen(**dic)


def wittenburg10(num_sensors,end_time,num_rep,ns,p,**kwargs):
    """
    Agents report only detections to remote, but exchange at every timestep a signature of their measurement.
    Training is done offline.
    """
    dic={"classify":True,"pretrain":True, # offline training
         "comm":comm_always,              # communicate at every timestep
         "comm_rem":comm_outlier,
         "num_sensors":num_sensors,
         "end_time":end_time,
         "num_rep":num_rep,
         "ns":ns,
         "p":p}
    dic.update(**kwargs)
    print(str(dic))
    return simulate_decen(**dic)


def zoumboulakis07(num_sensors,end_time,num_rep,ns,p,**kwargs):
    """
    Classical decentralized approach, each sensor works independently.
    We assume the detections are reported somewhere.
    Training is done offline.
    """
    dic={"classify":True,"pretrain":True,
         "num_sensors":num_sensors,
         "end_time":end_time,
         "num_rep":num_rep,
         "ns":0, # no neighbors
         "p":p}
    dic.update(**kwargs)
    print(str(dic))
    return simulate_decen(**dic)

def marin_perianu07(num_sensors,end_time,num_rep,ns,p,**kwargs):
    """
    Neighbors send their measurements at every timestep and measurement are then fuzzyfied, so there is no privacy.
    We assume the detections are reported somewhere.
    Training is done offline.
    Similar to wittenburg10.
    """
    dic={"classify":True,"pretrain":True,"comm":comm_always,"comm_rem":comm_outlier,
         "num_sensors":num_sensors,
         "end_time":end_time,
         "num_rep":num_rep,
         "ns":ns,
         "p":p}
    dic.update(**kwargs)
    print(str(dic))
    return simulate_decen(**dic)


def our_approach(num_sensors,end_time,num_rep,ns,p,**kwargs):
    dic={"learn":True,"classify":True,"calibrated":True,"supervised":True,"single":True,
         "num_sensors":num_sensors,
         "end_time":end_time,
         "num_rep":num_rep,
         "ns":ns,
         "p":p}
    dic.update(**kwargs)
    print(str(dic))
    return simulate_decen(**dic)


#run_centralized(4,5,1,p=0.5,classify=True,learn=True,debug=True)
#run_decentralized(4,5,1,p=0.5,ns=0.8,classify=True,learn=True,debug=True,calibrated=True)
#run_hierarchical(4,3,5,1,p=0.5,ns=0.8,classify=False,learn=False,debug=True,calibrated=True)

def compute_combined_data_mean(nx,ny,mx,my):
    """
    nx,y sample sizes
    mx,y means
    """
    return (nx*mx+ny*my)/float(nx+ny)

def compute_combined_data_variance(args):
    return np.sqrt((np.array(args)**2).sum()/len(args))

def compare_literature(out_dir,num_rep,end_time=100,values_n=default_values_n,values_ns=default_values_ns,p=0.5,**kwargs):
    """
    Compares the performance of different implementations in the literature
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # find the combinations of the functions to compare
    algs=[our_approach,zhang09,ruan08,zhang12,wittenburg10,marin_perianu07,bahrepour09,bahrepour10,zoumboulakis07,faulkner11,faulkner13]
    Z_names=["Proposed","ZMH09","RM08","ZHM12","WDWS10","MPH07","BMHH09","BMP10","ZR07","FOC11","FLK13"]
    vars_to_plot=[("cc",lab_cc,True),("cc_rem",lab_ccrem,True),("priv",lab_priv,None)]
    # vars_to_plot=[("prec",lab_prec,[0.0,1.1]),("rec",lab_rec,[0.0,1.1]),("f1","F1",[0.0,1.1]),("cost","Communication cost",None),("priv",lab_priv,None),("avg_cost","Average cost",[0.0,8.0]),("avg_cost_rem","Average cost superv.",[0.0,8.0]),("sr","Success ratio",[0.0,1.1]),("sr_rem","Success ratio superv.",[0.0,1.1]),("cc","No. of messages",True),("cc_rem",lab_ccrem,True)]
    data=np.zeros((len(algs),
                   len(values_ns),
                   len(var_names),
                   len(values_n),
                   end_time))
    data_sd=np.zeros((len(algs),
                      len(values_ns),
                      len(var_names),
                      len(values_n),
                      end_time))
    data_ci=np.zeros((len(algs),
                      len(values_ns),
                      len(var_names),
                      len(values_n),
                      end_time))

    out_name="comp_lit_p"+str(p)
    if not os.path.isfile(os.path.join(out_dir,"obj",out_name+".pkl")):
        for i in range(len(values_ns)):
            ns=values_ns[i]
            # performance does not depend on communication function if classification is disabled
            for z in range(len(algs)):
                fct=algs[z]
                print("running "+str(Z_names[z]))
                data[z,i,:,:,:],data_sd[z,i,:,:,:],data_ci[z,i,:,:,:]=fct(values_n,end_time,num_rep,ns,p,**kwargs)

        print("done")
        if not os.path.exists(os.path.join(out_dir,"obj")):
            os.makedirs(os.path.join(out_dir,"obj"))
        save_obj(out_dir,data,out_name);save_obj(out_dir,data_sd,out_name+"_sd");save_obj(out_dir,data_ci,out_name+"_ci")
    else:
        print("Reading saved data");data=load_obj(out_dir,out_name);data_sd=load_obj(out_dir,out_name+"_sd");data_ci=load_obj(out_dir,out_name+"_ci")

    ## TODO
    algs=algs[1:]
    Z_names=Z_names[1:]
    for i in range(len(values_ns)):
        ns=values_ns[i]
        for s,lab,ylim in vars_to_plot:
            j=return_dictionary_idx(s)
            for k in range(len(values_n)):
                n=values_n[k]
                vec=[[data[z,i,j,k,:],data_ci[z,i,j,k,:]] for z in range(len(algs))]
                plot_colors(support=end_time,curves=vec,curve_names=Z_names,xlab=lab_time,tit="",ylab=lab,filename=out_dir+"/"+s+"_n"+str(n)+"_ns"+str(ns)+"_p"+str(p)+"_compare_lit.pdf",ylim=ylim,font_size=16)
            # ## histogram
            vec=[[data[z,i,j,:,:],
                  data_ci[z,i,j,:,:]] for z in range(len(algs))]
            plot_histogram(values_n,vec,Z_names,zlab="Population size:",xlab=lab_alg,tit="",ylab=lab,filename=os.path.join(out_dir,s+"_ns"+str(ns)+"_p"+str(p)+"_hist_compare_lit.pdf"))

            if ylim==None:      # cost and priv: the variable is cumulated and it remains constant during the comparison, plot only the first value
                vec=[[data[z,i,j,:,0], # plot the first value of the simulation
                      data_ci[z,i,j,:,0]] for z in range(len(algs))]
                plot_colors(support=values_n,curves=vec,curve_names=Z_names,xlab=lab_n,tit="",ylab=lab,filename=os.path.join(out_dir,s+"_ns"+str(ns)+"_p"+str(p)+"_compare_lit.pdf"),ylim=True) # ylim=True for non-cumulative variables

    # some algorithms are independent of the neighborhood size
    algs_dependent=["ZMH09","ZHM12","WDWS10","MPH07","BMP10"]
    idx_algs_dependent=[Z_names.index(a) for a in algs_dependent]
    for i in range(len(values_n)):
        n=values_n[i]
        for s,lab,ylim in vars_to_plot:
            j=return_dictionary_idx(s)
            if s=="cc":         # do not plot ns=0 if variable is communication towards neighbors
                ns_mask=[z for z in range(len(values_ns)) if values_ns[z]!=0.0]
            else:
                ns_mask=range(len(values_ns))
            ## histogram
            vec=[[data[z,ns_mask,j,i,:],
                  data_ci[z,ns_mask,j,i,:]] for z in range(len(algs))]
            plot_histogram(np.asarray(values_ns)[ns_mask],vec,Z_names,zlab="Neighborhood size:",xlab=lab_alg,tit="",ylab=lab,filename=os.path.join(out_dir,s+"_n"+str(n)+"_p"+str(p)+"_hist_compare_lit.pdf"))
            if ylim==None:      # cost and priv: the variable is cumulated and it remains constant during the comparison, plot only the first value
                vec=[[data[z,:,j,i,0], # plot the first value of the simulation
                      data_ci[z,:,j,i,0]] for z in idx_algs_dependent]
                plot_colors(support=values_ns,curves=vec,curve_names=Z_names,xlab=lab_ns,tit="",ylab=lab,filename=os.path.join(out_dir,s+"_n"+str(n)+"_p"+str(p)+"_compare_lit.pdf"),ylim=True) # ylim=True for non-cumulative variables

def compare_communication_fcts(out_dir,num_rep,end_time=100,values_n=default_values_n,values_ns=default_values_ns,p=0.5,**kwargs):
    """
    Compares the performance of different communication parameters:

    Towards agents:
    - default: communicate if confidence is below a threshold
    - outlier: communicate if classified as outlier
    - always

    Towards remote:
    - default: communicate if classified as outlier
    - always
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    vars_to_plot=[("priv",lab_priv,None),("prec",lab_prec,[0.0,1.1]),("f1",lab_f1,[0.0,1.1])] #[("cc",lab_cc,True),("cc_rem",lab_ccrem,True),("prec",lab_prec,[0.0,1.1]),("rec",lab_rec,[0.0,1.1]),("f1","F1",[0.0,1.1]),("cost","Communication cost",None),("priv",lab_priv,None)]

    # find the combinations of the functions to compare
    # comms=list(itertools.product([comm_outlier,comm_always], # options for agent
    #                             [None,comm_always])) # options for remote
    comms=list(itertools.product([None,comm_outlier,comm_always], # options for agent
                                [None,comm_always])) # options for remote
    Z_names=[(("c" if c==None else str(c.__name__)[5]),("o" if r==None else str(r.__name__)[5])) for c,r in comms]
    data=np.zeros((len(comms)+1,
                   len(values_ns),
                   len(var_names),
                   len(values_n),
                   end_time))
    data_sd=np.zeros((len(comms)+1,
                      len(values_ns),
                      len(var_names),
                      len(values_n),
                      end_time))
    data_ci=np.zeros((len(comms)+1,
                      len(values_ns),
                      len(var_names),
                      len(values_n),
                      end_time))

    out_name="comp_fct_p"+str(p)
    if not os.path.isfile(os.path.join(out_dir,"obj",out_name+".pkl")):
        for i in range(len(values_ns)):
            ns=values_ns[i]
            # performance does not depend on communication function if classification is disabled
            for z in range(len(comms)):
                (comm,comm_rem)=comms[z]
                data[z,i,:,:,:],data_sd[z,i,:,:,:],data_ci[z,i,:,:,:]=simulate_decen(values_n,end_time,num_rep,ns=ns,p=p,classify=True,comm=comm,comm_rem=comm_rem,**kwargs)
            data[len(comms),i,:,:,:],data_sd[len(comms),i,:,:,:],data_ci[len(comms),i,:,:,:]=simulate_decen(values_n,end_time,num_rep=1,ns=ns,p=p,classify=False,**kwargs)

        print("done")
        if not os.path.exists(os.path.join(out_dir,"obj")):
            os.makedirs(os.path.join(out_dir,"obj"))
        save_obj(out_dir,data,out_name);save_obj(out_dir,data_sd,out_name+"_sd");save_obj(out_dir,data_ci,out_name+"_ci")
    else:
        print("Reading saved data");data=load_obj(out_dir,out_name);data_sd=load_obj(out_dir,out_name+"_sd");data_ci=load_obj(out_dir,out_name+"_ci")

    comms_to_plot=[[(None,None),"c,o"],[(comm_outlier,None),"o,o"],[(comm_always,None),"a,o"]]
    ids_to_plot=[comms.index(i) for i,j in comms_to_plot]
    labs_to_plot=[j for i,j in comms_to_plot]
    for i in range(len(values_ns)):
        ns=values_ns[i]
        for k in range(len(values_n)):
            n=values_n[k]
            for s,lab,ylim in vars_to_plot:
                j=return_dictionary_idx(s)
                vec=[[data[z,i,j,k,:],data_ci[z,i,j,k,:]] for z in ids_to_plot+[len(comms)]]
                plot_colors(support=end_time,curves=vec,curve_names=labs_to_plot+["No class"],tit="",ylab=lab,filename=out_dir+"/"+s+"_ns"+str(ns)+"_n"+str(n)+"_p"+str(p)+"_compare_comm_fct.pdf",ylim=ylim,num_cols=int((len(comms_to_plot)+1)/2+(len(comms_to_plot)+1)%2))

def compare_pretraining(out_dir,num_rep,end_time=100,values_n=default_values_n,const_ns=default_const_ns,const_p=default_const_p,**kwargs):
    """
    Compares the performance of different pretraining values
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # find the combinations of the functions to compare
    Z=[0,0.2,0.5,0.8,1]
    Z_names=[str(z) for z in Z]
    vars_to_plot=[("prec",[0.0,1.1]),("rec",[0.0,1.1]),("f1",[0.0,1.1]),("cost",None),("priv",None)]
    data=np.zeros((len(Z),
                   len(var_names),
                   len(values_n),
                   end_time))
    data_sd=np.zeros((len(Z),
                      len(var_names),
                      len(values_n),
                      end_time))
    data_ci=np.zeros((len(Z),
                      len(var_names),
                      len(values_n),
                      end_time))
    out_name="comp_pretrain_ns"+str(const_ns)+"_p"+str(const_p)
    if not os.path.isfile(os.path.join(out_dir,"obj",out_name+".pkl")):
        for i in range(len(Z)):
            z=Z[i]
            data[i,:,:,:],data_sd[i,:,:,:],data_ci[i,:,:,:]=simulate_decen(values_n,end_time,num_rep,ns=const_ns,p=const_p,classify=True,pretrain=z,**kwargs)

        print("done")

        if not os.path.exists(os.path.join(out_dir,"obj")):
            os.makedirs(os.path.join(out_dir,"obj"))
        save_obj(out_dir,data,out_name);save_obj(out_dir,data_sd,out_name+"_sd");save_obj(out_dir,data_ci,out_name+"_ci")
    else:
        print("Reading saved data");data=load_obj(out_dir,out_name);data_sd=load_obj(out_dir,out_name+"_sd");data_ci=load_obj(out_dir,out_name+"_ci")

    for i in range(len(values_n)):
        n=values_n[i]
        for s,ylim in vars_to_plot:
            j=return_dictionary_idx(s)
            vec=[[data[z,j,i,:],data_ci[z,j,i,:]] for z in range(len(Z))]
            plot_colors(support=end_time,curves=vec,curve_names=Z_names,tit=s+" (agents: "+str(n)+" neigh_size: "+str(const_ns)+")",ylab=s,filename=out_dir+"/"+s+"_pretrain_ns"+str(const_ns)+"_n"+str(n)+"_p"+str(const_p)+".pdf",ylim=ylim)

def plot_avg_transmission_cost(out_dir,num_rep,end_time=250,values_n=default_values_n,const_ns=default_const_ns,const_p=default_const_p,**kwargs):
    """
    Plots how the average transmission cost varies over time.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # performance does not depend on communication function if classification is disabled
    treatments=[("",{"classify":True}),
                ("_learn",{"learn":True,"classify":True})] # learning
    data=np.zeros((len(treatments),
                   len(var_names),
                   len(values_n),
                   end_time))
    data_sd=np.zeros((len(treatments),
                      len(var_names),
                      len(values_n),
                      end_time))
    data_ci=np.zeros((len(treatments),
                      len(var_names),
                      len(values_n),
                      end_time))

    out_name="avg_tr_cost_ns"+str(const_ns)+"_p"+str(const_p)
    if not os.path.isfile(os.path.join(out_dir,"obj",out_name+".pkl")):
        for t in range(len(treatments)):
            params=treatments[t][1]
            params.update(kwargs)
            data[t,:,:,:],data_sd[t,:,:,:],data_ci[t,:,:,:]=simulate_decen(values_n,end_time,num_rep,ns=const_ns,p=const_p,**params)

        print("done")

        if not os.path.exists(os.path.join(out_dir,"obj")):
            os.makedirs(os.path.join(out_dir,"obj"))
        save_obj(out_dir,data,out_name);save_obj(out_dir,data_sd,out_name+"_sd");save_obj(out_dir,data_ci,out_name+"_ci")
    else:
        print("Reading saved data");data=load_obj(out_dir,out_name);data_sd=load_obj(out_dir,out_name+"_sd");data_ci=load_obj(out_dir,out_name+"_ci")

    for j in range(len(values_n)):
        i=var_names.index("avg_cost")
        vec=[[data[t,i,j,:],data_ci[t,i,j,:]] for t in range(len(treatments))]
        i=var_names.index("avg_cost_rem")
        vec.extend([[data[t,i,j,:],data_ci[t,i,j,:]] for t in range(len(treatments))])
        plot_colors(support=end_time,curves=vec,curve_names=["Agents","Agents learn","Remote","Remote learn"],tit="Average transmission cost (decentralized, "+str(values_n[j])+" agents, neigh size "+str(const_ns)+")",ylab=lab_cost,filename=out_dir+"/avg_tr_cost_n"+str(values_n[j])+"_p"+str(const_p)+"_ns"+str(const_ns)+".pdf",ylim=[0,8])

def compare_params(fct,desc,out_dir,num_rep,end_time,var1_id,vars1,var2_id,vars2,const_id,const_var,**kwargs):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    ### Compare probability and number of sensors
    # every element contains a list with all variables
    treatments=[("",{"classify":False}),
                ("_class",{"classify":True}), # classification
                ("_learn",{"learn":True,"classify":True}), # learning
                ("_learn_calib",{"learn":True,"classify":True,"calibrated":True})] # learning and calibrated
    vars_to_plot=[("cc",True),("cc_rem",True),("prec",[0.0,1.1]),("rec",[0.0,1.1]),("f1",[0.0,1.1]),("cost",None),("priv",None),("avg_cost",[0.0,8.0]),("avg_cost_rem",[0.0,8.0]),("sr",[0.0,1.1]),("sr_rem",[0.0,1.1])]
    data=np.zeros((len(treatments),
                   len(vars1),
                   len(var_names),
                   len(vars2),
                   end_time))
    data_sd=np.zeros((len(treatments),
                      len(vars1),
                      len(var_names),
                      len(vars2),
                      end_time))
    data_ci=np.zeros((len(treatments),
                      len(vars1),
                      len(var_names),
                      len(vars2),
                      end_time))
    #------------------------------
    znames=[var2_id+str(x) for x in vars2]
    # populate the vectors
    out_name="data_"+var1_id+"X"+"_"+const_id+str(const_var)
    if not os.path.isfile(os.path.join(out_dir,"obj",out_name+".pkl")):
        for i in range(len(vars1)):
            v=vars1[i]
            extra_args=kwargs; extra_args.update({var1_id:v,const_id:const_var}) # pass the correct values for the parameters by adding them to the keyword arguments.
            for t in range(len(treatments)):
                params=treatments[t][1]
                params.update(extra_args)
                data[t,i,:,:,:],data_sd[t,i,:,:,:],data_ci[t,i,:,:,:]=fct(vars2,end_time,num_rep,**params)

        print("Done")
        if not os.path.exists(os.path.join(out_dir,"obj")):
            os.makedirs(os.path.join(out_dir,"obj"))
        save_obj(out_dir,data,out_name);save_obj(out_dir,data_sd,out_name+"_sd");save_obj(out_dir,data_ci,out_name+"_ci")
    else:
        print("Reading saved data");data=load_obj(out_dir,out_name);data_sd=load_obj(out_dir,out_name+"_sd");data_ci=load_obj(out_dir,out_name+"_ci")

    for i in range(len(vars1)):
        v=vars1[i]
        ## Vary the value of var2 for a given value of var1. Plot the 3 logics with line style and the value of var2 with colors. One plot for every value of var1 and const_var
        for what_to_plot,name in [[[0,1,2],""],        # non calibrated
                                  [[0,1,3],"_calib"]]: # calibrated
            for s,ylim in vars_to_plot:
                j=return_dictionary_idx(s)
                vec=[[data[t,i,j,:,:],data_ci[t,i,j,:,:]] for t in what_to_plot]
                labs=[desc+treatments[l][0] for l in what_to_plot]
                plot_styles(support=end_time,curves=vec,curve_names=labs,tit=s+" ("+desc+")",ylab=s,filename=out_dir+"/"+s+name+"_"+var1_id+str(v)+"_"+const_id+str(const_var)+".pdf",Z_names=znames,ylim=ylim)

    print("Done")

    znames=[var1_id+str(x) for x in vars1]

    for i in range(len(vars2)):
        n=vars2[i]
        for what_to_plot,name in [[[0,1,2],""],        # non calibrated
                                  [[0,1,3],"_calib"]]: # calibrated
            for s,ylim in vars_to_plot:
                j=return_dictionary_idx(s)
                vec=[[data[t,:,j,i,:],data_ci[t,:,j,i,:]] for t in what_to_plot]
                labs=[desc+treatments[l][0] for l in what_to_plot]
                plot_styles(support=end_time,curves=vec,curve_names=labs,tit=s+" ("+desc+")",ylab=s,filename=out_dir+"/"+s+name+"_"+var2_id+str(n)+"_"+const_id+str(const_var)+".pdf",Z_names=znames,ylim=ylim)

def compare_centr(out_dir,num_rep,end_time=100,values_n=default_values_n,values_p=default_values_p,**kwargs):
    compare_params(simulate_cen,"Cen",out_dir,num_rep,end_time,"p",values_p,"n",values_n,"ns",0,**kwargs) # centralized has no neighbors

def compare_decentr_prob(out_dir,num_rep,end_time=100,values_n=default_values_n,values_ns=default_values_ns,const_p=default_const_p,**kwargs):
    compare_params(simulate_decen,"Dec",out_dir,num_rep,end_time,"ns",values_ns,"n",values_n,"p",const_p,**kwargs)

def compare_decentr_prob2(out_dir,num_rep,end_time=100,values_n=default_values_n,values_ns=default_values_ns,const_p=default_const_p,**kwargs):
    compare_params2(simulate_decen,"Dec",out_dir,num_rep,end_time,"ns",values_ns,"n",values_n,"p",const_p,**kwargs)

def compare_decentr_nevents(out_dir,num_rep,end_time=100,values_n=default_values_n,values_ns=default_values_ns,const_p=default_const_nevents,**kwargs):
    compare_params(simulate_decen,"Dec",out_dir,num_rep,end_time,"ns",values_ns,"n",values_n,"p",const_p,**kwargs)

def compare_decentr_ns(out_dir,num_rep,end_time=100,values_n=default_values_n,values_p=default_values_p,const_ns=default_const_ns,**kwargs):
    compare_params(simulate_decen,"Dec",out_dir,num_rep,end_time,"p",values_p,"n",values_n,"ns",const_ns,**kwargs)

def param_exploration(out_dir,num_rep,end_time,values_n,values_ns,p,**kwargs):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # treatments=[("Classif. disabled",{"classify":False}),
    #             ("Classif. enabled",{"classify":True}), # classification
    #             # ("_learn",{"learn":True,"classify":True}), # learning
    #             ("Learning calibrated",{"learn":True,"classify":True,"calibrated":True})] # learning and calibrated
    treatments=[default_treatments[1]]+[default_treatments[3]]
    vars_to_plot=[("cost",True),("priv",True),("cc",True),("cc_rem",True),("f1",[0.0,1.1])]#,("avg_cost",[0.0,8.0]),("avg_cost_rem",[0.0,8.0]),("sr",[0.0,1.1]),("sr_rem",[0.0,1.1])]
    organiz=["Centralized","Decentralized"]#,"Hierarchical"]
    data=np.zeros((len(treatments),
                   len(values_ns),
                   len(organiz),           # 0=cen, 1=dec
                   len(var_names),
                   len(values_n)))
    data_sd=np.zeros((len(treatments),
                      len(values_ns),
                      len(organiz),        # 0=cen, 1=dec
                      len(var_names),
                      len(values_n)))
    data_ci=np.zeros((len(treatments),
                      len(values_ns),
                      len(organiz),        # 0=cen, 1=dec
                      len(var_names),
                      len(values_n)))
    out_name="param_expl_data_p"+str(p)
    print(os.path.join(out_dir,"obj",out_name+".pkl"))
    if not os.path.isfile(os.path.join(out_dir,"obj",out_name+".pkl")):
        for i in range(len(values_ns)):
            v=values_ns[i]
            for t in range(len(treatments)):
                params=treatments[t][1]
                params.update(kwargs)
                d,sd,ci=simulate_cen(values_n,end_time,num_rep,p=p,**params)
                data[t,i,0,:,:]=np.asarray(d)[:,:,end_time-1] # take last value
                data_sd[t,i,0,:,:]=np.asarray(sd)[:,:,end_time-1]
                data_ci[t,i,0,:,:]=np.asarray(ci)[:,:,end_time-1]
                d,sd,ci=simulate_decen(values_n,end_time,num_rep,ns=v,p=p,**params)
                data[t,i,1,:,:]=np.asarray(d)[:,:,end_time-1] # take last value
                data_sd[t,i,1,:,:]=np.asarray(sd)[:,:,end_time-1]
                data_ci[t,i,1,:,:]=np.asarray(ci)[:,:,end_time-1]
                # d,sd=simulate_hier(values_n,0.5,end_time,num_rep,ns=v,p=p,**params)
                # data[t,i,2,:,:]=np.asarray(d)[:,:,end_time-1] # take last value
                # data_sd[t,i,2,:,:]=np.asarray(sd)[:,:,end_time-1]
        print("done")
        if not os.path.exists(os.path.join(out_dir,"obj")):
            os.makedirs(os.path.join(out_dir,"obj"))
        save_obj(out_dir,data,out_name);save_obj(out_dir,data_sd,out_name+"_sd");save_obj(out_dir,data_ci,out_name+"_ci")
    else:
        print("Reading saved data");data=load_obj(out_dir,out_name);data_sd=load_obj(out_dir,out_name+"_sd");data_ci=load_obj(out_dir,out_name+"_ci")

    for i in range(len(values_ns)):
        v=values_ns[i]
        for s,ylim in vars_to_plot:
            j=return_dictionary_idx(s)
            vec=[[[data[t,i,o,j,:] for o in range(len(organiz))],
                  [data_ci[t,i,o,j,:] for o in range(len(organiz))]] for t in range(len(treatments))]
            labs=[t[0] for t in treatments]
            plot_colors(support=values_n,curves=vec,curve_names=labs,tit="Parameter exploration, "+s+" (ns:"+str(v)+" p:"+str(p)+")",xlab=lab_n,ylab=s,filename=out_dir+"/"+s+"_ns"+str(v)+"_p"+str(p)+".pdf",ylim=ylim,Z_names=organiz)

    for i in range(len(values_n)):
        v=values_n[i]
        for s,ylim in vars_to_plot:
            j=return_dictionary_idx(s)
            vec=[[[data[t,:,o,j,i] for o in range(len(organiz))],
                  [data_ci[t,:,o,j,i] for o in range(len(organiz))]] for t in range(len(treatments))]
            labs=[t[0] for t in treatments]
            plot_colors(support=values_ns,curves=vec,curve_names=labs,tit="Parameter exploration, "+s+" (n:"+str(v)+" p:"+str(p)+")",xlab=lab_ns,ylab=s,filename=out_dir+"/"+s+"_n"+str(v)+"_p"+str(p)+".pdf",ylim=ylim,Z_names=organiz)

def generate_figures():
    matplotlib.style.use('classic')
    ### parame exploration
    p=0.2
    ## parameters used to generate the data
    organiz=["Centralized","Distributed"]#,"Hierarchical"]
    values_ns=[0.0,0.2,0.5]
    values_n=[10,50]
    out_dir="../results/compare_lit"
    compare_literature(out_dir,0,end_time=200,values_n=values_n,values_ns=values_ns,p=p)
    values_ns=[0.0,0.2,0.3,0.4,0.5]
    values_n=[10,20,30,50]
    out_dir="../results/compare_lit_learn"
    compare_literature(out_dir,0,end_time=200,values_n=values_n,values_ns=values_ns,p=p)
    # show effect of learning in literature
    data_nolearn=load_obj("../results/compare_lit","comp_lit_p"+str(p));data_nolearn_ci=load_obj("../results/compare_lit","comp_lit_p"+str(p)+"_ci")
    data_learn=load_obj("../results/compare_lit_learn","comp_lit_p"+str(p));data_learn_ci=load_obj("../results/compare_lit_learn","comp_lit_p"+str(p)+"_ci")
    Z_names=["Proposed","ZMH09","RM08","ZHM12","WDWS10","MPH07","BMHH09","BMP10","ZR07","FOC11","FLK13"]
    vars_to_plot=[("priv",lab_priv,None)]
    # some algorithms are independent of the neighborhood size
    algs_use=["ZMH09","RM08","ZHM12","WDWS10","MPH07","BMHH09","BMP10","ZR07","FOC11","FLK13"]
    idx_algs_use=[Z_names.index(a) for a in algs_use]
    values_n_use=[10]
    idx_values_n_use=[values_n.index(a) for a in values_n_use]
    values_ns_use=[0.2]
    # idx_values_ns_use=[values_ns.index(a) for a in values_ns_use]
    for ns in values_ns_use:
        for i in idx_values_n_use:
            n=values_n[i]
            for s,lab,ylim in vars_to_plot:
                j=return_dictionary_idx(s)
                w=values_ns.index(ns)
                ## histogram
                vec=[[np.concatenate((data_learn[z,[w],j,i,:],data_nolearn[z,[w],j,i,:])),
                      np.concatenate((data_learn_ci[z,[w],j,i,:],data_nolearn_ci[z,[w],j,i,:]))] for z in idx_algs_use]
                plot_histogram(np.asarray(["Enabled","Disabled"]),vec,algs_use,zlab="Learning:",xlab=lab_alg,tit="",ylab=lab,filename=os.path.join(out_dir,"figure_"+s+"_n"+str(n)+"_ns"+str(ns)+"_p"+str(p)+"_hist_compare_lit.pdf"))

            # --------------------

    ## calibration
    out_dir="../results/plots_dec"
    values_ns=[0.0,0.2,0.5]
    values_n=[10,50]
    data=load_obj(out_dir,"data_nsX_p0.2"); data_sd=load_obj(out_dir,"data_sd_nsX_p0.2")
    for s,ylim,treats,ylab in [["priv",None,[1,2],lab_priv],["f1",True,[1,2],lab_f1]]:
        i=values_ns.index(0.2)
        k=values_n.index(50)
        j=return_dictionary_idx(s)
        vec=[[data[t,k,j,i,:],
              data_sd[t,k,j,i,:]] for t in treats]
        plot_styles(support=200,curves=vec,curve_names=[default_treatments[t][0] for t in treats],tit="",xlab="Time",ylab=ylab,filename=out_dir+"/figure_calib_"+s+".pdf",ylim=ylim)

    ## communication functions comparison
    out_dir="../results/compare_fct_single_prob_supervised_learn"
    compare_communication_fcts(out_dir,0,200,values_n,values_ns,p)
    # priv_ns0.2_p0.2
    # treatments=[("Classif. disabled",{"classify":False}),
    #             ("Classif. enabled",{"classify":True}), # classification
    #             # ("_learn",{"learn":True,"classify":True}), # learning
    #             ("Learning calibrated",{"learn":True,"classify":True,"calibrated":True})] # learning and calibrated
    treatments=[default_treatments[1]]+[default_treatments[3]]
    out_dir="../results/param_expl"
    out_name="param_expl_data_p"+str(p)
    values_ns=[0.0,0.2,0.3,0.4,0.5]
    values_n=[10,20,30,50]
    data=load_obj(out_dir,out_name);data_sd=load_obj(out_dir,out_name+"_sd");data_ci=load_obj(out_dir,out_name+"_ci")

    def plot_param_expl_ns(var,ylab,i,treatments_idx,organiz_idx,ylim_offset,ylim_step):
        ## over neighborhood size
        v=values_ns[i]
        j=return_dictionary_idx(var)
        vec=[[[data[t,i,o,j,:] for o in organiz_idx],
              [data_ci[t,i,o,j,:] for o in organiz_idx]] for t in treatments_idx]
        if ylim_offset is not False and ylim_step is not False:
            ylim=(0,np.nanmax(vec)+ylim_offset-(np.nanmax(vec)+ylim_offset)%ylim_step)
        else:
            ylim=True
        labs=[treatments[t][0] for t in treatments_idx]
        plot_colors(support=values_n,curves=vec,curve_names=labs,xlab=lab_n,ylab=ylab,filename=out_dir+"/figure_"+var+"_ns"+str(v)+"_p"+str(p)+".pdf",ylim=ylim,Z_names=[organiz[i] for i in organiz_idx])#tit="Parameter exploration, "+var+" (n:"+str(v)+" p:"+str(p)+")",

    def plot_param_expl_n(var,ylab,i,treatments_idx,organiz_idx,ylim_offset,ylim_step):
        ## over neighborhood size
        v=values_n[i]
        j=return_dictionary_idx(var)
        vec=[[[data[t,:,o,j,i] for o in organiz_idx],
              [data_ci[t,:,o,j,i] for o in organiz_idx]] for t in treatments_idx]
        if ylim_offset is not False and ylim_step is not False:
            ylim=(0,np.nanmax(vec)+ylim_offset-(np.nanmax(vec)+ylim_offset)%ylim_step)
        else:
            ylim=True
        labs=[treatments[t][0] for t in treatments_idx]
        plot_colors(support=values_ns,curves=vec,curve_names=labs,xlab=lab_ns,ylab=ylab,filename=out_dir+"/figure_"+var+"_n"+str(v)+"_p"+str(p)+".pdf",ylim=ylim,Z_names=[organiz[i] for i in organiz_idx])#tit="Parameter exploration, "+var+" (n:"+str(v)+" p:"+str(p)+")",

    ## second image
    ## over ns
    plot_param_expl_n("cc_rem","Communication to the supervisor",3,[0],[0,1],5,5)
    ## over pop size
    plot_param_expl_ns("cc_rem","Communication to the supervisor",1,[0],[0,1],5,5)
    ## third image: comm frequency over neighborhood size
    plot_param_expl_n("cc","Communication to other agents",3,[0],[1],50,50)
    ## fourth image: comm frequency over population size
    plot_param_expl_ns("cc","Communication to other agents",1,[0],[1],50,50)
    ## fifth image: privacy over neighborhood size
    plot_param_expl_n("priv",lab_priv,3,[0,1],[0,1],2000,5000)
    ## sixth image: privacy over population size
    plot_param_expl_ns("priv",lab_priv,2,[0,1],[0,1],5000,5000)
    ## seventh image: F1 over neighborhood size
    plot_param_expl_n("f1",lab_f1,3,[1],[0,1],False,False)
    plot_param_expl_ns("f1",lab_f1,2,[1],[0,1],False,False)
    plot_param_expl_n("prec",lab_prec,3,[1],[0,1],1,1)
    plot_param_expl_ns("prec",lab_prec,2,[1],[0,1],1,1)
