#!/bin/bash

host=$(hostname | awk '{print substr($0,0,5)}')
COMMAND="python3"
if [ "$host" = "euler" ]; then
    JOB="bsub -n 1"
    SHORT="-W 120:00"
    LONG="-W 500:00"
else
  JOB=""
  SHORT=""
  LONG=""
fi

function join_by { local d=$1; shift; echo -n "$1"; shift; printf "%s" "${@/#/$d}"; }

numrep=10
values_n_array=(10,20,30,50)
values_n="[$(join_by , ${values_n_array[@]})]"
values_ns_array=(0.0,0.2,0.3,0.4,0.5)
values_ns="[$(join_by , ${values_ns_array[@]})]"
values_p_array=(0.2)
values_p="[$(join_by , ${values_p_array[@]})]"
values_nevents_array=(300 600)
values_nevents="[$(join_by , ${values_nevents_array[@]})]"
const_n=${values_n_array[1]}
const_ns=${values_ns_array[0]}
const_p=${values_p_array[0]}
const_nevents=${values_nevents_array[0]}
end_time=200

for p in ${values_p_array[@]}; do
#for p in ${values_nevents_array[@]}; do
  params="$numrep,p=$p,end_time=$end_time,values_n=$values_n,values_ns=$values_ns"
  echo "fcts with p $p"
  $JOB $LONG $COMMAND -c "import main; main.compare_communication_fcts(\"./compare_fct_prob\",$params)"
  echo "fcts pretrain with p $p"
  $JOB $LONG $COMMAND -c "import main; main.compare_communication_fcts(\"./compare_fct_prob_pretrain\",$params,pretrain=True)"
  echo "fcts supervised with p $p"
  $JOB $LONG $COMMAND -c "import main; main.compare_communication_fcts(\"./compare_fct_prob_supervised\",$params,supervised=True)"
  echo "fcts learn with p $p"
  $JOB $LONG $COMMAND -c "import main; main.compare_communication_fcts(\"./compare_fct_prob_learn\",$params,learn_from_neighbors=True)"
  echo "fcts pretrain and learn with p $p"
  $JOB $LONG $COMMAND -c "import main; main.compare_communication_fcts(\"./compare_fct_prob_pretrain_learn\",$params,pretrain=True,learn_from_neighbors=True)"
  echo "fcts supervised and learn with p $p"
  $JOB $LONG $COMMAND -c "import main; main.compare_communication_fcts(\"./compare_fct_prob_supervised_learn\",$params,supervised=True,learn_from_neighbors=True)"
  echo "fcts single learn with p $p"
  $JOB $LONG $COMMAND -c "import main; main.compare_communication_fcts(\"./compare_fct_single_prob_learn\",$params,learn_from_neighbors=True,single=True)"
  echo "fcts single pretrain and learn with p $p"
  $JOB $LONG $COMMAND -c "import main; main.compare_communication_fcts(\"./compare_fct_single_prob_pretrain_learn\",$params,pretrain=True,learn_from_neighbors=True,single=True)"
  echo "fcts single supervised and learn with p $p"
  $JOB $LONG $COMMAND -c "import main; main.compare_communication_fcts(\"./compare_fct_single_prob_supervised_learn\",$params,supervised=True,learn_from_neighbors=True,single=True)"
done
for p in ${values_p_array[@]}; do
  echo "compare lit with p $p"
  $JOB $SHORT $COMMAND -c "import main; main.compare_literature(\"./compare_lit\",num_rep=$numrep,p=$p,end_time=$end_time,values_n=$values_n,values_ns=$values_ns)"
  echo "compare lit with learning enabled and with p $p"
  $JOB $SHORT $COMMAND -c "import main; main.compare_literature(\"./compare_lit_learn\",num_rep=$numrep,p=$p,end_time=$end_time,values_n=$values_n,values_ns=$values_ns,learn=True,calibrated=True)"
  echo "param exploration with p $p"
  $JOB $SHORT $COMMAND -c "import main; main.param_exploration(\"./param_expl\",num_rep=$numrep,p=$p,end_time=$end_time,values_n=$values_n,values_ns=$values_ns)"
done

for ns in ${values_ns_array[@]}; do
  #params="$numrep,end_time=$end_time,values_n=$values_n,const_ns=$ns,const_p=$const_p"
  params="$numrep,end_time=$end_time,values_n=$values_n,const_ns=$ns,const_p=$const_nevents"
  echo "pretraining"
  $JOB $SHORT $COMMAND -c "import main; main.compare_pretraining(\"./compare_pretrain\",$params)"
  echo "pretraining learn"
  $JOB $SHORT $COMMAND -c "import main; main.compare_pretraining(\"./compare_pretrain_learn\",$params,learn_from_neighbors=True)"
  echo "pretraining learn single"
  $JOB $SHORT $COMMAND -c "import main; main.compare_pretraining(\"./compare_pretrain_learn_single\",$params,single=True,learn_from_neighbors=True)"
done

#params="$numrep,end_time=250,values_n=$values_n,const_ns=$const_ns,const_p=$const_p"
params="$numrep,end_time=250,values_n=$values_n,const_ns=$const_ns,const_p=$const_nevents"
echo "avg transmission cost"
$JOB $SHORT $COMMAND -c "import main; main.plot_avg_transmission_cost(\"./plots_training\",$params)"

### Comparison of parameters

#params="$numrep,end_time=$end_time,values_n=$values_n,values_p=$values_p"
params="$numrep,end_time=$end_time,values_n=$values_n,values_p=$values_nevents"
echo "centr"
$JOB $SHORT $COMMAND -c "import main; main.compare_centr(\"./plots_cen\",$params)"
echo "centr pretrain"
$JOB $SHORT $COMMAND -c "import main; main.compare_centr(\"./plots_cen_pretrain\",$params,pretrain=True)"
echo "centr single"
$JOB $SHORT $COMMAND -c "import main; main.compare_centr(\"./plots_cen_single\",$params,single=True)"
echo "centr pretrain"
$JOB $SHORT $COMMAND -c "import main; main.compare_centr(\"./plots_cen_single_pretrain\",$params,pretrain=True,single=True)"

params="$numrep,end_time=$end_time,values_n=$values_n,values_ns=$values_ns,const_p=$const_p"
#params="$numrep,end_time=$end_time,values_n=$values_n,values_ns=$values_ns,const_p=$const_nevents"
echo "dec with fixed nevents"
$JOB $SHORT $COMMAND -c "import main; main.compare_decentr_nevents(\"./plots_dec\",$params)"
echo "dec pretrain with fixed nevents"
$JOB $SHORT $COMMAND -c "import main; main.compare_decentr_nevents(\"./plots_dec_pretrain\",$params,pretrain=True)"
echo "dec learn with fixed nevents"
$JOB $SHORT $COMMAND -c "import main; main.compare_decentr_nevents(\"./plots_dec_learn\",$params,learn_from_neighbors=True)"
echo "dec pretrain and learn with fixed nevents"
$JOB $SHORT $COMMAND -c "import main; main.compare_decentr_nevents(\"./plots_dec_pretrain_learn\",$params,learn_from_neighbors=True,pretrain=True)"

echo "dec single learn with fixed nevents"
$JOB $SHORT $COMMAND -c "import main; main.compare_decentr_nevents(\"./plots_dec_single_learn\",$params,learn_from_neighbors=True,single=True)"
echo "dec single pretrain and learn with fixed nevents"
$JOB $SHORT $COMMAND -c "import main; main.compare_decentr_nevents(\"./plots_dec_single_pretrain_learn\",$params,learn_from_neighbors=True,pretrain=True,single=True)"

# ## special test with p=0.8
# params="$numrep,end_time=$end_time,values_n=$values_n,values_ns=$values_ns,const_p=0.8"
# $JOB $SHORT $COMMAND -c "import main; main.compare_decentr_prob(\"./plots_dec\",$params)"

#params="$numrep,end_time=$end_time,values_n=$values_n,values_p=$values_p,const_ns=$const_ns"
params="$numrep,end_time=$end_time,values_n=$values_n,values_p=$values_nevents,const_ns=$const_ns"
echo "dec with fixed ns"
$JOB $SHORT $COMMAND -c "import main; main.compare_decentr_ns(\"./plots_dec\",$params)"
echo "dec pretrain with fixed ns"
$JOB $SHORT $COMMAND -c "import main; main.compare_decentr_ns(\"./plots_dec_pretrain\",$params,pretrain=True)"
echo "dec learn with fixed ns"
$JOB $SHORT $COMMAND -c "import main; main.compare_decentr_ns(\"./plots_dec_learn\",$params,learn_from_neighbors=True)"
echo "dec pretrain and learn with fixed ns"
$JOB $SHORT $COMMAND -c "import main; main.compare_decentr_ns(\"./plots_dec_pretrain_learn\",$params,pretrain=True,learn_from_neighbors=True)"

echo "dec single learn with fixed ns"
$JOB $SHORT $COMMAND -c "import main; main.compare_decentr_ns(\"./plots_dec_single_learn\",$params,learn_from_neighbors=True,single=True)"
echo "dec single pretrain and learn with fixed ns"
$JOB $SHORT $COMMAND -c "import main; main.compare_decentr_ns(\"./plots_dec_single_pretrain_learn\",$params,pretrain=True,learn_from_neighbors=True,single=True)"

### supervised
#params="$numrep,end_time=$end_time,values_n=$values_n,values_p=$values_p"
params="$numrep,end_time=$end_time,values_n=$values_n,values_p=$values_nevents"
echo "supervised centr"
$JOB $SHORT $COMMAND -c "import main; main.compare_centr(\"./plots_supervised_cen\",$params,supervised=True)"
echo "supervised centr single"
$JOB $SHORT $COMMAND -c "import main; main.compare_centr(\"./plots_supervised_cen_single\",$params,single=True,supervised=True)"

params="$numrep,end_time=$end_time,values_n=$values_n,values_ns=$values_ns,const_p=$const_nevents"
#params="$numrep,end_time=$end_time,values_n=$values_n,values_ns=$values_ns,const_p=$const_p"
echo "supervised dec with fixed nevents"
$JOB $SHORT $COMMAND -c "import main; main.compare_decentr_nevents(\"./plots_supervised_dec\",$params,supervised=True)"
echo "supervised dec learn with fixed nevents"
$JOB $SHORT $COMMAND -c "import main; main.compare_decentr_nevents(\"./plots_supervised_dec_learn\",$params,learn_from_neighbors=True,supervised=True)"

echo "supervised dec single learn with fixed nevents"
$JOB $SHORT $COMMAND -c "import main; main.compare_decentr_nevents(\"./plots_supervised_dec_single_learn\",$params,learn_from_neighbors=True,single=True,supervised=True)"

# ## special test with p=0.8
# params="$numrep,end_time=$end_time,values_n=$values_n,values_ns=$values_ns,const_p=0.8"
# $JOB $SHORT $COMMAND -c "import main; main.compare_decentr_prob(\"./plots_supervised_dec\",$params,supervised=True)"

#params="$numrep,end_time=$end_time,values_n=$values_n,values_p=$values_p,const_ns=$const_ns"
params="$numrep,end_time=$end_time,values_n=$values_n,values_p=$values_nevents,const_ns=$const_ns"
echo "supervised dec with fixed ns"
$JOB $SHORT $COMMAND -c "import main; main.compare_decentr_ns(\"./plots_supervised_dec\",$params,supervised=True)"
echo "supervised dec learn with fixed ns"
$JOB $SHORT $COMMAND -c "import main; main.compare_decentr_ns(\"./plots_supervised_dec_learn\",$params,learn_from_neighbors=True,supervised=True)"

echo "supervised dec single learn with fixed ns"
$JOB $SHORT $COMMAND -c "import main; main.compare_decentr_ns(\"./plots_supervised_dec_single_learn\",$params,learn_from_neighbors=True,single=True,supervised=True)"
