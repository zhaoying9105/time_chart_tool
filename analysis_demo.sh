export PYTHONPATH=$PWD/src:$PYTHONPATH

# # 相同op name放在一行，展示op 的type,call_stack,kernel-names,pid,tid,kernel-duration
# python3 -m time_chart_tool analysis --aggregation 'name' --show 'dtype,call_stack,kernel-names,pid,tid,kernel-duration' --label 'test'  'data/executor_trainer-runner_6_0_20251203_140933/mlu_profiler_trace_rank_0_0_0.json' 

# # 相同op name,call_stack,dtype放在一行，展示op 的kernel-names,pid,tid,kernel-duration
# python3 -m time_chart_tool analysis --aggregation 'name,call_stack,dtype' --show 'kernel-names,pid,tid,kernel-duration' --label 'test'  'data/executor_trainer-runner_6_0_20251203_140933/mlu_profiler_trace_rank_0_0_0.json' 

# 按照op 下发顺序展示
python3 -m time_chart_tool analysis --aggregation 'op_index' --show 'name,call_stack,dtype,shape,kernel-names,pid,tid' --label 'test'  'data/executor_trainer-runner_6_0_20251203_140933/mlu_profiler_trace_rank_0_0_0.json' 
