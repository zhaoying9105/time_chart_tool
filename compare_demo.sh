python3 -m time_chart_tool compare --aggregation 'name,call_stack'   --show 'call_stack,kernel-names,pid,tid' --compare "dtype,shape" \
 'data/executor_trainer-runner_6_0_20251203_140933/mlu_profiler_trace_rank_0_0_0.json:label_1' \
 'data/executor_trainer-runner_6_0_20251203_140933/mlu_profiler_trace_rank_0_0_0.json:label_2'