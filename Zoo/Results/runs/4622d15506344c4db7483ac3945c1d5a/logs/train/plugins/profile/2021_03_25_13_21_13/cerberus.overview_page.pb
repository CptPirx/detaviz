?	bod?Q@bod?Q@!bod?Q@		?`W?R@	?`W?R@!	?`W?R@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6bod?Q@??5"'@1H?}8"@A?-?|????I??F?@Y??7???J@*	V??@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator0?GQg?J@!????7?X@)0?GQg?J@1????7?X@:Preprocessing2F
Iterator::Model?E`?o?J@!      Y@)_???F??1z????֡?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismr?#?J@!%?&%??X@)??Y?N|?1??܊%I??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap5??.?J@!??????X@)o??;??x?1??t`i??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 75.0% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.moderate"?6.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2s5.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9	?`W?R@I??7?~(@Q?c?^v)@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??5"'@??5"'@!??5"'@      ??!       "	H?}8"@H?}8"@!H?}8"@*      ??!       2	?-?|?????-?|????!?-?|????:	??F?@??F?@!??F?@B      ??!       J	??7???J@??7???J@!??7???J@R      ??!       Z	??7???J@??7???J@!??7???J@b      ??!       JGPUY	?`W?R@b q??7?~(@y?c?^v)@?"-
IteratorGetNext/_1_Send?0wHކ??!?0wHކ??"D
&gradient_tape/model/bl_1/MatMul/MatMulMatMuln?AkK??!%a_?K???0"/
model/bl_1/MatMulMatMul?A?M`Ѫ?!Du9?a}??0"3
model/bl_1/transpose	Transpose ;?c?z??!OY?8I??"F
*gradient_tape/model/bl_1/MatMul_1/MatMul_1MatMul??Aܰ??!?[;7????"F
(gradient_tape/model/bl_1/MatMul_1/MatMulMatMul5?r???!?Xmt`??0"5
model/bl_1/transpose_2	Transpose?bx?E???!?9G9???"M
.gradient_tape/model/bl_1/transpose_2/transpose	Transpose7=x	????!?m???"F
*gradient_tape/model/bl_2/MatMul_1/MatMul_1MatMul???V/ ??!ijǦS??"/
model/bl_1/MatMul_1MatMul??v?????!F??????Q      Y@Y|t???G'@ap???V@qe?:@_??y?
??s??"?
host?Your program is HIGHLY input-bound because 75.0% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?6.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"s5.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 