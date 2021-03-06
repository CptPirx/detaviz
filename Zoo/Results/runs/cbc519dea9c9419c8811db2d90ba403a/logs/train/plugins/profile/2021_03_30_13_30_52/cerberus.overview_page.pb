?	y?'e?2@y?'e?2@!y?'e?2@	kLDDp?F@kLDDp?F@!kLDDp?F@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6y?'e?2@
MK????1w??-u??A5?b??^??I?"??D@YzrM??!@*	?C?l??@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator?K??$o!@!?e??X@)?K??$o!@1?e??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???s!@!u???w?X@)"????v?1??7???:Preprocessing2F
Iterator::ModelG?@?]v!@!      Y@)??
?s?1?ST??@??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMapH4?"q!@!x&q?X@)???^o?1??o>??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 45.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.high"?39.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2s4.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9kLDDp?F@I ?ܿ(?E@Q?5|???$@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	
MK????
MK????!
MK????      ??!       "	w??-u??w??-u??!w??-u??*      ??!       2	5?b??^??5?b??^??!5?b??^??:	?"??D@?"??D@!?"??D@B      ??!       J	zrM??!@zrM??!@!zrM??!@R      ??!       Z	zrM??!@zrM??!@!zrM??!@b      ??!       JGPUYkLDDp?F@b q ?ܿ(?E@y?5|???$@?"-
IteratorGetNext/_1_Send???KJ]??!???KJ]??"F
*gradient_tape/model/bl_1/MatMul_1/MatMul_1MatMul?W_???!
?[????"F
*gradient_tape/model/bl_2/MatMul_1/MatMul_1MatMul??(?C???!?? ?????"/
model/bl_1/MatMulMatMulvt3????!2L?K??0"D
&gradient_tape/model/bl_1/MatMul/MatMulMatMulx?.M?2??!@?F4???0"F
*gradient_tape/model/tabl/MatMul_1/MatMul_1MatMulܻ??Κ??!݅?????"F
(gradient_tape/model/bl_1/MatMul_1/MatMulMatMul???m୚?!ŉ?????0"/
model/bl_1/MatMul_1MatMul?P???h??!K???\???"D
(gradient_tape/model/bl_2/MatMul/MatMul_1MatMul???{??!/T???&??"3
model/bl_1/transpose	Transpose?޻?????!?C]?????Q      Y@Y|t???G'@ap???V@q??e????ya?????"?
host?Your program is HIGHLY input-bound because 45.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?39.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"s4.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 