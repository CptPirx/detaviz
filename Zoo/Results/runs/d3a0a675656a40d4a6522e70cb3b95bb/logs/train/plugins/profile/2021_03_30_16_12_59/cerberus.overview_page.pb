?	0?[w??R@0?[w??R@!0?[w??R@	????K?Q@????K?Q@!????K?Q@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails60?[w??R@??K??@1??qo~?$@AD?в???IC??f?@Y?a??J@*	?Q?^?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator9?Z??J@!?|??#?X@)9?Z??J@1?|??#?X@:Preprocessing2F
Iterator::Model9b-> K@!      Y@)??
(ԃ?1{4??!\??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism)?????J@!???=??X@)h??'?H??1?l???'??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap?7?q??J@!ʂ? ??X@)V?6???z?1I⽠ĩ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 71.9% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.moderate"?8.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2s5.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9????K?Q@IK~???~,@Q??????+@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??K??@??K??@!??K??@      ??!       "	??qo~?$@??qo~?$@!??qo~?$@*      ??!       2	D?в???D?в???!D?в???:	C??f?@C??f?@!C??f?@B      ??!       J	?a??J@?a??J@!?a??J@R      ??!       Z	?a??J@?a??J@!?a??J@b      ??!       JGPUY????K?Q@b qK~???~,@y??????+@?"-
IteratorGetNext/_1_Send??u?,???!??u?,???"D
&gradient_tape/model/bl_1/MatMul/MatMulMatMul?5Q?.??!?~??????0"/
model/bl_1/MatMulMatMul1?~)??!?i4?Q5??0"3
model/bl_1/transpose	Transpose?x?Z??!?.?"???"F
*gradient_tape/model/bl_1/MatMul_1/MatMul_1MatMul??>*???!5??psm??"F
(gradient_tape/model/bl_1/MatMul_1/MatMulMatMul??z?s??!?B???0"5
model/bl_1/transpose_2	Transposey
x_J??!:?I?k(??"M
.gradient_tape/model/bl_1/transpose_2/transpose	Transposeު?rX??!????p??"F
*gradient_tape/model/bl_2/MatMul_1/MatMul_1MatMul>mm?0ހ?!?Vv?=???"/
model/bl_1/MatMul_1MatMul??>*???!?*q_????Q      Y@Y|t???G'@ap???V@qO[?m??ylX?R??"?
host?Your program is HIGHLY input-bound because 71.9% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?8.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"s5.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 