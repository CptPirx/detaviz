?	?l??3?$@?l??3?$@!?l??3?$@	??40eE@??40eE@!??40eE@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?l??3?$@?B?y???1~p>u??@A???;?B??I?? @???Y?@?9w?@*	+??3?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator????х@!?/~y'?X@)????х@1?/~y'?X@:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap??f?@!?????X@)%????1?8?X???:Preprocessing2F
Iterator::Model7???@!      Y@)?3??`i?1Pe?*??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?"??~?@!?&Iu?X@)?Xİ?h?1?)? ????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 42.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.moderate"?11.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2s5.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??40eE@IT?I??0@QJo?mNID@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?B?y????B?y???!?B?y???      ??!       "	~p>u??@~p>u??@!~p>u??@*      ??!       2	???;?B?????;?B??!???;?B??:	?? @????? @???!?? @???B      ??!       J	?@?9w?@?@?9w?@!?@?9w?@R      ??!       Z	?@?9w?@?@?9w?@!?@?9w?@b      ??!       JGPUY??40eE@b qT?I??0@yJo?mNID@?"-
IteratorGetNext/_1_Send??????!??????"D
&gradient_tape/model/bl_1/MatMul/MatMulMatMul??7????!f??????0"/
model/bl_1/MatMulMatMuli?l$lT??!s?VN????0"F
*gradient_tape/model/bl_1/MatMul_1/MatMul_1MatMul?v??I??!? U/C??"F
(gradient_tape/model/bl_1/MatMul_1/MatMulMatMul??d?i???!?'X}????0"3
model/bl_1/transpose	Transpose?3?]???! ?E(????"F
*gradient_tape/model/bl_2/MatMul_1/MatMul_1MatMul??Ŀ?F??!?Df????"/
model/bl_1/MatMul_1MatMul?s?????!_?f?7??"M
.gradient_tape/model/bl_1/transpose_2/transpose	Transpose??+?K#??!./??????"5
model/bl_1/transpose_2	Transpose?yZ`U???!??n?E??Q      Y@Y|t???G'@ap???V@q?`Ϟ??y?"?T????"?
host?Your program is HIGHLY input-bound because 42.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?11.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"s5.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 