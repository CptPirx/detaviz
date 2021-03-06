?	??yS?9^@??yS?9^@!??yS?9^@	
??b??S@
??b??S@!
??b??S@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??yS?9^@l??7F@1????(9.@Ao+?6??I????@Yl#??W@*	??|????@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generatorkf-?X@!?r	{I?X@)kf-?X@1?r	{I?X@:Preprocessing2F
Iterator::Model??KX@!      Y@)o??\????1߁"??e??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismw+Kt?X@!?VY?D?X@)?cx?g???1n?Xp?s??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap????X@!'??1??X@)?qR??8s?1??f??s?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 79.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.moderate"?5.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9
??b??S@I???LK? @Q?U????(@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	l??7F@l??7F@!l??7F@      ??!       "	????(9.@????(9.@!????(9.@*      ??!       2	o+?6??o+?6??!o+?6??:	????@????@!????@B      ??!       J	l#??W@l#??W@!l#??W@R      ??!       Z	l#??W@l#??W@!l#??W@b      ??!       JGPUY
??b??S@b q???LK? @y?U????(@?"-
IteratorGetNext/_1_Send?5?O??!?5?O??"D
&gradient_tape/model/bl_1/MatMul/MatMulMatMulD??|????!???????0"/
model/bl_1/MatMulMatMul?Va.ç?!X??9W??0"3
model/bl_1/transpose	Transpose6 ??P???!Zd?g?#??"F
*gradient_tape/model/bl_1/MatMul_1/MatMul_1MatMul?m?7r??!?3Gm???"F
(gradient_tape/model/bl_1/MatMul_1/MatMulMatMul?l??Ճ??!f?N?|???0"M
.gradient_tape/model/bl_1/transpose_2/transpose	Transpose[?!??x?!????w??"5
model/bl_1/transpose_2	Transpose1>7?R?x?!+d[Cn3??"F
*gradient_tape/model/bl_2/MatMul_1/MatMul_1MatMul? B?(?w?!m????b??"/
model/bl_1/MatMul_1MatMulw %5Nw?!?2?6???Q      Y@Y|t???G'@ap???V@q??؉?d??y?pԪ&??"?

host?Your program is HIGHLY input-bound because 79.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?5.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 