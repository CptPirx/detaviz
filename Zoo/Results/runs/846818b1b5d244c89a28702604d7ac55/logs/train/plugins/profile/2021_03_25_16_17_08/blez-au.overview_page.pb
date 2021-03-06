?	?+???-@?+???-@!?+???-@	<??P?QK@<??P?QK@!<??P?QK@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?+???-@????_??1????@Aߩ?{????IY?????Y???K @*	???(?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::GeneratorR`Lx @!????'?X@)R`Lx @1????'?X@:Preprocessing2F
Iterator::Model?7??{ @!      Y@)???ig?1??????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism~??E}z @!??????X@)?f??f?1Ӥ	?ʽ??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap?Bty @!uBq&??X@)?Z'.?+`?1??y&h???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 54.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.moderate"?9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2s5.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9;??P?QK@I??vL?U.@Q?`?7?1>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????_??????_??!????_??      ??!       "	????@????@!????@*      ??!       2	ߩ?{????ߩ?{????!ߩ?{????:	Y?????Y?????!Y?????B      ??!       J	???K @???K @!???K @R      ??!       Z	???K @???K @!???K @b      ??!       JGPUY;??P?QK@b q??vL?U.@y?`?7?1>@?"-
IteratorGetNext/_1_Sendr2?:n??!r2?:n??"D
&gradient_tape/model/bl_1/MatMul/MatMulMatMul?o]k???!??p78???0"/
model/bl_1/MatMulMatMul????ֲ?!(R??H??0"F
(gradient_tape/model/bl_1/MatMul_1/MatMulMatMul?GGҦ??!?F%?????0"F
*gradient_tape/model/bl_1/MatMul_1/MatMul_1MatMul??3?U??!??xPjc??"/
model/bl_1/MatMul_1MatMul?????4??!??.@E??"M
.gradient_tape/model/bl_1/transpose_2/transpose	Transpose?c?????!?ƫ??"5
model/bl_1/transpose_2	Transpose??????!???E???"3
model/bl_1/transpose	Transpose?8??+Ԗ?!?@?	????"F
*gradient_tape/model/bl_2/MatMul_1/MatMul_1MatMul???T???!>????V??Q      Y@Y|t???G'@ap???V@qA???W??yh*S?h??"?
host?Your program is HIGHLY input-bound because 54.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"s5.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 