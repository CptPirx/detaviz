?	NA~6? @NA~6? @!NA~6? @	?????#H@?????#H@!?????#H@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6NA~6? @???h*??1wLݕ]@A???`ũ??I???(_???Y?+f?@*	??S??@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::GeneratorH??'?j@!?nga?X@)H??'?j@1?nga?X@:Preprocessing2F
Iterator::Model???q?t@!      Y@)?????o?1?A?`?7??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismeM.?p@!??'??X@)???x!n?1
͜^?߶?:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap???m@!?#?:?X@)?_??s`?1R=?Es???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 48.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.high"?16.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2s6.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?????#H@I%߹??7@Qe?p?S?<@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???h*?????h*??!???h*??      ??!       "	wLݕ]@wLݕ]@!wLݕ]@*      ??!       2	???`ũ?????`ũ??!???`ũ??:	???(_??????(_???!???(_???B      ??!       J	?+f?@?+f?@!?+f?@R      ??!       Z	?+f?@?+f?@!?+f?@b      ??!       JGPUY?????#H@b q%߹??7@ye?p?S?<@?"-
IteratorGetNext/_1_Send??p??!??p??"D
&gradient_tape/model/bl_1/MatMul/MatMulMatMul&?iR???!??????0"F
*gradient_tape/model/bl_1/MatMul_1/MatMul_1MatMul??j+??!?WX^?u??"/
model/bl_1/MatMulMatMul?qa?/??!??D????0"F
(gradient_tape/model/bl_1/MatMul_1/MatMulMatMulqMWZ?8??!??G憁??0"F
*gradient_tape/model/bl_2/MatMul_1/MatMul_1MatMul͛?ڬz??!t!??1???"/
model/bl_1/MatMul_1MatMulg?l?(֡?!???C????"M
.gradient_tape/model/bl_1/transpose_2/transpose	Transpose?֖????!Z???8???"F
*gradient_tape/model/tabl/MatMul_1/MatMul_1MatMul?֖????!?YW?????"5
model/bl_1/transpose_2	Transpose@?V ???!Z-u???Q      Y@Y|t???G'@ap???V@qA?g??,??y??????"?
host?Your program is HIGHLY input-bound because 48.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?16.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"s6.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 