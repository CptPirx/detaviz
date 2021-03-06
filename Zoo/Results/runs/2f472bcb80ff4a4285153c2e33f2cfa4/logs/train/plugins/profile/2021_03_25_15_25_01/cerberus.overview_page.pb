?	Yk(?'I@Yk(?'I@!Yk(?'I@	?7~q?S@?7~q?S@!?7~q?S@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6Yk(?'I@??j??G??1??x??@A??????I?/???@Y???
?C@*	V-?\?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator???<H?C@!??wv??X@)???<H?C@1??wv??X@:Preprocessing2F
Iterator::ModelRb??v?C@!      Y@)??@???x?10?y p???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismLP÷??C@!0??X@)v?1<??x?1????P&??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap?n??C@!?<???X@)]?`7l[t?1Dd?Qp???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 78.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.moderate"?6.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?7~q?S@I?r?w?@Qm?n??+@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??j??G????j??G??!??j??G??      ??!       "	??x??@??x??@!??x??@*      ??!       2	????????????!??????:	?/???@?/???@!?/???@B      ??!       J	???
?C@???
?C@!???
?C@R      ??!       Z	???
?C@???
?C@!???
?C@b      ??!       JGPUY?7~q?S@b q?r?w?@ym?n??+@?"-
IteratorGetNext/_1_Send?[z?&??!?[z?&??"D
&gradient_tape/model/bl_1/MatMul/MatMulMatMul?2? ????!b*?F<??0"/
model/bl_1/MatMulMatMulz+kw????!???>??0"F
*gradient_tape/model/bl_1/MatMul_1/MatMul_1MatMul??pd ??!k)B???"3
model/bl_1/transpose	Transpose???l???!?#?p???"F
(gradient_tape/model/bl_1/MatMul_1/MatMulMatMul<~]??v??!???3<??0"5
model/bl_1/transpose_2	Transpose??p??0??!6?[V????"M
.gradient_tape/model/bl_1/transpose_2/transpose	Transpose?Ň$????!M??*???"F
*gradient_tape/model/bl_2/MatMul_1/MatMul_1MatMul????Y@??!sw呃???"/
model/bl_1/MatMul_1MatMulI?)?Ba??!?????Q      Y@Y|t???G'@ap???V@q??N\a??y?w'ށ??"?

host?Your program is HIGHLY input-bound because 78.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?6.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 