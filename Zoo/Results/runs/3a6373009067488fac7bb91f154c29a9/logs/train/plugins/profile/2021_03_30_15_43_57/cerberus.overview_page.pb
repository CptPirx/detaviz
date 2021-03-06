?	u?w?<@u?w?<@!u?w?<@	??1"O@??1"O@!??1"O@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6u?w?<@֪]???15?l?/2@AiQ?????I??I`?@Y?E(???1@*	??v?/??@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator?v???]2@! [:???X@)?v???]2@1 [:???X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??w??`2@!?'Ư??X@)b.?|?1\??"???:Preprocessing2F
Iterator::ModelND??~b2@!      Y@)*6?u?!{?1%4?΁r??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap?5|_2@!f?!?N?X@)?h㈵?t?1?UDw~???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 62.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.moderate"?13.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??1"O@Iⲇ"0@Q$??3?5@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	֪]???֪]???!֪]???      ??!       "	5?l?/2@5?l?/2@!5?l?/2@*      ??!       2	iQ?????iQ?????!iQ?????:	??I`?@??I`?@!??I`?@B      ??!       J	?E(???1@?E(???1@!?E(???1@R      ??!       Z	?E(???1@?E(???1@!?E(???1@b      ??!       JGPUY??1"O@b qⲇ"0@y$??3?5@?"-
IteratorGetNext/_1_Send??TI??!??TI??"D
&gradient_tape/model/bl_1/MatMul/MatMulMatMulBf?8??!??|-U???0"/
model/bl_1/MatMulMatMulT??????!/YEHy??0"F
*gradient_tape/model/bl_1/MatMul_1/MatMul_1MatMul??D"??!??i&Z??"3
model/bl_1/transpose	Transpose̟??x??!??cC???"F
(gradient_tape/model/bl_1/MatMul_1/MatMulMatMul G?%6N??!Ԙ??[???0"5
model/bl_1/transpose_2	Transpose??`????!Ծ??N??"M
.gradient_tape/model/bl_1/transpose_2/transpose	TransposeE???????!э3p^???"F
*gradient_tape/model/bl_2/MatMul_1/MatMul_1MatMulB??F?O??!?N1?2??"/
model/bl_1/MatMul_1MatMul?K????!?
{^|???Q      Y@Y|t???G'@ap???V@q???Tr??y?n?+S???"?

host?Your program is HIGHLY input-bound because 62.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?13.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 