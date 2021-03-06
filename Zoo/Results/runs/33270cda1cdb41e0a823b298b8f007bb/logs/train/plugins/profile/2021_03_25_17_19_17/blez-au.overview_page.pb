?	0?[w??)@0?[w??)@!0?[w??)@	?eR }HK@?eR }HK@!?eR }HK@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails60?[w??)@?(?	0??1W?[?@A???/fK??I????/v??Y?;p?S@*	?z??@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator#M?<?@!"? o??X@)#M?<?@1"? o??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?vN?@?@!??<??X@)??Gߤi??15?)????:Preprocessing2F
Iterator::Model[%X?@!      Y@)?#?&?f?1>?????:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap?.oך@!=3m?3?X@)??.??Y?1ף??d??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 54.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.moderate"?11.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2s5.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?eR }HK@I???{0@Q*?I?"?<@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?(?	0???(?	0??!?(?	0??      ??!       "	W?[?@W?[?@!W?[?@*      ??!       2	???/fK?????/fK??!???/fK??:	????/v??????/v??!????/v??B      ??!       J	?;p?S@?;p?S@!?;p?S@R      ??!       Z	?;p?S@?;p?S@!?;p?S@b      ??!       JGPUY?eR }HK@b q???{0@y*?I?"?<@?"-
IteratorGetNext/_1_Send?TlxL???!?TlxL???"D
&gradient_tape/model/bl_1/MatMul/MatMulMatMul g#Y??!5????P??0"/
model/bl_1/MatMulMatMul?uVX???!?R?3???0"F
*gradient_tape/model/bl_1/MatMul_1/MatMul_1MatMul?o???	??!?)@??3??"F
(gradient_tape/model/bl_1/MatMul_1/MatMulMatMul?j??h??!ն?^???0"/
model/bl_1/MatMul_1MatMul???Ж???!?;}k???"F
*gradient_tape/model/bl_2/MatMul_1/MatMul_1MatMul??l????!.L???]??"5
model/bl_1/transpose_2	Transpose?qW?>*??!?Ҝ'??"M
.gradient_tape/model/bl_1/transpose_2/transpose	Transpose??T????!T?45????"3
model/bl_1/transpose	Transpose?t|ҮƔ?!??ȫ????Q      Y@Y|t???G'@ap???V@qn??Q???y?s%&,???"?
host?Your program is HIGHLY input-bound because 54.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?11.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"s5.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 