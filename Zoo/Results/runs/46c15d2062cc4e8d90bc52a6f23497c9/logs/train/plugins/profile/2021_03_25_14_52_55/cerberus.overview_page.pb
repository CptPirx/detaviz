?	?p!??W@?p!??W@!?p!??W@	JIz:T@JIz:T@!JIz:T@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?p!??W@??	h???1P??0{?+@A^h??HK??I5(???
@Y??s??S@*	??Ν??@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator?׃I?7S@!D?(??X@)?׃I?7S@1D?(??X@:Preprocessing2F
Iterator::Model?????8S@!      Y@)?27߈?y?13;2W?܀?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism>?ɋ8S@!nF?y?X@))??Rbw?1s??i~?:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMapN+?@.8S@!:<er??X@);?ީ?{n?1J?
?w?s?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 80.9% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.moderate"?3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9JIz:T@Iན?PQ@Q???"??-@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??	h?????	h???!??	h???      ??!       "	P??0{?+@P??0{?+@!P??0{?+@*      ??!       2	^h??HK??^h??HK??!^h??HK??:	5(???
@5(???
@!5(???
@B      ??!       J	??s??S@??s??S@!??s??S@R      ??!       Z	??s??S@??s??S@!??s??S@b      ??!       JGPUYJIz:T@b qན?PQ@y???"??-@?"-
IteratorGetNext/_1_SendL~?x? ??!L~?x? ??"D
&gradient_tape/model/bl_1/MatMul/MatMulMatMul??B(?v??!Hէ=?/??0"/
model/bl_1/MatMulMatMul?w+?????!Č
?h???0"3
model/bl_1/transpose	Transpose?a???!52}???"F
*gradient_tape/model/bl_1/MatMul_1/MatMul_1MatMul??FT?̐?!?g?v?u??"F
(gradient_tape/model/bl_1/MatMul_1/MatMulMatMul??/T?D??!?&?????0"5
model/bl_1/transpose_2	Transpose	?&!?9??!(????/??"M
.gradient_tape/model/bl_1/transpose_2/transpose	Transposec?? ???!???w??"F
*gradient_tape/model/bl_2/MatMul_1/MatMul_1MatMul?Kti??!90?Ğ???"/
model/bl_1/MatMul_1MatMul?>??ƒ??!4-z?????Q      Y@Y|t???G'@ap???V@q?׭?մ?yC.?? |??"?

host?Your program is HIGHLY input-bound because 80.9% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 