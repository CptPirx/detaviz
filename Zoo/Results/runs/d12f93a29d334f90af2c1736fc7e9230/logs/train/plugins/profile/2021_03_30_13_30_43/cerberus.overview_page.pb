?	en?݃+@en?݃+@!en?݃+@	?}??h?D@?}??h?D@!?}??h?D@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6en?݃+@?0?{????1c??????Awf??\ì?I??	?_w@Y	]??@*	A`???;?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::GeneratorLU???@!?#??X@)LU???@1?#??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismO<g?@!?'????X@)???4??1vr?????:Preprocessing2F
Iterator::ModelЗ??\?@!      Y@)?n??S}?1m?3??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap{?L?x?@!??~??X@)f?ʉvu?1???o???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 41.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.high"?40.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2s7.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?}??h?D@I?ra?oH@QDx7?W?"@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?0?{?????0?{????!?0?{????      ??!       "	c??????c??????!c??????*      ??!       2	wf??\ì?wf??\ì?!wf??\ì?:	??	?_w@??	?_w@!??	?_w@B      ??!       J		]??@	]??@!	]??@R      ??!       Z		]??@	]??@!	]??@b      ??!       JGPUY?}??h?D@b q?ra?oH@yDx7?W?"@?"-
IteratorGetNext/_1_Send'???E???!'???E???"F
*gradient_tape/model/bl_1/MatMul_1/MatMul_1MatMul?G}??!?????o??"F
*gradient_tape/model/bl_2/MatMul_1/MatMul_1MatMul	??pe???!(?! ???"D
&gradient_tape/model/bl_1/MatMul/MatMulMatMul???:???!=?i?????0"F
*gradient_tape/model/tabl/MatMul_1/MatMul_1MatMul????N??!Y?|?j???"/
model/bl_1/MatMul_1MatMul$׽???!>{??????"F
(gradient_tape/model/bl_1/MatMul_1/MatMulMatMul!?>? d??!kX?????0"/
model/bl_1/MatMulMatMul?8ҿ???!O?*D?b??0"D
(gradient_tape/model/bl_2/MatMul/MatMul_1MatMul??7}W??!???-r??"/
model/bl_2/MatMul_1MatMul??sX.???!0g??????Q      Y@Y|t???G'@ap???V@q-?^?F???y!y??o??"?
host?Your program is HIGHLY input-bound because 41.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?40.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"s7.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 