?	5???@\@5???@\@!5???@\@	_??M@_??M@!_??M@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails65???@\@?????P@1?#??GY@A??ihw??IL??OH@Y;??bF?@*	!?rh???@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator#??)ȏ@!Ã?Ѥ?X@)#??)ȏ@1Ã?Ѥ?X@:Preprocessing2F
Iterator::Model|ds?<?@!      Y@)B?p?-??1&???
???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismM??y ?@!??D?]?X@)7?X?O??1???]????:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap1ҋ???@!???쯻X@)i?wak??1s??6??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.moderate"?3.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9_??M@IP6ǁ?@Q????R^V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????P@?????P@!?????P@      ??!       "	?#??GY@?#??GY@!?#??GY@*      ??!       2	??ihw????ihw??!??ihw??:	L??OH@L??OH@!L??OH@B      ??!       J	;??bF?@;??bF?@!;??bF?@R      ??!       Z	;??bF?@;??bF?@!;??bF?@b      ??!       JGPUY_??M@b qP6ǁ?@y????R^V@?"d
8gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?u.Q???!?u.Q???0"d
8gradient_tape/model/conv1d_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??@X1J??!Gq?~mZ??0"b
7gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropInputConv2DBackpropInput???MM??!'o??????0"1
model/conv1d_8/conv1dConv2D0p??*??!??Q?<<??"d
8gradient_tape/model/conv1d_5/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?5Q\? ??!N??:O|??0"d
8gradient_tape/model/conv1d_9/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???<????!?zsbe???0"1
model/conv1d_9/conv1dConv2D,	??¥??!?[??n??"1
model/conv1d_5/conv1dConv2D???????!??n?/!??"2
model/conv1d_10/conv1dConv2D?:????!0-?
?@??"1
model/conv1d_6/conv1dConv2DĎ?'????!V#Mp??Q      Y@Y?[????@a#??A?8X@qT(?wt??y?u???h?"?

both?Your program is MODERATELY input-bound because 5.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 