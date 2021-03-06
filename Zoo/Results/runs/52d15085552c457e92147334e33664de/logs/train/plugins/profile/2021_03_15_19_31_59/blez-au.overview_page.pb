?	????a?N@????a?N@!????a?N@	?8??E?@?8??E?@!?8??E?@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6????a?N@??^
?@1?????K@A???6?4??I?}8gD??Y^gE?t@*	1??+?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator^i???@!]??/??X@)^i???@1]??/??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismK?X?U?@!??7?q?X@)?L???x?1B?&`]!??:Preprocessing2F
Iterator::Model?9:Z?@!      Y@)	?L?nx?1????%??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMapx}??O?@!B??>??X@)?iP4`q?1I??k:???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2s3.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?8??E?@IP#p?@@Q<?b???U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??^
?@??^
?@!??^
?@      ??!       "	?????K@?????K@!?????K@*      ??!       2	???6?4?????6?4??!???6?4??:	?}8gD???}8gD??!?}8gD??B      ??!       J	^gE?t@^gE?t@!^gE?t@R      ??!       Z	^gE?t@^gE?t@!^gE?t@b      ??!       JGPUY?8??E?@b qP#p?@@y<?b???U@?"d
8gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropFilterConv2DBackpropFilter	?F/???!	?F/???0"-
IteratorGetNext/_1_Send9SMĤ?!???L?@??"b
6gradient_tape/model/conv1d/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?o,?C???!???????0"d
8gradient_tape/model/conv1d_5/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??%????!???]???0"d
8gradient_tape/model/conv1d_9/conv1d/Conv2DBackpropFilterConv2DBackpropFilter'?v????!y(<;=???0"d
8gradient_tape/model/conv1d_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?G??aj??!j?Sm?q??0"b
7gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropInputConv2DBackpropInput??>???!????d??0"/
model/conv1d/conv1dConv2D??$?S??!f??V?'??"d
8gradient_tape/model/conv1d_6/conv1d/Conv2DBackpropFilterConv2DBackpropFilterEV?B??!??Өk??0"1
model/conv1d_8/conv1dConv2DHكq??!???????Q      Y@Y?q?q@a?q?q<X@q?mL????y?&=??n?"?

both?Your program is MODERATELY input-bound because 6.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nomoderate"s3.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 