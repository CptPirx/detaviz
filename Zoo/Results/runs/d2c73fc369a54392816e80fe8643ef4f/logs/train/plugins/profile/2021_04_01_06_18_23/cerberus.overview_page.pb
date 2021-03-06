?	3R???k@3R???k@!3R???k@	?P??e@@?P??e@@!?P??e@@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails63R???k@?????@1?qs?Gb@Az5@i?Q??I^ؚ??D@Y??P?\?Q@*	?Z?1?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator&9{?R@!n????X@)&9{?R@1n????X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?j+???R@!?B???X@)?5?ڋh??1xit?c??:Preprocessing2F
Iterator::Model??7/N?R@!      Y@)?ɐc??1y??u`??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap??mm??R@!???wg?X@)???J#f??13X?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 32.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?P??e@@I ?*??@Q??".cP@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????@?????@!?????@      ??!       "	?qs?Gb@?qs?Gb@!?qs?Gb@*      ??!       2	z5@i?Q??z5@i?Q??!z5@i?Q??:	^ؚ??D@^ؚ??D@!^ؚ??D@B      ??!       J	??P?\?Q@??P?\?Q@!??P?\?Q@R      ??!       Z	??P?\?Q@??P?\?Q@!??P?\?Q@b      ??!       JGPUY?P??e@@b q ?*??@y??".cP@?"d
8gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??◬?!??◬?0"-
IteratorGetNext/_1_Send?7?8????!???Ē??"b
6gradient_tape/model/conv1d/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?$?d?o??!?BnZ???0"d
8gradient_tape/model/conv1d_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilterV|?dtA??!?I???M??0"d
8gradient_tape/model/conv1d_9/conv1d/Conv2DBackpropFilterConv2DBackpropFilter? ????!WK=?,J??0"d
8gradient_tape/model/conv1d_5/conv1d/Conv2DBackpropFilterConv2DBackpropFilter6??c????!^???L??0"1
model/conv1d_8/conv1dConv2D??rY??!z???z???"b
7gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropInputConv2DBackpropInputc???{???!?]i*e???0"1
model/conv1d_6/conv1dConv2D???5pא?!?-ܛ??"2
model/conv1d_10/conv1dConv2D4????!?-??}???Q      Y@Y=?]??@a6??9X@q?[?ݲz??y?Ǟ?U?a?"?	
host?Your program is HIGHLY input-bound because 32.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 