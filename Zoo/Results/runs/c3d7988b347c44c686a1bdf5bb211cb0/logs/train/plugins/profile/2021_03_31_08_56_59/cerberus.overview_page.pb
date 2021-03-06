?	??ٮP?g@??ٮP?g@!??ٮP?g@	N????TB@N????TB@!N????TB@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??ٮP?g@l\??O@1?e3??[@A%?)? ???I'?5??@Y'?_qQ@*	??~j?S?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator?DJ?y?Q@!?67?X@)?DJ?y?Q@1?67?X@:Preprocessing2F
Iterator::Model?~k'J?Q@!      Y@)l|&??i??1~?g ??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap?{??P?Q@!?H??X@)??n??o??1???Ｄ?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelisml?<?Q@!>??X@)?1?3/???1??s?_͔?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 36.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9N????TB@Ix?x???@Q?EqNZM@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	l\??O@l\??O@!l\??O@      ??!       "	?e3??[@?e3??[@!?e3??[@*      ??!       2	%?)? ???%?)? ???!%?)? ???:	'?5??@'?5??@!'?5??@B      ??!       J	'?_qQ@'?_qQ@!'?_qQ@R      ??!       Z	'?_qQ@'?_qQ@!'?_qQ@b      ??!       JGPUYN????TB@b qx?x???@y?EqNZM@?"d
8gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?o??5???!?o??5???0"-
IteratorGetNext/_1_Send
옩%˨?!????-]??"b
6gradient_tape/model/conv1d/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???q???!???q???0"d
8gradient_tape/model/conv1d_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??.?T=??!0???w??0"b
7gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropInputConv2DBackpropInput?	??ȗ?!d??p??0"1
model/conv1d_8/conv1dConv2D?0?????!?UG?Td??"d
8gradient_tape/model/conv1d_9/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?<??RR??!ݾM?N??0"d
8gradient_tape/model/conv1d_5/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?qg??M??!????'???0"1
model/conv1d_5/conv1dConv2D????Zy??!??Ě????"1
model/conv1d_9/conv1dConv2DxH?o??!p7???
??Q      Y@Y	????@a?'ji9X@qݾ??Ŵ??y??w?f?"?	
host?Your program is HIGHLY input-bound because 36.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
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