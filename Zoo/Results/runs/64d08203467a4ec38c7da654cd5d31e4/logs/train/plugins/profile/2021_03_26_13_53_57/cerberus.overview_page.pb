?	c????6s@c????6s@!c????6s@	%?O?MD@%?O?MD@!%?O?MD@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6c????6s@??M?@1?I?2!f@A ?g?????Iܝ??.???YZ/?r?5_@*	"??~~??@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generatorx?=\r@^@!????L?X@)x?=\r@^@1????L?X@:Preprocessing2F
Iterator::Model__?R#F^@!      Y@)-?}͡?1?
'sf??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??[??C^@!??͘)?X@)?6qr?C??1??%????:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap?Tm7?A^@!?>{_a?X@)??a?????16???WH??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 40.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9$?O?MD@I?o?f???Q]O???L@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??M?@??M?@!??M?@      ??!       "	?I?2!f@?I?2!f@!?I?2!f@*      ??!       2	 ?g????? ?g?????! ?g?????:	ܝ??.???ܝ??.???!ܝ??.???B      ??!       J	Z/?r?5_@Z/?r?5_@!Z/?r?5_@R      ??!       Z	Z/?r?5_@Z/?r?5_@!Z/?r?5_@b      ??!       JGPUY$?O?MD@b q?o?f???y]O???L@?"-
IteratorGetNext/_1_Send????;???!????;???"d
8gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropFilterConv2DBackpropFilter2>?>???!????6??0"d
8gradient_tape/model/conv1d_9/conv1d/Conv2DBackpropFilterConv2DBackpropFilterV?)aF??!K?1p???0"d
8gradient_tape/model/conv1d_5/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??3????!?	8????0"b
6gradient_tape/model/conv1d/conv1d/Conv2DBackpropFilterConv2DBackpropFilter>?]??R??!?д? ???0"1
model/conv1d_8/conv1dConv2D??Y\???!	?:l???"b
7gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropInputConv2DBackpropInputH??????!Μ~?c???0"d
8gradient_tape/model/conv1d_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilterAF???!yr 29??0"/
model/conv1d/conv1dConv2DV?k???!?.w?I??"e
9gradient_tape/model/conv1d_10/conv1d/Conv2DBackpropFilterConv2DBackpropFilterS??䘐?!c.TS!S??0Q      Y@Y???6?@a\ ?H?:X@q?
?3?*??ys????R?"?	
host?Your program is HIGHLY input-bound because 40.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
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