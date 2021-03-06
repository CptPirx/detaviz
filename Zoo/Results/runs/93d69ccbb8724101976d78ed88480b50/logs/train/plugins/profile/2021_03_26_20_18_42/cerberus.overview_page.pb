?	?PSi?a@?PSi?a@!?PSi?a@	?i?Ŀ+:@?i?Ŀ+:@!?i?Ŀ+:@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?PSi?a@???ܴ%:@1?d??7?R@Af3??J??I>[{s@Y]???oB@*	X9??	?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator??^?2qC@!9?/K??X@)??^?2qC@19?/K??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?\??yC@!j???7?X@)!\?z???1?)Hy???:Preprocessing2F
Iterator::Modelr?#~C@!      Y@)??T[??1[W?? ??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap?"nNsC@!U??5?X@)?dT?ݐ?1%?h?¡??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 26.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2t18.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?i?Ŀ+:@IF*?4&4@Q?5???J@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???ܴ%:@???ܴ%:@!???ܴ%:@      ??!       "	?d??7?R@?d??7?R@!?d??7?R@*      ??!       2	f3??J??f3??J??!f3??J??:	>[{s@>[{s@!>[{s@B      ??!       J	]???oB@]???oB@!]???oB@R      ??!       Z	]???oB@]???oB@!]???oB@b      ??!       JGPUY?i?Ŀ+:@b qF*?4&4@y?5???J@?"d
8gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropFilterConv2DBackpropFilter:????!:????0"-
IteratorGetNext/_1_Send??"???!

?e6??"b
6gradient_tape/model/conv1d/conv1d/Conv2DBackpropFilterConv2DBackpropFilter????{S??!?y?????0"d
8gradient_tape/model/conv1d_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilterMqp?2??!D???A???0"d
8gradient_tape/model/conv1d_5/conv1d/Conv2DBackpropFilterConv2DBackpropFilter6o?NgQ??!+???n???0"d
8gradient_tape/model/conv1d_9/conv1d/Conv2DBackpropFilterConv2DBackpropFilter????CJ??!?%%#????0"b
7gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropInputConv2DBackpropInput?B?w㛔?!?"?3=??0"1
model/conv1d_8/conv1dConv2D?9??[M??!?Zy?o???"2
model/conv1d_10/conv1dConv2Ds??0?`??!??t{	??"1
model/conv1d_6/conv1dConv2D?{?G??!?:3??-??Q      Y@Y?t??-	@a9_4Ñ6X@qZS??d???y?6?XOf?"?

host?Your program is HIGHLY input-bound because 26.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t18.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 