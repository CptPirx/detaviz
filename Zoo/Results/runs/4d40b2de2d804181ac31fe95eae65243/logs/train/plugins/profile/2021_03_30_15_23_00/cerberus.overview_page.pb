?	n??yfS@n??yfS@!n??yfS@	|`H?|?@|`H?|?@!|`H?|?@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6n??yfS@3???? @1si??+C@A?i???1??I?&P??)@Y?P??n8@*	????=??@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::GeneratorqǛ?	9@!????.?X@)qǛ?	9@1????.?X@:Preprocessing2F
Iterator::Model?zM
"9@!      Y@)?v?
?ݦ?1?%O?????:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMapxρ?9@!?????X@)? ??ǟ?18/񘜿?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism"H?9@!mX,???X@)/?.?H??1??9??*??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 31.5% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.high"?16.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9}`H?|?@I\?~?2o3@Q?U?W?H@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	3???? @3???? @!3???? @      ??!       "	si??+C@si??+C@!si??+C@*      ??!       2	?i???1???i???1??!?i???1??:	?&P??)@?&P??)@!?&P??)@B      ??!       J	?P??n8@?P??n8@!?P??n8@R      ??!       Z	?P??n8@?P??n8@!?P??n8@b      ??!       JGPUY}`H?|?@b q\?~?2o3@y?U?W?H@?"d
8gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?X?????!?X?????0"-
IteratorGetNext/_1_Send??????!Z?5_????"b
6gradient_tape/model/conv1d/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??$????!???qtϿ?0"d
8gradient_tape/model/conv1d_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter7`?k???!?߮?GY??0"d
8gradient_tape/model/conv1d_5/conv1d/Conv2DBackpropFilterConv2DBackpropFilter5oH!??!?Ƽ?j]??0"d
8gradient_tape/model/conv1d_9/conv1d/Conv2DBackpropFilterConv2DBackpropFilter????!??!m=?2?a??0"b
7gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropInputConv2DBackpropInput?X8E???!?H??b??0"1
model/conv1d_8/conv1dConv2D??zwQ??!B???:???"1
model/conv1d_6/conv1dConv2D??N?<???!\??3q???"2
model/conv1d_10/conv1dConv2Dʁ?qߑ?!?	NMh???Q      Y@Yx4?)?5@a\???S>X@q????`??y?PϬ#=v?"?

host?Your program is HIGHLY input-bound because 31.5% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?16.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 