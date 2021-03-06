?	?w(
?/b@?w(
?/b@!?w(
?/b@	?ҙ??j@?ҙ??j@!?ҙ??j@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?w(
?/b@GN??@17?Ӂ,+`@Ax??Dg???I??<??@Y?lɪO @*	???(<??@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator??g???#@!?LaUa?X@)??g???#@1?LaUa?X@:Preprocessing2F
Iterator::ModelQ?O?I$@!      Y@)~8H????1o?(n?!??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMapX??V??#@!?f?L?X@)$a?N"?1?k4<???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism5|???#@!\ב[??X@)6t??Pn??1?$?"??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.moderate"?3.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?ҙ??j@IЁ???@Q?^??9V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	GN??@GN??@!GN??@      ??!       "	7?Ӂ,+`@7?Ӂ,+`@!7?Ӂ,+`@*      ??!       2	x??Dg???x??Dg???!x??Dg???:	??<??@??<??@!??<??@B      ??!       J	?lɪO @?lɪO @!?lɪO @R      ??!       Z	?lɪO @?lɪO @!?lɪO @b      ??!       JGPUY?ҙ??j@b qЁ???@y?^??9V@?"d
8gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropFilterConv2DBackpropFilteri+.Jȯ?!i+.Jȯ?0"d
8gradient_tape/model/conv1d_4/conv1d/Conv2DBackpropFilterConv2DBackpropFiltera 7????!?1????0"d
8gradient_tape/model/conv1d_9/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?E?+i??!?~	ꇾ?0"d
8gradient_tape/model/conv1d_5/conv1d/Conv2DBackpropFilterConv2DBackpropFiltero??#bV??!???H????0"b
7gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropInputConv2DBackpropInputF[??G˖?!s=*h??0"1
model/conv1d_8/conv1dConv2D?&?̏???!?"7?;??"1
model/conv1d_6/conv1dConv2D?^M???!j??Ō??"2
model/conv1d_10/conv1dConv2D??????!K3?4????"1
model/conv1d_9/conv1dConv2D??$?o???!??+??"1
model/conv1d_5/conv1dConv2D?ZKZ!???!?d?;???Q      Y@YD!????@a?6?C!9X@q?!?2?i??y??T?ـc?"?

both?Your program is MODERATELY input-bound because 5.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 