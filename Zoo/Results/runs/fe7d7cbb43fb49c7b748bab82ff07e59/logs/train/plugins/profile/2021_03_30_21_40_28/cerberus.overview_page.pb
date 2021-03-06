?	PVW^@PVW^@!PVW^@	PE	??&@@PE	??&@@!PE	??&@@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6PVW^@?&??ۋ@1?x??[SR@A?u?ݑ???IBB?/hQ@Y?#DiC@*	i??|???@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator7??$D@!?o8??X@)7??$D@1?o8??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????C'D@!??????X@)U?W????1mz xr??:Preprocessing2F
Iterator::Model?@???(D@!      Y@)e??Q??1?f???Q??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap-?\o?%D@!?d?|??X@)ǁW˝??1?-w?Cğ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 32.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.moderate"?4.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9PE	??&@@I?>?L??@Q?`??N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?&??ۋ@?&??ۋ@!?&??ۋ@      ??!       "	?x??[SR@?x??[SR@!?x??[SR@*      ??!       2	?u?ݑ????u?ݑ???!?u?ݑ???:	BB?/hQ@BB?/hQ@!BB?/hQ@B      ??!       J	?#DiC@?#DiC@!?#DiC@R      ??!       Z	?#DiC@?#DiC@!?#DiC@b      ??!       JGPUYPE	??&@@b q?>?L??@y?`??N@?"d
8gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??O?i??!??O?i??0"-
IteratorGetNext/_1_SendS8??a??!???????"b
6gradient_tape/model/conv1d/conv1d/Conv2DBackpropFilterConv2DBackpropFilterЖ?????!??
$]??0"d
8gradient_tape/model/conv1d_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?At7???!?7????0"d
8gradient_tape/model/conv1d_5/conv1d/Conv2DBackpropFilterConv2DBackpropFilternd+Ř?!????L-??0"d
8gradient_tape/model/conv1d_9/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??/?X???!*????B??0"b
7gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropInputConv2DBackpropInput?n?)?ݔ?!	a?m???0"1
model/conv1d_8/conv1dConv2DZٕG???!???7??"2
model/conv1d_10/conv1dConv2D??? ???!???	Kb??"1
model/conv1d_6/conv1dConv2DưY?6???!?_x????Q      Y@Y?t??-	@a9_4Ñ6X@q?????y??yY%???8q?"?

host?Your program is HIGHLY input-bound because 32.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 