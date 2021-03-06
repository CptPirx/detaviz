?	?]J]2?T@?]J]2?T@!?]J]2?T@	?E?΁?!@?E?΁?!@!?E?΁?!@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?]J]2?T@Z??m?@1?%??P@A?3??E`??IU?W??@YX?vMh@*	???ƛ??@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator ?? ?"@!???B>?X@) ?? ?"@1???B>?X@:Preprocessing2F
Iterator::Model?&?%?6"@!      Y@)u?????1}$@?S??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?d#?+"@!?_1s??X@)oc?#?w??1??_????:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap?w?-;$"@!??b?X@)J????1????0???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.moderate"?6.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?E?΁?!@I?ԩݶ?"@Q?|x?XiT@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Z??m?@Z??m?@!Z??m?@      ??!       "	?%??P@?%??P@!?%??P@*      ??!       2	?3??E`???3??E`??!?3??E`??:	U?W??@U?W??@!U?W??@B      ??!       J	X?vMh@X?vMh@!X?vMh@R      ??!       Z	X?vMh@X?vMh@!X?vMh@b      ??!       JGPUY?E?΁?!@b q?ԩݶ?"@y?|x?XiT@?"d
8gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropFilterConv2DBackpropFilter,?TYA??!,?TYA??0"d
8gradient_tape/model/conv1d_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilterg??3?0??!`?Bs???0"d
8gradient_tape/model/conv1d_9/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??k????!\???|־?0"d
8gradient_tape/model/conv1d_5/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?^+#ߚ?!	???"???0"b
7gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropInputConv2DBackpropInput????Ֆ?!???ޡ??0"1
model/conv1d_8/conv1dConv2D??R6?n??!|&ϋ?o??"1
model/conv1d_6/conv1dConv2D??a$V??![[0r???"2
model/conv1d_10/conv1dConv2DQGjN??!4E?}5???"1
model/conv1d_5/conv1dConv2Dt??	Y??!b??>Y???"1
model/conv1d_9/conv1dConv2D?-??9??!0??E??Q      Y@Y?B??7	@a굨?G6X@qy-?oS??yN2E???h?"?

both?Your program is MODERATELY input-bound because 8.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?6.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 