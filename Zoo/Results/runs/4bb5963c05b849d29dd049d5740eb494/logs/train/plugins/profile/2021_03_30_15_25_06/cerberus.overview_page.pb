?	a????/N@a????/N@!a????/N@	??7?w?@??7?w?@!??7?w?@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6a????/N@Χ?UJ???19a?hV C@A??1 Ǯ?I??/?l1@Y?)Wx?+@*	 ?rhQ??@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator???.4'@!t??X@)???.4'@1t??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism1???Z@!B?4??X@)E.8??_??1???+????:Preprocessing2F
Iterator::Model???m@!      Y@)?wak????1??????:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap?D??22@!ud?V[?X@)?l?????1???Ҫ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?28.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??7?w?@I?"? @@Qu?V?9?O@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Χ?UJ???Χ?UJ???!Χ?UJ???      ??!       "	9a?hV C@9a?hV C@!9a?hV C@*      ??!       2	??1 Ǯ???1 Ǯ?!??1 Ǯ?:	??/?l1@??/?l1@!??/?l1@B      ??!       J	?)Wx?+@?)Wx?+@!?)Wx?+@R      ??!       Z	?)Wx?+@?)Wx?+@!?)Wx?+@b      ??!       JGPUY??7?w?@b q?"? @@yu?V?9?O@?"d
8gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?? ?.H??!?? ?.H??0"d
8gradient_tape/model/conv1d_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter~ ?????!??????0"d
8gradient_tape/model/conv1d_9/conv1d/Conv2DBackpropFilterConv2DBackpropFilterSj?????!???8Bҿ?0"d
8gradient_tape/model/conv1d_5/conv1d/Conv2DBackpropFilterConv2DBackpropFilter4?b]???!n???̆??0"b
7gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropInputConv2DBackpropInput?vnڱɚ?!L????0"1
model/conv1d_8/conv1dConv2D>??5????!4??J8???"2
model/conv1d_10/conv1dConv2D?\*????!???<?R??"1
model/conv1d_6/conv1dConv2D7??ܰ???!lZ'Xk???"1
model/conv1d_9/conv1dConv2Dt"`???!]??;g???"1
model/conv1d_5/conv1dConv2D?a?`W???!|???????Q      Y@Y05mvq`@aV?Lt?<X@q=K?|?5??y????v5v?"?

both?Your program is POTENTIALLY input-bound because 3.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?28.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 