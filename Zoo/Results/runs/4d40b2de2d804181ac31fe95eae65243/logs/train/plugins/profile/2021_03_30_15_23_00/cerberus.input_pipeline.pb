	n??yfS@n??yfS@!n??yfS@	|`H?|?@|`H?|?@!|`H?|?@"w
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
	3???? @3???? @!3???? @      ??!       "	si??+C@si??+C@!si??+C@*      ??!       2	?i???1???i???1??!?i???1??:	?&P??)@?&P??)@!?&P??)@B      ??!       J	?P??n8@?P??n8@!?P??n8@R      ??!       Z	?P??n8@?P??n8@!?P??n8@b      ??!       JGPUY}`H?|?@b q\?~?2o3@y?U?W?H@