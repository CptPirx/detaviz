	.?R?e@.?R?e@!.?R?e@	??f,o%@??f,o%@!??f,o%@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6.?R?e@P÷?n@1?$@M?]c@A?1 Ǟ??I(????,@YM?d??G@*	???M?Q?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator>+N??)@!??f??X@)>+N??)@1??f??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismkH?c??)@!?mp??X@)#gaO;???1ϰ 3??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap??4F??)@!F]??c?X@)s?"?k??1??~????:Preprocessing2F
Iterator::Modeld??A?)@!      Y@)j.7값?1K?$???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??f,o%@Id6?@Q/+6?|?V@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	P÷?n@P÷?n@!P÷?n@      ??!       "	?$@M?]c@?$@M?]c@!?$@M?]c@*      ??!       2	?1 Ǟ???1 Ǟ??!?1 Ǟ??:	(????,@(????,@!(????,@B      ??!       J	M?d??G@M?d??G@!M?d??G@R      ??!       Z	M?d??G@M?d??G@!M?d??G@b      ??!       JGPUY??f,o%@b qd6?@y/+6?|?V@