	?3??3o@?3??3o@!?3??3o@	s?Vn?C@s?Vn?C@!s?Vn?C@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?3??3o@???}??@1???8b@A??B=}??I?F?0}o@Y?w???W@*	X9??X??@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator<p??oX@!????X@)<p??oX@1????X@:Preprocessing2F
Iterator::Modelb,?/tX@!      Y@)?	??Y??1~?m????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??(?[rX@!$I2?@?X@)?zO崧??19v?p???:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap?v?qX@!?9????X@)????ԑ?1))??:??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 38.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9t?Vn?C@I`??y?
@Q?????3M@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???}??@???}??@!???}??@      ??!       "	???8b@???8b@!???8b@*      ??!       2	??B=}????B=}??!??B=}??:	?F?0}o@?F?0}o@!?F?0}o@B      ??!       J	?w???W@?w???W@!?w???W@R      ??!       Z	?w???W@?w???W@!?w???W@b      ??!       JGPUYt?Vn?C@b q`??y?
@y?????3M@