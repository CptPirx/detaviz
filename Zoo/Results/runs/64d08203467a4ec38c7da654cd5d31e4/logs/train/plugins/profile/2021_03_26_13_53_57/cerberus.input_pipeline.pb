	c????6s@c????6s@!c????6s@	%?O?MD@%?O?MD@!%?O?MD@"w
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
	??M?@??M?@!??M?@      ??!       "	?I?2!f@?I?2!f@!?I?2!f@*      ??!       2	 ?g????? ?g?????! ?g?????:	ܝ??.???ܝ??.???!ܝ??.???B      ??!       J	Z/?r?5_@Z/?r?5_@!Z/?r?5_@R      ??!       Z	Z/?r?5_@Z/?r?5_@!Z/?r?5_@b      ??!       JGPUY$?O?MD@b q?o?f???y]O???L@