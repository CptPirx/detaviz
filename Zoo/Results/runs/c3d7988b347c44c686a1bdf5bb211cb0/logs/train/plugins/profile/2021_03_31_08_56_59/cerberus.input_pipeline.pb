	??ٮP?g@??ٮP?g@!??ٮP?g@	N????TB@N????TB@!N????TB@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??ٮP?g@l\??O@1?e3??[@A%?)? ???I'?5??@Y'?_qQ@*	??~j?S?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator?DJ?y?Q@!?67?X@)?DJ?y?Q@1?67?X@:Preprocessing2F
Iterator::Model?~k'J?Q@!      Y@)l|&??i??1~?g ??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap?{??P?Q@!?H??X@)??n??o??1???Ｄ?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelisml?<?Q@!>??X@)?1?3/???1??s?_͔?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 36.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9N????TB@Ix?x???@Q?EqNZM@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	l\??O@l\??O@!l\??O@      ??!       "	?e3??[@?e3??[@!?e3??[@*      ??!       2	%?)? ???%?)? ???!%?)? ???:	'?5??@'?5??@!'?5??@B      ??!       J	'?_qQ@'?_qQ@!'?_qQ@R      ??!       Z	'?_qQ@'?_qQ@!'?_qQ@b      ??!       JGPUYN????TB@b qx?x???@y?EqNZM@