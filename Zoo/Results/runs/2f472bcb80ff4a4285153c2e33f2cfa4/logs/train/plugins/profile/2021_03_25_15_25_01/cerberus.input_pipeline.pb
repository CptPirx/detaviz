	Yk(?'I@Yk(?'I@!Yk(?'I@	?7~q?S@?7~q?S@!?7~q?S@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6Yk(?'I@??j??G??1??x??@A??????I?/???@Y???
?C@*	V-?\?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator???<H?C@!??wv??X@)???<H?C@1??wv??X@:Preprocessing2F
Iterator::ModelRb??v?C@!      Y@)??@???x?10?y p???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismLP÷??C@!0??X@)v?1<??x?1????P&??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap?n??C@!?<???X@)]?`7l[t?1Dd?Qp???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 78.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.moderate"?6.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?7~q?S@I?r?w?@Qm?n??+@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??j??G????j??G??!??j??G??      ??!       "	??x??@??x??@!??x??@*      ??!       2	????????????!??????:	?/???@?/???@!?/???@B      ??!       J	???
?C@???
?C@!???
?C@R      ??!       Z	???
?C@???
?C@!???
?C@b      ??!       JGPUY?7~q?S@b q?r?w?@ym?n??+@