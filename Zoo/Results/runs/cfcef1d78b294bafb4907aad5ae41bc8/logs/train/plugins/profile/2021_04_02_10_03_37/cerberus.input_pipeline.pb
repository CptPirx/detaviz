	?Z??Bq@?Z??Bq@!?Z??Bq@	???x???@???x???@!???x???@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?Z??Bq@m???e@1?P???df@A5_%???I?q??rW@Y??̔??U@*	?z??+?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::GeneratorkQL޳W@!??`?g?X@)kQL޳W@1??`?g?X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???c?W@!????+?X@)NG 7???1?{??????:Preprocessing2F
Iterator::Model?ϸp ?W@!      Y@)J?>?ɛ?1?9??sG??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap?~? ?W@!?????X@)????b)??1 4??"??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 31.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9???x???@I 6??t@Q???H?7P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	m???e@m???e@!m???e@      ??!       "	?P???df@?P???df@!?P???df@*      ??!       2	5_%???5_%???!5_%???:	?q??rW@?q??rW@!?q??rW@B      ??!       J	??̔??U@??̔??U@!??̔??U@R      ??!       Z	??̔??U@??̔??U@!??̔??U@b      ??!       JGPUY???x???@b q 6??t@y???H?7P@