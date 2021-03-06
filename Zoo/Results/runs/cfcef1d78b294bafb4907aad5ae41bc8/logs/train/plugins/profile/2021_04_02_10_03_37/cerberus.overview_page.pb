?	?Z??Bq@?Z??Bq@!?Z??Bq@	???x???@???x???@!???x???@"w
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
	m???e@m???e@!m???e@      ??!       "	?P???df@?P???df@!?P???df@*      ??!       2	5_%???5_%???!5_%???:	?q??rW@?q??rW@!?q??rW@B      ??!       J	??̔??U@??̔??U@!??̔??U@R      ??!       Z	??̔??U@??̔??U@!??̔??U@b      ??!       JGPUY???x???@b q 6??t@y???H?7P@?"-
IteratorGetNext/_1_Send??M?Zյ?!??M?Zյ?"d
8gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropFilterConv2DBackpropFilterI	҉?6??!?wZQ???0"d
8gradient_tape/model/conv1d_9/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?Fo???!?`??I??0"d
8gradient_tape/model/conv1d_5/conv1d/Conv2DBackpropFilterConv2DBackpropFilterݗ5??!?҅????0"d
8gradient_tape/model/conv1d_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??&Ad??!???,???0"b
6gradient_tape/model/conv1d/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?ם?]
??!??Xx???0"1
model/conv1d_8/conv1dConv2D˂eӠ??!???9VE??"b
7gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropInputConv2DBackpropInput??k????!m?1H???0"/
model/conv1d/conv1dConv2D.?&x6???!?D?????"d
8gradient_tape/model/conv1d_6/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??+??#??!????????0Q      Y@Y_ζ ??@a?I?O?;X@qHv??@J??yPLXW?R?"?	
host?Your program is HIGHLY input-bound because 31.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 