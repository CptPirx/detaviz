?	|?o?^?O@|?o?^?O@!|?o?^?O@	n??43@n??43@!n??43@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6|?o?^?O@J
,?)???1h?.?K?C@A????i??I??H?E%@Y??ص?=(@*	㥛Đg?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator?????+@!???K3?X@)?????+@1???K3?X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism3?}ƅ,@!N\?<??X@)+?@.q???1?>7?6???:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap??OU??+@!?@[???X@)???o^???1@?i????:Preprocessing2F
Iterator::Model??g?,@!      Y@)zȔA՘?1?cG????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 19.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?16.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9o??43@I?\???3@Q??rif?N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	J
,?)???J
,?)???!J
,?)???      ??!       "	h?.?K?C@h?.?K?C@!h?.?K?C@*      ??!       2	????i??????i??!????i??:	??H?E%@??H?E%@!??H?E%@B      ??!       J	??ص?=(@??ص?=(@!??ص?=(@R      ??!       Z	??ص?=(@??ص?=(@!??ص?=(@b      ??!       JGPUYo??43@b q?\???3@y??rif?N@?"-
IteratorGetNext/_1_Send??<?D???!??<?D???"d
8gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?X??>??!F?a???0"b
6gradient_tape/model/conv1d/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?r&?#??!?&A:?V??0"d
8gradient_tape/model/conv1d_4/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??}?.̚?!OݰK???0"d
8gradient_tape/model/conv1d_5/conv1d/Conv2DBackpropFilterConv2DBackpropFilterA???$|??!????ϟ??0"d
8gradient_tape/model/conv1d_9/conv1d/Conv2DBackpropFilterConv2DBackpropFilter:n??Y??!?N ???0"b
7gradient_tape/model/conv1d_8/conv1d/Conv2DBackpropInputConv2DBackpropInput?X??:=??!.??|?r??0"1
model/conv1d_8/conv1dConv2Du1{Y????!.???????"1
model/conv1d_5/conv1dConv2D/D?J????!q(?? ???"2
model/conv1d_10/conv1dConv2DJ{?t??!?Z K??Q      Y@Y?(5?0@a?VNx??X@q???=????yW????u?"?

both?Your program is MODERATELY input-bound because 19.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?16.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 