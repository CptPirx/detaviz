?	x?~?~?W@x?~?~?W@!x?~?~?W@	ȶ|??S@ȶ|??S@!ȶ|??S@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6x?~?~?W@?`?????1`X?|[ *@A_\??ט?I???@YԂ}?R@*	??S?-??@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generatorۧ?1?R@!??֩??X@)ۧ?1?R@1??֩??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism*Ŏơ?R@!?L????X@) ???jw?1FO???~?:Preprocessing2F
Iterator::ModelIZ???R@!      Y@)ђ???w?1R{?lYU~?:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap+øD?R@!jx7?X@)???Fu:p?16P0Ihcu?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 78.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.moderate"?7.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9ȶ|??S@IRWۿn@Q??1?)+@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?`??????`?????!?`?????      ??!       "	`X?|[ *@`X?|[ *@!`X?|[ *@*      ??!       2	_\??ט?_\??ט?!_\??ט?:	???@???@!???@B      ??!       J	Ԃ}?R@Ԃ}?R@!Ԃ}?R@R      ??!       Z	Ԃ}?R@Ԃ}?R@!Ԃ}?R@b      ??!       JGPUYȶ|??S@b qRWۿn@y??1?)+@?"-
IteratorGetNext/_1_Send?K????!?K????"D
&gradient_tape/model/bl_1/MatMul/MatMulMatMul?????q??!<tZ?2|??0"/
model/bl_1/MatMulMatMul?g?]n???!??2?????0"3
model/bl_1/transpose	Transpose??I?ܡ??!1H-y????"F
*gradient_tape/model/bl_1/MatMul_1/MatMul_1MatMulh$??z6??!???c???"F
(gradient_tape/model/bl_1/MatMul_1/MatMulMatMuljOL????!& 1~{??0"M
.gradient_tape/model/bl_1/transpose_2/transpose	Transpose???}?!YM|?????"5
model/bl_1/transpose_2	Transpose????4?|?!,Y#????"F
*gradient_tape/model/bl_2/MatMul_1/MatMul_1MatMulu*g?޴{?!?'v?K&??"/
model/bl_1/MatMul_1MatMul	??M+gz?!ݪ&[??Q      Y@Y|t???G'@ap???V@q?F+66O??y?? jr??"?

host?Your program is HIGHLY input-bound because 78.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?7.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 