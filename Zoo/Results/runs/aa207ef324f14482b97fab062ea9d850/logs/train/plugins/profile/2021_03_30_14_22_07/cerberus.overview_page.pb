?	?ZD??%@?ZD??%@!?ZD??%@	?Qb?CA@?Qb?CA@!?Qb?CA@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?ZD??%@p???D??19'0????A?u??ݰ??I3?<F?@Y<g??@*	w???Y?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::GeneratoraQ??@!B>J	??X@)aQ??@1B>J	??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????#
@!?2????X@)??ĭ?x?1y?b>???:Preprocessing2F
Iterator::ModelYP?i@!      Y@)h?N???t?1?X??Ɔ??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMapDi??@!m?}???X@)??ŉ?vt?1-V?f?u??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 34.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.high"?40.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2t14.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?Qb?CA@IT??c?K@QB?fW?F%@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	p???D??p???D??!p???D??      ??!       "	9'0????9'0????!9'0????*      ??!       2	?u??ݰ???u??ݰ??!?u??ݰ??:	3?<F?@3?<F?@!3?<F?@B      ??!       J	<g??@<g??@!<g??@R      ??!       Z	<g??@<g??@!<g??@b      ??!       JGPUY?Qb?CA@b qT??c?K@yB?fW?F%@?"F
*gradient_tape/model/bl_1/MatMul_1/MatMul_1MatMul?nȬ???!?nȬ???"-
IteratorGetNext/_1_Send?q(??}??!8p?#???"F
*gradient_tape/model/bl_2/MatMul_1/MatMul_1MatMulj???6??!vk??1W??"F
*gradient_tape/model/tabl/MatMul_1/MatMul_1MatMul????Aۣ?!??~'????"D
&gradient_tape/model/bl_1/MatMul/MatMulMatMuli4ֽ????!=q9??"??0"F
(gradient_tape/model/bl_1/MatMul_1/MatMulMatMul??PA??![?<?Wd??0"/
model/bl_1/MatMul_1MatMul?(?3?	??!mF?ψ???"D
(gradient_tape/model/bl_2/MatMul/MatMul_1MatMul?kk????!,?:?V??"/
model/bl_1/MatMulMatMulc1g?E??!"?????0"/
model/bl_2/MatMul_1MatMul]?F??!?>?8???Q      Y@Y|t???G'@ap???V@q??_䚮@yr#c'R???"?
host?Your program is HIGHLY input-bound because 34.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?40.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"t14.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 