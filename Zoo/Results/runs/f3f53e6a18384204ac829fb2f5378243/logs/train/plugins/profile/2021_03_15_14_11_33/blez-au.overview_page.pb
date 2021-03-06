?	Cs?FZB$@Cs?FZB$@!Cs?FZB$@	???(cI@???(cI@!???(cI@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6Cs?FZB$@?g?????1L???ԫ@A??,??Ή?Ivnڌ?P??Y?@?ش?@*	????G?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator5?ׂ޻@!?(T???X@)5?ׂ޻@1?(T???X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?p̲'?@!Χ?5?X@)??#?Gk?1Du??k??:Preprocessing2F
Iterator::Model,.??M?@!      Y@)?(??0i?1(??zGS??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap?H¾?@!r?	??X@)?z??^?1?Mt?%??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 50.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.moderate"?13.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2s6.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9???(cI@I??ZX?4@QU=?)?5=@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?g??????g?????!?g?????      ??!       "	L???ԫ@L???ԫ@!L???ԫ@*      ??!       2	??,??Ή???,??Ή?!??,??Ή?:	vnڌ?P??vnڌ?P??!vnڌ?P??B      ??!       J	?@?ش?@?@?ش?@!?@?ش?@R      ??!       Z	?@?ش?@?@?ش?@!?@?ش?@b      ??!       JGPUY???(cI@b q??ZX?4@yU=?)?5=@?"-
IteratorGetNext/_1_Send??Z.??!??Z.??"D
&gradient_tape/model/bl_1/MatMul/MatMulMatMul?Q^??a??!]?2?c???0"/
model/bl_1/MatMulMatMuln?z??!?7?7???0"F
*gradient_tape/model/bl_1/MatMul_1/MatMul_1MatMul???????!{?΃k???"F
(gradient_tape/model/bl_1/MatMul_1/MatMulMatMul???x1??!ckI????0"F
*gradient_tape/model/bl_2/MatMul_1/MatMul_1MatMulS?-????!?0%7L???"/
model/bl_1/MatMul_1MatMul&??????!%&?????"5
model/bl_1/transpose_2	Transposei~?u??!?H?????"M
.gradient_tape/model/bl_1/transpose_2/transpose	Transpose:=Q???!jW0EM??"3
model/bl_1/transpose	Transpose???2
`??!'lǖ???Q      Y@Y|t???G'@ap???V@qA?c?????yM	?1??"?
host?Your program is HIGHLY input-bound because 50.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?13.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"s6.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 