
Z
sgemm_32x32x32_TN_vec*28֐�@��H��Xb&gradient_tape/model/bl_1/MatMul/MatMulh
E
sgemm_32x32x32_NN_vec*28���@��H��$Xbmodel/bl_1/MatMulh
�
W_Z18sgemm_largek_lds64ILb0ELb1ELi5ELi5ELi4ELi4ELi4ELi32EEvPfPKfS2_iiiiiiS2_S2_ffiiPiS3_*28���@��H��b*gradient_tape/model/bl_1/MatMul_1/MatMul_1h
X
sgemm_32x32x32_NT*28�@��H��Xb(gradient_tape/model/bl_1/MatMul_1/MatMulh
�
t_ZN10tensorflow7functor37SwapDimension1And2InTensor3UsingTilesIjLi256ELi32ELi32ELb0EEEvPKT_NS0_9DimensionILi3EEEPS2_*28���@��H��bmodel/bl_1/transposeh
�
W_Z18sgemm_largek_lds64ILb0ELb1ELi6ELi3ELi4ELi5ELi2ELi64EEvPfPKfS2_iiiiiiS2_S2_ffiiPiS3_*28��w@��H��b*gradient_tape/model/bl_2/MatMul_1/MatMul_1h
@
sgemm_32x32x32_NN*28��q@��H��bmodel/bl_1/MatMul_1h
�
t_ZN10tensorflow7functor37SwapDimension1And2InTensor3UsingTilesIjLi256ELi32ELi32ELb0EEEvPKT_NS0_9DimensionILi3EEEPS2_*28��g@��H��bmodel/bl_1/transpose_2h
�
t_ZN10tensorflow7functor37SwapDimension1And2InTensor3UsingTilesIjLi256ELi32ELi32ELb0EEEvPKT_NS0_9DimensionILi3EEEPS2_*28��V@��H��b.gradient_tape/model/bl_1/transpose_2/transposeh
Y
sgemm_32x32x32_TN_vec*28��+@��H��b(gradient_tape/model/bl_2/MatMul/MatMul_1h
@
sgemm_32x32x32_NN*28��)@��H��bmodel/bl_2/MatMul_1h
W
sgemm_32x32x32_NT*28��&@��H��Xb(gradient_tape/model/bl_2/MatMul_1/MatMulh
�
w_Z13gemv2N_kernelIiifffLi128ELi32ELi4ELi4ELi1ELb0E16cublasGemvParamsI30cublasGemvTensorStridedBatchedIKfES1_IfEfEEvT10_*28�� @��H��b*gradient_tape/model/bl_3/MatMul_1/MatMul_1h
T
sgemm_32x32x32_TN*28��@�xH��b(gradient_tape/model/bl_3/MatMul/MatMul_1h
C
sgemm_32x32x32_NN_vec*28×@�xH��Xbmodel/bl_2/MatMulh
X
sgemm_32x32x32_NT_vec*28��@�xH��Xb&gradient_tape/model/bl_2/MatMul/MatMulh
T
sgemm_32x32x32_TN*28��@�hH��b(gradient_tape/model/tabl/MatMul/MatMul_1h
�
�_Z17gemv2T_kernel_valIiifffLi128ELi16ELi2ELi2ELb0ELb0E16cublasGemvParamsI30cublasGemvTensorStridedBatchedIKfES1_IfEfEEvT10_T3_S7_*28��@�XH��bmodel/bl_3/MatMul_1h
�
r_ZN10tensorflow7functor37SwapDimension1And2InTensor3UsingTilesIjLi32ELi5ELi32ELb0EEEvPKT_NS0_9DimensionILi3EEEPS2_*28��@�XH��b,gradient_tape/model/bl_2/transpose/transposeh
T
sgemm_32x32x32_NT*28��@�PH��Xb&gradient_tape/model/bl_3/MatMul/MatMulh
�
r_ZN10tensorflow7functor37SwapDimension1And2InTensor3UsingTilesIjLi32ELi32ELi5ELb0EEEvPKT_NS0_9DimensionILi3EEEPS2_*28��@�PH�xbmodel/bl_2/transposeh
�
�_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvS4_PNT_17ResultElementTypeExS6_*28��@�@H��b4model/dropout_1/dropout/random_uniform/RandomUniformh
W
sgemm_32x32x32_NT_vec*28��@�HH�PXb&gradient_tape/model/tabl/MatMul/MatMulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIffEEKNS4_INS5_IKfLi2ELi1EiEELi16ES7_EEKNS_20TensorBroadcastingOpIKNS_5arrayIxLy2EEESF_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��
@�0H��bmodel/bl_1/addh
�
�_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvS4_PNT_17ResultElementTypeExS6_*28��	@�0H��b4model/dropout_2/dropout/random_uniform/RandomUniformh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIbfNS0_13greater_equalIfEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��	@�8H�@b$model/dropout_1/dropout/GreaterEqualh
�
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIjLi0ELi2ELi1ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*28��@�/H��bmodel/bl_2/transpose_2h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_20scalar_difference_opIffEEKNS9_INS0_18scalar_quotient_opIffEEKNS_18TensorCwiseUnaryOpINS0_13scalar_exp_opIfEEKS8_EEKNS_20TensorBroadcastingOpIKNS_5arrayIiLy2EEESH_EEEEKNSK_IKNSL_IxLy2EEEKNS4_INS5_IKfLi2ELi1ExEELi16ES7_EEEEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�0H�xb:categorical_crossentropy/softmax_cross_entropy_with_logitsh
�
�_ZN10tensorflow7functor28FillPhiloxRandomKernelLaunchINS_6random19UniformDistributionINS2_12PhiloxRandomEfEEEEvS4_PNT_17ResultElementTypeExS6_*28��@�0H�`b4model/dropout_3/dropout/random_uniform/RandomUniformh
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28��@�0H�8b"Adam/Adam/update/ResourceApplyAdamh
?
sgemm_32x32x32_NN*28��@�(H��Xbmodel/bl_3/MatMulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�(H�pb+gradient_tape/model/dropout_1/dropout/Mul_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�(H�hbmodel/dropout_1/dropout/Mul_1h
�
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIjLi0ELi2ELi1ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*28��@�(H��b.gradient_tape/model/bl_2/transpose_2/transposeh
�
�_Z13gemmk1_kernelIfLi256ELi5ELb1ELb0ELb0ELb0E30cublasGemvTensorStridedBatchedIKfES0_IfEfEv18cublasGemmk1ParamsIT_T6_T7_T8_N8biasTypeINS7_10value_typeES8_E4typeEE*28��@�(H�8Xb(gradient_tape/model/bl_3/MatMul_1/MatMulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi2ELi1EiEELi16ES7_EEKNS_20TensorBroadcastingOpIKNS_5arrayIxLy2EEESF_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�(H��bAdam/Adam/update_4/mulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_18TensorConversionOpIfKNS9_INS0_13scalar_cmp_opISB_SB_LNS0_14ComparisonNameE5EEESF_KNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�(H�0b)gradient_tape/model/activation_1/ReluGradh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_17TensorReshapingOpIKNS_5arrayIiLy1EEENS_9TensorMapINS_6TensorIfLi2ELi1ExEELi16ENS_11MakePointerEEEEEKNS_17TensorReductionOpINS0_10MaxReducerIfEES7_KNS_20TensorBroadcastingOpIKNS5_IxLy2EEEKNS8_INS9_IKfLi2ELi1ExEELi16ESB_EEEESB_EEEENS_9GpuDeviceEEExEEvT_T0_*28��@�(H�pb:categorical_crossentropy/softmax_cross_entropy_with_logitsh
�
|_ZN10tensorflow7functor30ColumnReduceMax16ColumnsKernelIPfS2_NS0_3SumIfEEEEvT_T0_iiT1_NSt15iterator_traitsIS5_E10value_typeE*28��@�(H�0bAdam/Adam/update_1/Sumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorEvalToOpIKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfS6_EEKNS_20TensorBroadcastingOpIKNS_5arrayIxLy2EEEKNS_9TensorMapINS_6TensorIS6_Li2ELi1ExEELi16ENS_11MakePointerEEEEEKNS4_INS0_20scalar_difference_opIffEEKNS8_IKNS9_IiLy2EEEKNS_18TensorForcedEvalOpIKNS_18TensorCwiseUnaryOpINS0_13scalar_log_opIfEEKNSC_INSD_IfLi2ELi1ExEELi16ESF_EEEEEEEESU_EEEESF_EENS_9GpuDeviceEEExEEvT_T0_*28�@� H��b:categorical_crossentropy/softmax_cross_entropy_with_logitsh
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28��@� H��b$Adam/Adam/update_2/ResourceApplyAdamh
�
�_Z20reduce_1Block_kernelIfLi128ELi7E30cublasGemvTensorStridedBatchedIfES1_EvPKT_S2_T2_iS4_S2_T3_19cublasPointerMode_t18cublasLtEpilogue_tS0_IKN8biasTypeINS6_10value_typeES2_E4typeEE*28��@� H��b*gradient_tape/model/tabl/MatMul_2/MatMul_1h
�
|_ZN10tensorflow7functor30ColumnReduceMax16ColumnsKernelIPfS2_NS0_3SumIfEEEEvT_T0_iiT1_NSt15iterator_traitsIS5_E10value_typeE*28��@�(H�Xb$gradient_tape/model/tabl/add_2/Sum_1h
>
sgemm_32x32x32_NN*28��@� H�PXbmodel/tabl/MatMulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_17TensorReductionOpINS0_10SumReducerIfEEKNS_5arrayIiLy1EEEKNS_18TensorForcedEvalOpIKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSI_EEKNS_20TensorBroadcastingOpIKNSC_IxLy2EEEKNS4_INS5_ISI_Li2ELi1ExEELi16ES7_EEEEKNSG_INS0_20scalar_difference_opIffEEKNSK_IKNSC_IiLy2EEEKNSF_IKNS_18TensorCwiseUnaryOpINS0_13scalar_log_opIfEEKNS4_INS5_IfLi2ELi1ExEELi16ES7_EEEEEEEES11_EEEEEES7_EEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H��b:categorical_crossentropy/softmax_cross_entropy_with_logitsh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_11scalar_leftIffNS0_17scalar_product_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@� H�(bmodel/dropout_1/dropout/Mulh
�
p_ZN10tensorflow7functor18ColumnReduceKernelIPfS2_NS0_3SumIfEEEEvT_T0_iiT1_NSt15iterator_traitsIS5_E10value_typeE*28��@� H�pb"gradient_tape/model/bl_1/add/Sum_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_11scalar_leftIffNS0_17scalar_product_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@� H�(b)gradient_tape/model/dropout_1/dropout/Mulh
�
\_Z10dot_kernelIfLi128ELi0E15cublasDotParamsI30cublasGemvTensorStridedBatchedIKfES1_IfEEEvT2_*28��@� H�hb*gradient_tape/model/tabl/MatMul_2/MatMul_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_pow_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�`b
Adam/Pow_1h
�
|_ZN10tensorflow7functor30ColumnReduceMax16ColumnsKernelIPfS2_NS0_3SumIfEEEEvT_T0_iiT1_NSt15iterator_traitsIS5_E10value_typeE*28��@� H�(bAdam/Adam/update_4/Sumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_pow_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�`bAdam/Powh
�
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIjLi0ELi2ELi1ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*28��@�H�hbmodel/bl_3/transposeh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi2ELi1EiEELi16ES7_EEKNS_20TensorBroadcastingOpIKNS_5arrayIxLy2EEESF_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@� H�`bAdam/Adam/update_3/mulh
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28��@� H�(b$Adam/Adam/update_3/ResourceApplyAdamh
�
n_ZN10tensorflow7functor15CleanupSegmentsIPfS2_NS0_3SumIfEEEEvT_T0_iiiT1_NSt15iterator_traitsIS5_E10value_typeE*28��@�H��bAdam/Adam/update_3/Sumh
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28��@� H�(b%Adam/Adam/update_13/ResourceApplyAdamh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_20scalar_difference_opIKfSB_EEKNS_20TensorBroadcastingOpIKNS_5arrayIxLy2EEEKNS4_INS5_ISB_Li2ELi1ExEELi16ES7_EEEEKNSD_IKNSE_IiLy2EEEKS8_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@� H�(b:categorical_crossentropy/softmax_cross_entropy_with_logitsh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_17TensorReshapingOpIKNS_5arrayIiLy1EEENS_9TensorMapINS_6TensorIfLi2ELi1ExEELi16ENS_11MakePointerEEEEEKNS_17TensorReductionOpINS0_10SumReducerIfEES7_KNS_18TensorCwiseUnaryOpINS0_13scalar_exp_opIfEEKSC_EESB_EEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�hb:categorical_crossentropy/softmax_cross_entropy_with_logitsh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_18scalar_quotient_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H��bAdam/Adam/update_3/truedivh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIffEEKNS4_INS5_IKfLi2ELi1EiEELi16ES7_EEKNS_20TensorBroadcastingOpIKNS_5arrayIxLy2EEESF_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@� H�(bmodel/bl_2/addh
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28��@� H�(b%Adam/Adam/update_12/ResourceApplyAdamh
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28��@�H�Pb%Adam/Adam/update_11/ResourceApplyAdamh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi2ELi1EiEELi16ES7_EEKNS_20TensorBroadcastingOpIKNS_5arrayIxLy2EEESF_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�`bLgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mulh
�
n_ZN10tensorflow7functor15CleanupSegmentsIPfS2_NS0_3SumIfEEEEvT_T0_iiiT1_NSt15iterator_traitsIS5_E10value_typeE*28��@�H�xbAdam/Adam/update/Sumh
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28��@�H� b$Adam/Adam/update_6/ResourceApplyAdamh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKbLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�Pbmodel/dropout_1/dropout/Casth
�
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIjLi0ELi2ELi1ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*28��@�H� b,gradient_tape/model/bl_3/transpose/transposeh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_20TensorBroadcastingOpIKNS_5arrayIiLy1EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�Xb;gradient_tape/categorical_crossentropy/weighted_loss/Tile_1h
�
�_Z13gemmk1_kernelIfLi256ELi5ELb1ELb0ELb0ELb0E30cublasGemvTensorStridedBatchedIKfES0_IfEfEv18cublasGemmk1ParamsIT_T6_T7_T8_N8biasTypeINS7_10value_typeES8_E4typeEE*28�@�H�(bmodel/tabl/MatMul_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_max_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H� bmodel/activation_1/Reluh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_11scalar_leftIffNS0_17scalar_product_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H��b)gradient_tape/model/dropout_3/dropout/Mulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi2ELi1EiEELi16ES7_EEKNS_20TensorBroadcastingOpIKNS_5arrayIxLy2EEESF_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�(bAdam/Adam/update/mulh
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28��@�H� b$Adam/Adam/update_5/ResourceApplyAdamh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorEvalToOpIKNS_18TensorCwiseUnaryOpINS0_13scalar_log_opIfEEKNS_9TensorMapINS_6TensorIfLi2ELi1ExEELi16ENS_11MakePointerEEEEESA_EENS_9GpuDeviceEEExEEvT_T0_*28��@�H�Xb:categorical_crossentropy/softmax_cross_entropy_with_logitsh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_18scalar_quotient_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�(b*gradient_tape/model/tabl/truediv_1/RealDivh
�
|_ZN10tensorflow7functor30ColumnReduceMax16ColumnsKernelIPfS2_NS0_3SumIfEEEEvT_T0_iiT1_NSt15iterator_traitsIS5_E10value_typeE*28��@�H�(bAdam/Adam/update_9/Sumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIbfNS0_13greater_equalIfEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�hb$model/dropout_2/dropout/GreaterEqualh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�hbEgradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nanh
�
p_ZN10tensorflow7functor18ColumnReduceKernelIPfS2_NS0_3SumIfEEEEvT_T0_iiT1_NSt15iterator_traitsIS5_E10value_typeE*28��@�H� b"gradient_tape/model/bl_3/add/Sum_1h
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28��@�H�(b$Adam/Adam/update_8/ResourceApplyAdamh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H��bMulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIffEEKNS4_INS5_IKfLi2ELi1EiEELi16ES7_EEKNS_20TensorBroadcastingOpIKNS_5arrayIxLy2EEESF_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H��bmodel/tabl/add_2h
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28��@�H� b$Adam/Adam/update_7/ResourceApplyAdamh
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28��@�H� b$Adam/Adam/update_1/ResourceApplyAdamh
�
n_ZN10tensorflow7functor15CleanupSegmentsIPfS2_NS0_3SumIfEEEEvT_T0_iiiT1_NSt15iterator_traitsIS5_E10value_typeE*28��@�H� b"gradient_tape/model/bl_2/add/Sum_1h
�
p_ZN10tensorflow7functor18ColumnReduceKernelIPfS2_NS0_3SumIfEEEEvT_T0_iiT1_NSt15iterator_traitsIS5_E10value_typeE*28��@�H� bAdam/Adam/update_3/Sumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_14scalar_sqrt_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�xbAdam/Adam/update_4/Sqrth
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_18scalar_quotient_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H��bAdam/Adam/update/truedivh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_14scalar_sqrt_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H��bAdam/Adam/update_6/Sqrth
�
\_Z10dot_kernelIfLi128ELi0E15cublasDotParamsI30cublasGemvTensorStridedBatchedIKfES1_IfEEEvT2_*28��@�H�'b*gradient_tape/model/tabl/MatMul_1/MatMul_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi2ELi1EiEELi16ES7_EEKNS_20TensorBroadcastingOpIKNS_5arrayIxLy2EEESF_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H� bAdam/Adam/update_9/mulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi2ELi1EiEELi16ES7_EEKNS_20TensorBroadcastingOpIKNS_5arrayIxLy2EEESF_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28�@�H� bAdam/Adam/update_6/mulh
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28��@�H� b$Adam/Adam/update_9/ResourceApplyAdamh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIffEEKNS4_INS5_IKfLi2ELi1EiEELi16ES7_EEKNS_20TensorBroadcastingOpIKNS_5arrayIxLy2EEESF_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H� bmodel/bl_3/addh
�
p_ZN10tensorflow7functor18ColumnReduceKernelIPfS2_NS0_3SumIfEEEEvT_T0_iiT1_NSt15iterator_traitsIS5_E10value_typeE*28��@�H� bAdam/Adam/update/Sumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_18scalar_quotient_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H� bAdam/Adam/update_1/truedivh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_18TensorConversionOpIfKNS9_INS0_13scalar_cmp_opISB_SB_LNS0_14ComparisonNameE5EEESF_KNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�Xb)gradient_tape/model/activation_2/ReluGradh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_16scalar_square_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�PbAdam/Adam/update_10/Squareh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�`b+gradient_tape/model/dropout_3/dropout/Mul_1h
�
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIjLi0ELi2ELi1ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*28��@�H� b.gradient_tape/model/bl_3/transpose_2/transposeh
�
n_ZN10tensorflow7functor15CleanupSegmentsIPfS2_NS0_3SumIfEEEEvT_T0_iiiT1_NSt15iterator_traitsIS5_E10value_typeE*28��@�H� b"gradient_tape/model/bl_3/add/Sum_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H��bgradient_tape/model/tabl/mul_1h
�
�_Z13gemmk1_kernelIfLi256ELi5ELb1ELb0ELb0ELb0E30cublasGemvTensorStridedBatchedIKfES0_IfEfEv18cublasGemmk1ParamsIT_T6_T7_T8_N8biasTypeINS7_10value_typeES8_E4typeEE*28��@�H� Xb(gradient_tape/model/tabl/MatMul_2/MatMulh
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28��@�H� b%Adam/Adam/update_10/ResourceApplyAdamh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_min_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�Pb(Adam/Adam/update_1/clip_by_value/Minimumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_18scalar_quotient_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H� b,gradient_tape/model/tabl/truediv_1/RealDiv_1h
�
t_ZN10tensorflow7functor17BlockReduceKernelIPfS2_Li256ENS0_3SumIfEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE*28��@�H� b"gradient_tape/model/tabl/mul_1/Sumh
�
O_ZN10tensorflow7functor15ApplyAdamKernelIfEEviPT_S3_S3_PKS2_S5_S5_S5_S5_S5_S5_b*28��@�H� b$Adam/Adam/update_4/ResourceApplyAdamh
�
t_ZN10tensorflow7functor17BlockReduceKernelIPfS2_Li256ENS0_3SumIfEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE*28��@�H� b*categorical_crossentropy/weighted_loss/Sumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_14scalar_sqrt_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H� bAdam/Adam/update_1/Sqrth
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_max_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�`b!Adam/Adam/update_10/clip_by_valueh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKiLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�XbCasth
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_min_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�`b(Adam/Adam/update_6/clip_by_value/Minimumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi2ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi2ELi1EiEELi16ES7_EEKNS_20TensorBroadcastingOpIKNS_5arrayIxLy2EEESF_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H� bAdam/Adam/update_1/mulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�XbAssignAddVariableOph
�
�_Z20reduce_1Block_kernelIfLi128ELi7E30cublasGemvTensorStridedBatchedIfES1_EvPKT_S2_T2_iS4_S2_T3_19cublasPointerMode_t18cublasLtEpilogue_tS0_IKN8biasTypeINS6_10value_typeES2_E4typeEE*28��@�H�(b*gradient_tape/model/tabl/MatMul_1/MatMul_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_18scalar_quotient_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H� bAdam/Adam/update_6/truedivh
�
p_ZN10tensorflow7functor18ColumnReduceKernelIPfS2_NS0_3SumIfEEEEvT_T0_iiT1_NSt15iterator_traitsIS5_E10value_typeE*28��@�H� bAdam/Adam/update_6/Sumh
�
p_ZN10tensorflow7functor18ColumnReduceKernelIPfS2_NS0_3SumIfEEEEvT_T0_iiT1_NSt15iterator_traitsIS5_E10value_typeE*28��@�H� b"gradient_tape/model/bl_2/add/Sum_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_18scalar_quotient_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H� bAdam/Adam/update_4/truedivh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�pb"gradient_tape/model/tabl/mul_1/Mulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_max_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�Pbmodel/activation_2/Reluh
�
t_ZN10tensorflow7functor17BlockReduceKernelIPfS2_Li256ENS0_3SumIfEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE*28��@�H�(b"gradient_tape/model/tabl/mul_2/Sumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_min_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H��b*Adam/Adam/update_1/clip_by_value_1/Minimumh
�
U_Z11scal_kernelIffLi1ELb1ELi6ELi5ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_*28��@�H�b*gradient_tape/model/bl_1/MatMul_1/MatMul_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�Xb
div_no_nanh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H� b+gradient_tape/model/dropout_2/dropout/Mul_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_18scalar_quotient_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�Pbmodel/tabl/truediv_1h
�
U_Z11scal_kernelIffLi1ELb1ELi6ELi5ELi5ELi3EEv21cublasTransposeParamsIT0_EPKT_PS3_PKS1_*28��@�H�b*gradient_tape/model/bl_2/MatMul_1/MatMul_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_sum_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bAdam/Adam/update/addh
�
b_ZN10tensorflow7functor22ShuffleInTensor3SimpleIjLi0ELi2ELi1ELb0EEEviPKT_NS0_9DimensionILi3EEEPS2_*28��@�H�bmodel/bl_3/transpose_2h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_14scalar_sqrt_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bAdam/Adam/update_10/Sqrth
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_13scalar_exp_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bmodel/tabl/Exph
�
�_Z13gemmk1_kernelIfLi256ELi5ELb1ELb0ELb0ELb0E30cublasGemvTensorStridedBatchedIKfES0_IfEfEv18cublasGemmk1ParamsIT_T6_T7_T8_N8biasTypeINS7_10value_typeES8_E4typeEE*28��@�H� bmodel/tabl/MatMul_2h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_sum_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bAdam/Adam/update_1/addh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_sum_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bAdam/Adam/update_6/addh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_14scalar_sqrt_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�HbAdam/Adam/update/Sqrth
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKbLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�Pbmodel/dropout_2/dropout/Casth
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�xbmodel/tabl/mul_3h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_11scalar_leftIffNS0_17scalar_product_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bmodel/dropout_2/dropout/Mulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_sum_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bAdam/Adam/update_10/addh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H��b"gradient_tape/model/tabl/mul_3/Mulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_20scalar_difference_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�bmodel/tabl/subh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_18scalar_quotient_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H� bAdam/Adam/update_11/truedivh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_18scalar_quotient_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�b gradient_tape/model/tabl/truedivh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�bmodel/dropout_2/dropout/Mul_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIxxEEKNS4_INS5_IKxLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�(bAdam/addh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_16scalar_square_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bAdam/Adam/update/Squareh
�
n_ZN10tensorflow7functor15CleanupSegmentsIPfS2_NS0_3SumIfEEEEvT_T0_iiiT1_NSt15iterator_traitsIS5_E10value_typeE*28��@�H�bAdam/Adam/update_6/Sumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_8equal_toIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�Xbgradient_tape/model/tabl/Equalh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_max_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b Adam/Adam/update_4/clip_by_valueh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_14scalar_sqrt_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bAdam/Adam/update_11/Sqrth
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_11scalar_leftIffNS0_17scalar_product_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b)gradient_tape/model/dropout_2/dropout/Mulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_11scalar_leftIffNS0_17scalar_product_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H� bmodel/dropout_3/dropout/Mulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�bmodel/tabl/addh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_18scalar_quotient_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�bAdam/Adam/update_7/truedivh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_min_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�Pb*Adam/Adam/update_9/clip_by_value_1/Minimumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�bmodel/dropout_3/dropout/Mul_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_min_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b)Adam/Adam/update_10/clip_by_value/Minimumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_11scalar_leftIffNS0_17scalar_product_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�Xbmodel/tabl/mul_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_11scalar_leftIffNS0_20scalar_difference_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bmodel/tabl/sub_2h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_18scalar_opposite_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b&gradient_tape/model/tabl/truediv_1/Negh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIffEEKNS9_INSA_IKfSC_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EESG_EESG_EEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bAdam/gradients/AddN_2h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EESF_EEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bAdam/gradients/AddNh
�
t_ZN10tensorflow7functor17BlockReduceKernelIPfS2_Li256ENS0_3SumIfEEEEvT_T0_iT2_NSt15iterator_traitsIS5_E10value_typeE*28��@�H�bAdam/Adam/update_7/Sumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_16scalar_square_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bAdam/Adam/update_1/Squareh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_max_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�Xb Adam/Adam/update_3/clip_by_valueh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_min_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�Xb+Adam/Adam/update_11/clip_by_value_1/Minimumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKiLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�b8categorical_crossentropy/weighted_loss/num_elements/Casth
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_sum_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bAdam/Adam/update_11/addh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_18scalar_opposite_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H� b"gradient_tape/model/tabl/sub_1/Negh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�bmodel/tabl/add_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_max_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28�@�H�Hb!Adam/Adam/update_11/clip_by_valueh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_16scalar_square_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28�@�H�PbAdam/Adam/update_11/Squareh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_17scalar_product_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�Pb$gradient_tape/model/tabl/mul_2/Mul_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_20scalar_difference_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�bmodel/tabl/sub_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_min_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b*Adam/Adam/update_3/clip_by_value_1/Minimumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_sum_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bAdam/Adam/update_7/addh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_14scalar_sqrt_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bAdam/Adam/update_9/Sqrth
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_min_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b*Adam/Adam/update_6/clip_by_value_1/Minimumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_sum_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bAdam/Adam/update_4/addh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_18scalar_opposite_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�Pb gradient_tape/model/tabl/sub/Negh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_max_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�bmodel/activation_3/Reluh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_max_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b!Adam/Adam/update_12/clip_by_valueh
�
�_Z13gemmk1_kernelIfLi256ELi5ELb1ELb0ELb0ELb0E30cublasGemvTensorStridedBatchedIKfES0_IfEfEv18cublasGemmk1ParamsIT_T6_T7_T8_N8biasTypeINS7_10value_typeES8_E4typeEE*28��@�H� Xb(gradient_tape/model/tabl/MatMul_1/MatMulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_14scalar_sqrt_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bAdam/Adam/update_3/Sqrth
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_sum_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bAdam/Adam/update_3/addh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_18scalar_quotient_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�b,gradient_tape/model/tabl/truediv_1/RealDiv_2h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKxLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�bAdam/Cast_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_14scalar_sqrt_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28�@�H�bAdam/Adam/update_7/Sqrth
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_max_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b Adam/Adam/update_1/clip_by_valueh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�bgradient_tape/model/tabl/mulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EESF_EEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H� bAdam/gradients/AddN_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_18scalar_quotient_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H� bAdam/Adam/update_9/truedivh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_max_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28�@�H�b Adam/Adam/update/clip_by_value_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_17scalar_product_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28�@�H�b$gradient_tape/model/tabl/mul_1/Mul_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_16scalar_square_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bAdam/Adam/update_3/Squareh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_max_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b"Adam/Adam/update_1/clip_by_value_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEKNS_18TensorConversionOpIfKNS9_INS0_13scalar_cmp_opISB_SB_LNS0_14ComparisonNameE5EEESF_KNS_20TensorCwiseNullaryOpINS0_18scalar_constant_opISB_EESF_EEEEEEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H� b)gradient_tape/model/activation_3/ReluGradh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_16scalar_square_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bAdam/Adam/update_4/Squareh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_min_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28�@�H�b(Adam/Adam/update_9/clip_by_value/Minimumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_16scalar_square_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28�@�H�bAdam/Adam/update_6/Squareh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�bAssignAddVariableOp_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_max_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b Adam/Adam/update_9/clip_by_valueh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_min_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b&Adam/Adam/update/clip_by_value/Minimumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13div_no_nan_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H� b,categorical_crossentropy/weighted_loss/valueh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�b"gradient_tape/model/tabl/mul_2/Mulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_18scalar_quotient_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�bAdam/Adam/update_10/truedivh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_max_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b"Adam/Adam/update_4/clip_by_value_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_min_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b(Adam/Adam/update_4/clip_by_value/Minimumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_max_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b#Adam/Adam/update_11/clip_by_value_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_max_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bAdam/Adam/update/clip_by_valueh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_max_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b"Adam/Adam/update_9/clip_by_value_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_min_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b(Adam/Adam/update/clip_by_value_1/Minimumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_min_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b)Adam/Adam/update_12/clip_by_value/Minimumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_min_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b(Adam/Adam/update_3/clip_by_value/Minimumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKxSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�bAssignAddVariableOp_2h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H� bAdam/Adam/update_10/mulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_17scalar_product_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bAdam/Adam/update_7/mulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKbLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�bmodel/dropout_3/dropout/Casth
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H� b&gradient_tape/model/tabl/truediv_1/mulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_min_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b+Adam/Adam/update_10/clip_by_value_1/Minimumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_min_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b)Adam/Adam/update_11/clip_by_value/Minimumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_18TensorConversionOpIfKNS4_INS5_IKbLi1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�bgradient_tape/model/tabl/Casth
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIbfNS0_13greater_equalIfEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b$model/dropout_3/dropout/GreaterEqualh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EESF_EEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�bAdam/gradients/AddN_4h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_11scalar_leftIffNS0_17scalar_product_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bmodel/tabl/mul_2h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_max_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b"Adam/Adam/update_6/clip_by_value_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_max_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b Adam/Adam/update_7/clip_by_valueh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_sum_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bAdam/Adam/update_9/addh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIxLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKxSB_EEKS8_KNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�bAdam/Adam/AssignAddVariableOph
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_16scalar_square_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bAdam/Adam/update_9/Squareh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�b$gradient_tape/model/tabl/mul_3/Mul_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_min_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b*Adam/Adam/update_4/clip_by_value_1/Minimumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_min_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b*Adam/Adam/update_7/clip_by_value_1/Minimumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1ExEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_13scalar_sum_opIKfSB_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EESF_EEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�bAdam/gradients/AddN_3h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_max_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b"Adam/Adam/update_7/clip_by_value_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_17scalar_product_opIffEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISC_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28��@�H�bAdam/Adam/update_11/mulh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_max_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b Adam/Adam/update_6/clip_by_valueh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_max_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b#Adam/Adam/update_10/clip_by_value_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_min_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b(Adam/Adam/update_7/clip_by_value/Minimumh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_18scalar_opposite_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b"gradient_tape/model/tabl/sub_2/Negh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_16scalar_square_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�bAdam/Adam/update_7/Squareh
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_12scalar_rightIffNS0_13scalar_max_opIffEELb0EEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_*28��@�H�b"Adam/Adam/update_3/clip_by_value_1h
�
�_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIbLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_19TensorCwiseBinaryOpINS0_21scalar_boolean_and_opEKNS4_INS5_IKbLi1ELi1EiEELi16ES7_EEKNS4_INS5_ISB_Li1ELi1ExEELi16ES7_EEEEEENS_9GpuDeviceEEExEEvT_T0_*28�@�H�b
LogicalAndh