å
ï
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
8
Const
output"dtype"
valuetensor"
dtypetype
,
Exp
x"T
y"T"
Ttype:

2
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
¿
MatrixDiagV3
diagonal"T
k
num_rows
num_cols
padding_value"T
output"T"	
Ttype"Q
alignstring
RIGHT_LEFT:2
0
LEFT_RIGHT
RIGHT_LEFT	LEFT_LEFTRIGHT_RIGHT

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8 ò
j
bl_1/W1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:dx*
shared_name	bl_1/W1
c
bl_1/W1/Read/ReadVariableOpReadVariableOpbl_1/W1*
_output_shapes

:dx*
dtype0
j
bl_1/W2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*
shared_name	bl_1/W2
c
bl_1/W2/Read/ReadVariableOpReadVariableOpbl_1/W2*
_output_shapes

:(*
dtype0
n
	bl_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*
shared_name	bl_1/bias
g
bl_1/bias/Read/ReadVariableOpReadVariableOp	bl_1/bias*
_output_shapes

:x*
dtype0
j
bl_2/W1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:x<*
shared_name	bl_2/W1
c
bl_2/W1/Read/ReadVariableOpReadVariableOpbl_2/W1*
_output_shapes

:x<*
dtype0
j
bl_2/W2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name	bl_2/W2
c
bl_2/W2/Read/ReadVariableOpReadVariableOpbl_2/W2*
_output_shapes

:*
dtype0
n
	bl_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*
shared_name	bl_2/bias
g
bl_2/bias/Read/ReadVariableOpReadVariableOp	bl_2/bias*
_output_shapes

:<*
dtype0
j
tabl/W1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*
shared_name	tabl/W1
c
tabl/W1/Read/ReadVariableOpReadVariableOptabl/W1*
_output_shapes

:<*
dtype0
j
tabl/W2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name	tabl/W2
c
tabl/W2/Read/ReadVariableOpReadVariableOptabl/W2*
_output_shapes

:*
dtype0
h
tabl/WVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nametabl/W
a
tabl/W/Read/ReadVariableOpReadVariableOptabl/W*
_output_shapes

:*
dtype0
l

tabl/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
tabl/alpha
e
tabl/alpha/Read/ReadVariableOpReadVariableOp
tabl/alpha*
_output_shapes
:*
dtype0
r
	tabl/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	tabl/bias
k
tabl/bias/Read/ReadVariableOpReadVariableOp	tabl/bias*"
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
x
Adam/bl_1/W1/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dx*
shared_nameAdam/bl_1/W1/m
q
"Adam/bl_1/W1/m/Read/ReadVariableOpReadVariableOpAdam/bl_1/W1/m*
_output_shapes

:dx*
dtype0
x
Adam/bl_1/W2/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*
shared_nameAdam/bl_1/W2/m
q
"Adam/bl_1/W2/m/Read/ReadVariableOpReadVariableOpAdam/bl_1/W2/m*
_output_shapes

:(*
dtype0
|
Adam/bl_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*!
shared_nameAdam/bl_1/bias/m
u
$Adam/bl_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/bl_1/bias/m*
_output_shapes

:x*
dtype0
x
Adam/bl_2/W1/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x<*
shared_nameAdam/bl_2/W1/m
q
"Adam/bl_2/W1/m/Read/ReadVariableOpReadVariableOpAdam/bl_2/W1/m*
_output_shapes

:x<*
dtype0
x
Adam/bl_2/W2/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameAdam/bl_2/W2/m
q
"Adam/bl_2/W2/m/Read/ReadVariableOpReadVariableOpAdam/bl_2/W2/m*
_output_shapes

:*
dtype0
|
Adam/bl_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*!
shared_nameAdam/bl_2/bias/m
u
$Adam/bl_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/bl_2/bias/m*
_output_shapes

:<*
dtype0
x
Adam/tabl/W1/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*
shared_nameAdam/tabl/W1/m
q
"Adam/tabl/W1/m/Read/ReadVariableOpReadVariableOpAdam/tabl/W1/m*
_output_shapes

:<*
dtype0
x
Adam/tabl/W2/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameAdam/tabl/W2/m
q
"Adam/tabl/W2/m/Read/ReadVariableOpReadVariableOpAdam/tabl/W2/m*
_output_shapes

:*
dtype0
v
Adam/tabl/W/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameAdam/tabl/W/m
o
!Adam/tabl/W/m/Read/ReadVariableOpReadVariableOpAdam/tabl/W/m*
_output_shapes

:*
dtype0
z
Adam/tabl/alpha/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/tabl/alpha/m
s
%Adam/tabl/alpha/m/Read/ReadVariableOpReadVariableOpAdam/tabl/alpha/m*
_output_shapes
:*
dtype0

Adam/tabl/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/tabl/bias/m
y
$Adam/tabl/bias/m/Read/ReadVariableOpReadVariableOpAdam/tabl/bias/m*"
_output_shapes
:*
dtype0
x
Adam/bl_1/W1/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dx*
shared_nameAdam/bl_1/W1/v
q
"Adam/bl_1/W1/v/Read/ReadVariableOpReadVariableOpAdam/bl_1/W1/v*
_output_shapes

:dx*
dtype0
x
Adam/bl_1/W2/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*
shared_nameAdam/bl_1/W2/v
q
"Adam/bl_1/W2/v/Read/ReadVariableOpReadVariableOpAdam/bl_1/W2/v*
_output_shapes

:(*
dtype0
|
Adam/bl_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x*!
shared_nameAdam/bl_1/bias/v
u
$Adam/bl_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/bl_1/bias/v*
_output_shapes

:x*
dtype0
x
Adam/bl_2/W1/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:x<*
shared_nameAdam/bl_2/W1/v
q
"Adam/bl_2/W1/v/Read/ReadVariableOpReadVariableOpAdam/bl_2/W1/v*
_output_shapes

:x<*
dtype0
x
Adam/bl_2/W2/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameAdam/bl_2/W2/v
q
"Adam/bl_2/W2/v/Read/ReadVariableOpReadVariableOpAdam/bl_2/W2/v*
_output_shapes

:*
dtype0
|
Adam/bl_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*!
shared_nameAdam/bl_2/bias/v
u
$Adam/bl_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/bl_2/bias/v*
_output_shapes

:<*
dtype0
x
Adam/tabl/W1/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<*
shared_nameAdam/tabl/W1/v
q
"Adam/tabl/W1/v/Read/ReadVariableOpReadVariableOpAdam/tabl/W1/v*
_output_shapes

:<*
dtype0
x
Adam/tabl/W2/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameAdam/tabl/W2/v
q
"Adam/tabl/W2/v/Read/ReadVariableOpReadVariableOpAdam/tabl/W2/v*
_output_shapes

:*
dtype0
v
Adam/tabl/W/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameAdam/tabl/W/v
o
!Adam/tabl/W/v/Read/ReadVariableOpReadVariableOpAdam/tabl/W/v*
_output_shapes

:*
dtype0
z
Adam/tabl/alpha/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/tabl/alpha/v
s
%Adam/tabl/alpha/v/Read/ReadVariableOpReadVariableOpAdam/tabl/alpha/v*
_output_shapes
:*
dtype0

Adam/tabl/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/tabl/bias/v
y
$Adam/tabl/bias/v/Read/ReadVariableOpReadVariableOpAdam/tabl/bias/v*"
_output_shapes
:*
dtype0

NoOpNoOp
Å<
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*<
valueö;Bó; Bì;
´
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
|

output_dim
W1
W2
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
|
 
output_dim
!W1
"W2
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
R
(trainable_variables
)regularization_losses
*	variables
+	keras_api
R
,trainable_variables
-regularization_losses
.	variables
/	keras_api

0
output_dim
1W1
2W2
3W
	4alpha
5bias
6trainable_variables
7regularization_losses
8	variables
9	keras_api
R
:trainable_variables
;regularization_losses
<	variables
=	keras_api

>iter

?beta_1

@beta_2
	Adecay
Blearning_ratemumvmw!mx"my#mz1m{2m|3m}4m~5mvvv!v"v#v1v2v3v4v5v
N
0
1
2
!3
"4
#5
16
27
38
49
510
 
N
0
1
2
!3
"4
#5
16
27
38
49
510
­
trainable_variables
Clayer_metrics
regularization_losses

Dlayers
Emetrics
Flayer_regularization_losses
Gnon_trainable_variables
	variables
 
 
OM
VARIABLE_VALUEbl_1/W12layer_with_weights-0/W1/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEbl_1/W22layer_with_weights-0/W2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	bl_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 

0
1
2
­
trainable_variables
Hlayer_metrics
regularization_losses

Ilayers
Jmetrics
Klayer_regularization_losses
Lnon_trainable_variables
	variables
 
 
 
­
trainable_variables
Mlayer_metrics
regularization_losses

Nlayers
Ometrics
Player_regularization_losses
Qnon_trainable_variables
	variables
 
 
 
­
trainable_variables
Rlayer_metrics
regularization_losses

Slayers
Tmetrics
Ulayer_regularization_losses
Vnon_trainable_variables
	variables
 
OM
VARIABLE_VALUEbl_2/W12layer_with_weights-1/W1/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEbl_2/W22layer_with_weights-1/W2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	bl_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
#2
 

!0
"1
#2
­
$trainable_variables
Wlayer_metrics
%regularization_losses

Xlayers
Ymetrics
Zlayer_regularization_losses
[non_trainable_variables
&	variables
 
 
 
­
(trainable_variables
\layer_metrics
)regularization_losses

]layers
^metrics
_layer_regularization_losses
`non_trainable_variables
*	variables
 
 
 
­
,trainable_variables
alayer_metrics
-regularization_losses

blayers
cmetrics
dlayer_regularization_losses
enon_trainable_variables
.	variables
 
OM
VARIABLE_VALUEtabl/W12layer_with_weights-2/W1/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEtabl/W22layer_with_weights-2/W2/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEtabl/W1layer_with_weights-2/W/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
tabl/alpha5layer_with_weights-2/alpha/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	tabl/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
#
10
21
32
43
54
 
#
10
21
32
43
54
­
6trainable_variables
flayer_metrics
7regularization_losses

glayers
hmetrics
ilayer_regularization_losses
jnon_trainable_variables
8	variables
 
 
 
­
:trainable_variables
klayer_metrics
;regularization_losses

llayers
mmetrics
nlayer_regularization_losses
onon_trainable_variables
<	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
?
0
1
2
3
4
5
6
7
	8

p0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	qtotal
	rcount
s	variables
t	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

q0
r1

s	variables
rp
VARIABLE_VALUEAdam/bl_1/W1/mNlayer_with_weights-0/W1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/bl_1/W2/mNlayer_with_weights-0/W2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/bl_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/bl_2/W1/mNlayer_with_weights-1/W1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/bl_2/W2/mNlayer_with_weights-1/W2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/bl_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/tabl/W1/mNlayer_with_weights-2/W1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/tabl/W2/mNlayer_with_weights-2/W2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/tabl/W/mMlayer_with_weights-2/W/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/tabl/alpha/mQlayer_with_weights-2/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/tabl/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/bl_1/W1/vNlayer_with_weights-0/W1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/bl_1/W2/vNlayer_with_weights-0/W2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/bl_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/bl_2/W1/vNlayer_with_weights-1/W1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/bl_2/W2/vNlayer_with_weights-1/W2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/bl_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/tabl/W1/vNlayer_with_weights-2/W1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/tabl/W2/vNlayer_with_weights-2/W2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/tabl/W/vMlayer_with_weights-2/W/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/tabl/alpha/vQlayer_with_weights-2/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/tabl/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿd(
¸
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1bl_1/W1bl_1/W2	bl_1/biasbl_2/W1bl_2/W2	bl_2/biastabl/W1tabl/W
tabl/alphatabl/W2	tabl/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *-
f(R&
$__inference_signature_wrapper_442856
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamebl_1/W1/Read/ReadVariableOpbl_1/W2/Read/ReadVariableOpbl_1/bias/Read/ReadVariableOpbl_2/W1/Read/ReadVariableOpbl_2/W2/Read/ReadVariableOpbl_2/bias/Read/ReadVariableOptabl/W1/Read/ReadVariableOptabl/W2/Read/ReadVariableOptabl/W/Read/ReadVariableOptabl/alpha/Read/ReadVariableOptabl/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"Adam/bl_1/W1/m/Read/ReadVariableOp"Adam/bl_1/W2/m/Read/ReadVariableOp$Adam/bl_1/bias/m/Read/ReadVariableOp"Adam/bl_2/W1/m/Read/ReadVariableOp"Adam/bl_2/W2/m/Read/ReadVariableOp$Adam/bl_2/bias/m/Read/ReadVariableOp"Adam/tabl/W1/m/Read/ReadVariableOp"Adam/tabl/W2/m/Read/ReadVariableOp!Adam/tabl/W/m/Read/ReadVariableOp%Adam/tabl/alpha/m/Read/ReadVariableOp$Adam/tabl/bias/m/Read/ReadVariableOp"Adam/bl_1/W1/v/Read/ReadVariableOp"Adam/bl_1/W2/v/Read/ReadVariableOp$Adam/bl_1/bias/v/Read/ReadVariableOp"Adam/bl_2/W1/v/Read/ReadVariableOp"Adam/bl_2/W2/v/Read/ReadVariableOp$Adam/bl_2/bias/v/Read/ReadVariableOp"Adam/tabl/W1/v/Read/ReadVariableOp"Adam/tabl/W2/v/Read/ReadVariableOp!Adam/tabl/W/v/Read/ReadVariableOp%Adam/tabl/alpha/v/Read/ReadVariableOp$Adam/tabl/bias/v/Read/ReadVariableOpConst*5
Tin.
,2*	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *(
f#R!
__inference__traced_save_443836
ñ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebl_1/W1bl_1/W2	bl_1/biasbl_2/W1bl_2/W2	bl_2/biastabl/W1tabl/W2tabl/W
tabl/alpha	tabl/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/bl_1/W1/mAdam/bl_1/W2/mAdam/bl_1/bias/mAdam/bl_2/W1/mAdam/bl_2/W2/mAdam/bl_2/bias/mAdam/tabl/W1/mAdam/tabl/W2/mAdam/tabl/W/mAdam/tabl/alpha/mAdam/tabl/bias/mAdam/bl_1/W1/vAdam/bl_1/W2/vAdam/bl_1/bias/vAdam/bl_2/W1/vAdam/bl_2/W2/vAdam/bl_2/bias/vAdam/tabl/W1/vAdam/tabl/W2/vAdam/tabl/W/vAdam/tabl/alpha/vAdam/tabl/bias/v*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *+
f&R$
"__inference__traced_restore_443966ÏÕ
¢
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_443442

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¸
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÂ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿx:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
ÕQ
ò
__inference__traced_save_443836
file_prefix&
"savev2_bl_1_w1_read_readvariableop&
"savev2_bl_1_w2_read_readvariableop(
$savev2_bl_1_bias_read_readvariableop&
"savev2_bl_2_w1_read_readvariableop&
"savev2_bl_2_w2_read_readvariableop(
$savev2_bl_2_bias_read_readvariableop&
"savev2_tabl_w1_read_readvariableop&
"savev2_tabl_w2_read_readvariableop%
!savev2_tabl_w_read_readvariableop)
%savev2_tabl_alpha_read_readvariableop(
$savev2_tabl_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_adam_bl_1_w1_m_read_readvariableop-
)savev2_adam_bl_1_w2_m_read_readvariableop/
+savev2_adam_bl_1_bias_m_read_readvariableop-
)savev2_adam_bl_2_w1_m_read_readvariableop-
)savev2_adam_bl_2_w2_m_read_readvariableop/
+savev2_adam_bl_2_bias_m_read_readvariableop-
)savev2_adam_tabl_w1_m_read_readvariableop-
)savev2_adam_tabl_w2_m_read_readvariableop,
(savev2_adam_tabl_w_m_read_readvariableop0
,savev2_adam_tabl_alpha_m_read_readvariableop/
+savev2_adam_tabl_bias_m_read_readvariableop-
)savev2_adam_bl_1_w1_v_read_readvariableop-
)savev2_adam_bl_1_w2_v_read_readvariableop/
+savev2_adam_bl_1_bias_v_read_readvariableop-
)savev2_adam_bl_2_w1_v_read_readvariableop-
)savev2_adam_bl_2_w2_v_read_readvariableop/
+savev2_adam_bl_2_bias_v_read_readvariableop-
)savev2_adam_tabl_w1_v_read_readvariableop-
)savev2_adam_tabl_w2_v_read_readvariableop,
(savev2_adam_tabl_w_v_read_readvariableop0
,savev2_adam_tabl_alpha_v_read_readvariableop/
+savev2_adam_tabl_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename¦
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*¸
value®B«)B2layer_with_weights-0/W1/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-0/W2/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/W1/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/W2/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/W1/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/W2/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/W/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/alpha/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-0/W1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-0/W2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/W1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/W2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/W1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/W2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-2/W/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-0/W1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-0/W2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/W1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/W2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/W1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/W2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-2/W/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÚ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÈ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0"savev2_bl_1_w1_read_readvariableop"savev2_bl_1_w2_read_readvariableop$savev2_bl_1_bias_read_readvariableop"savev2_bl_2_w1_read_readvariableop"savev2_bl_2_w2_read_readvariableop$savev2_bl_2_bias_read_readvariableop"savev2_tabl_w1_read_readvariableop"savev2_tabl_w2_read_readvariableop!savev2_tabl_w_read_readvariableop%savev2_tabl_alpha_read_readvariableop$savev2_tabl_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_adam_bl_1_w1_m_read_readvariableop)savev2_adam_bl_1_w2_m_read_readvariableop+savev2_adam_bl_1_bias_m_read_readvariableop)savev2_adam_bl_2_w1_m_read_readvariableop)savev2_adam_bl_2_w2_m_read_readvariableop+savev2_adam_bl_2_bias_m_read_readvariableop)savev2_adam_tabl_w1_m_read_readvariableop)savev2_adam_tabl_w2_m_read_readvariableop(savev2_adam_tabl_w_m_read_readvariableop,savev2_adam_tabl_alpha_m_read_readvariableop+savev2_adam_tabl_bias_m_read_readvariableop)savev2_adam_bl_1_w1_v_read_readvariableop)savev2_adam_bl_1_w2_v_read_readvariableop+savev2_adam_bl_1_bias_v_read_readvariableop)savev2_adam_bl_2_w1_v_read_readvariableop)savev2_adam_bl_2_w2_v_read_readvariableop+savev2_adam_bl_2_bias_v_read_readvariableop)savev2_adam_tabl_w1_v_read_readvariableop)savev2_adam_tabl_w2_v_read_readvariableop(savev2_adam_tabl_w_v_read_readvariableop,savev2_adam_tabl_alpha_v_read_readvariableop+savev2_adam_tabl_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *7
dtypes-
+2)	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ñ
_input_shapesß
Ü: :dx:(:x:x<::<:<::::: : : : : : : :dx:(:x:x<::<:<:::::dx:(:x:x<::<:<::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:dx:$ 

_output_shapes

:(:$ 

_output_shapes

:x:$ 

_output_shapes

:x<:$ 

_output_shapes

::$ 

_output_shapes

:<:$ 

_output_shapes

:<:$ 

_output_shapes

::$	 

_output_shapes

:: 


_output_shapes
::($
"
_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:dx:$ 

_output_shapes

:(:$ 

_output_shapes

:x:$ 

_output_shapes

:x<:$ 

_output_shapes

::$ 

_output_shapes

:<:$ 

_output_shapes

:<:$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::($
"
_output_shapes
::$ 

_output_shapes

:dx:$ 

_output_shapes

:(:$  

_output_shapes

:x:$! 

_output_shapes

:x<:$" 

_output_shapes

::$# 

_output_shapes

:<:$$ 

_output_shapes

:<:$% 

_output_shapes

::$& 

_output_shapes

:: '

_output_shapes
::(($
"
_output_shapes
::)

_output_shapes
: 
¾
d
H__inference_activation_3_layer_call_and_return_conditional_losses_442653

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡!
ù
A__inference_model_layer_call_and_return_conditional_losses_442696
input_1
bl_1_442665
bl_1_442667
bl_1_442669
bl_2_442674
bl_2_442676
bl_2_442678
tabl_442683
tabl_442685
tabl_442687
tabl_442689
tabl_442691
identity¢bl_1/StatefulPartitionedCall¢bl_2/StatefulPartitionedCall¢tabl/StatefulPartitionedCall
bl_1/StatefulPartitionedCallStatefulPartitionedCallinput_1bl_1_442665bl_1_442667bl_1_442669*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_bl_1_layer_call_and_return_conditional_losses_4423312
bl_1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall%bl_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_4423562
activation_1/PartitionedCallÿ
dropout_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_4423812
dropout_1/PartitionedCall´
bl_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0bl_2_442674bl_2_442676bl_2_442678*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_bl_2_layer_call_and_return_conditional_losses_4424492
bl_2/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall%bl_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_4424742
activation_2/PartitionedCallÿ
dropout_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_4424992
dropout_2/PartitionedCallÎ
tabl/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0tabl_442683tabl_442685tabl_442687tabl_442689tabl_442691*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_tabl_layer_call_and_return_conditional_losses_4426202
tabl/StatefulPartitionedCall
activation_3/PartitionedCallPartitionedCall%tabl/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_4426532
activation_3/PartitionedCallÖ
IdentityIdentity%activation_3/PartitionedCall:output:0^bl_1/StatefulPartitionedCall^bl_2/StatefulPartitionedCall^tabl/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿd(:::::::::::2<
bl_1/StatefulPartitionedCallbl_1/StatefulPartitionedCall2<
bl_2/StatefulPartitionedCallbl_2/StatefulPartitionedCall2<
tabl/StatefulPartitionedCalltabl/StatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
!
_user_specified_name	input_1
Ø
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_442499

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
Ø
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_443447

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿx:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
ÄÑ
¨
A__inference_model_layer_call_and_return_conditional_losses_443300

inputs(
$bl_1_shape_1_readvariableop_resource(
$bl_1_shape_3_readvariableop_resource*
&bl_1_reshape_6_readvariableop_resource(
$bl_2_shape_1_readvariableop_resource(
$bl_2_shape_3_readvariableop_resource*
&bl_2_reshape_6_readvariableop_resource(
$tabl_shape_1_readvariableop_resource 
tabl_readvariableop_resource"
tabl_readvariableop_2_resource(
$tabl_shape_5_readvariableop_resource&
"tabl_add_2_readvariableop_resource
identity¢bl_1/Reshape_6/ReadVariableOp¢bl_1/transpose_1/ReadVariableOp¢bl_1/transpose_3/ReadVariableOp¢bl_2/Reshape_6/ReadVariableOp¢bl_2/transpose_1/ReadVariableOp¢bl_2/transpose_3/ReadVariableOp¢tabl/ReadVariableOp¢tabl/ReadVariableOp_1¢tabl/ReadVariableOp_2¢tabl/ReadVariableOp_3¢tabl/add_2/ReadVariableOp¢tabl/transpose_1/ReadVariableOp¢tabl/transpose_4/ReadVariableOp
bl_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
bl_1/transpose/perm
bl_1/transpose	Transposeinputsbl_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(d2
bl_1/transposeZ

bl_1/ShapeShapebl_1/transpose:y:0*
T0*
_output_shapes
:2

bl_1/Shapek
bl_1/unstackUnpackbl_1/Shape:output:0*
T0*
_output_shapes
: : : *	
num2
bl_1/unstack
bl_1/Shape_1/ReadVariableOpReadVariableOp$bl_1_shape_1_readvariableop_resource*
_output_shapes

:dx*
dtype02
bl_1/Shape_1/ReadVariableOpm
bl_1/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"d   x   2
bl_1/Shape_1o
bl_1/unstack_1Unpackbl_1/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2
bl_1/unstack_1y
bl_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2
bl_1/Reshape/shape
bl_1/ReshapeReshapebl_1/transpose:y:0bl_1/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
bl_1/Reshape§
bl_1/transpose_1/ReadVariableOpReadVariableOp$bl_1_shape_1_readvariableop_resource*
_output_shapes

:dx*
dtype02!
bl_1/transpose_1/ReadVariableOp
bl_1/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
bl_1/transpose_1/perm£
bl_1/transpose_1	Transpose'bl_1/transpose_1/ReadVariableOp:value:0bl_1/transpose_1/perm:output:0*
T0*
_output_shapes

:dx2
bl_1/transpose_1}
bl_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"d   ÿÿÿÿ2
bl_1/Reshape_1/shape
bl_1/Reshape_1Reshapebl_1/transpose_1:y:0bl_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:dx2
bl_1/Reshape_1
bl_1/MatMulMatMulbl_1/Reshape:output:0bl_1/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
bl_1/MatMulr
bl_1/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :(2
bl_1/Reshape_2/shape/1r
bl_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :x2
bl_1/Reshape_2/shape/2»
bl_1/Reshape_2/shapePackbl_1/unstack:output:0bl_1/Reshape_2/shape/1:output:0bl_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
bl_1/Reshape_2/shape
bl_1/Reshape_2Reshapebl_1/MatMul:product:0bl_1/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(x2
bl_1/Reshape_2
bl_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
bl_1/transpose_2/perm 
bl_1/transpose_2	Transposebl_1/Reshape_2:output:0bl_1/transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx(2
bl_1/transpose_2`
bl_1/Shape_2Shapebl_1/transpose_2:y:0*
T0*
_output_shapes
:2
bl_1/Shape_2q
bl_1/unstack_2Unpackbl_1/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2
bl_1/unstack_2
bl_1/Shape_3/ReadVariableOpReadVariableOp$bl_1_shape_3_readvariableop_resource*
_output_shapes

:(*
dtype02
bl_1/Shape_3/ReadVariableOpm
bl_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"(      2
bl_1/Shape_3o
bl_1/unstack_3Unpackbl_1/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2
bl_1/unstack_3}
bl_1/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   2
bl_1/Reshape_3/shape
bl_1/Reshape_3Reshapebl_1/transpose_2:y:0bl_1/Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
bl_1/Reshape_3§
bl_1/transpose_3/ReadVariableOpReadVariableOp$bl_1_shape_3_readvariableop_resource*
_output_shapes

:(*
dtype02!
bl_1/transpose_3/ReadVariableOp
bl_1/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
bl_1/transpose_3/perm£
bl_1/transpose_3	Transpose'bl_1/transpose_3/ReadVariableOp:value:0bl_1/transpose_3/perm:output:0*
T0*
_output_shapes

:(2
bl_1/transpose_3}
bl_1/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"(   ÿÿÿÿ2
bl_1/Reshape_4/shape
bl_1/Reshape_4Reshapebl_1/transpose_3:y:0bl_1/Reshape_4/shape:output:0*
T0*
_output_shapes

:(2
bl_1/Reshape_4
bl_1/MatMul_1MatMulbl_1/Reshape_3:output:0bl_1/Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bl_1/MatMul_1r
bl_1/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :x2
bl_1/Reshape_5/shape/1r
bl_1/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
bl_1/Reshape_5/shape/2½
bl_1/Reshape_5/shapePackbl_1/unstack_2:output:0bl_1/Reshape_5/shape/1:output:0bl_1/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
bl_1/Reshape_5/shape
bl_1/Reshape_5Reshapebl_1/MatMul_1:product:0bl_1/Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
bl_1/Reshape_5¥
bl_1/Reshape_6/ReadVariableOpReadVariableOp&bl_1_reshape_6_readvariableop_resource*
_output_shapes

:x*
dtype02
bl_1/Reshape_6/ReadVariableOp
bl_1/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   x      2
bl_1/Reshape_6/shape
bl_1/Reshape_6Reshape%bl_1/Reshape_6/ReadVariableOp:value:0bl_1/Reshape_6/shape:output:0*
T0*"
_output_shapes
:x2
bl_1/Reshape_6
bl_1/addAddV2bl_1/Reshape_5:output:0bl_1/Reshape_6:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2

bl_1/addr
activation_1/ReluRelubl_1/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
activation_1/Relu
dropout_1/IdentityIdentityactivation_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
dropout_1/Identity
bl_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
bl_2/transpose/perm
bl_2/transpose	Transposedropout_1/Identity:output:0bl_2/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
bl_2/transposeZ

bl_2/ShapeShapebl_2/transpose:y:0*
T0*
_output_shapes
:2

bl_2/Shapek
bl_2/unstackUnpackbl_2/Shape:output:0*
T0*
_output_shapes
: : : *	
num2
bl_2/unstack
bl_2/Shape_1/ReadVariableOpReadVariableOp$bl_2_shape_1_readvariableop_resource*
_output_shapes

:x<*
dtype02
bl_2/Shape_1/ReadVariableOpm
bl_2/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"x   <   2
bl_2/Shape_1o
bl_2/unstack_1Unpackbl_2/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2
bl_2/unstack_1y
bl_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿx   2
bl_2/Reshape/shape
bl_2/ReshapeReshapebl_2/transpose:y:0bl_2/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
bl_2/Reshape§
bl_2/transpose_1/ReadVariableOpReadVariableOp$bl_2_shape_1_readvariableop_resource*
_output_shapes

:x<*
dtype02!
bl_2/transpose_1/ReadVariableOp
bl_2/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
bl_2/transpose_1/perm£
bl_2/transpose_1	Transpose'bl_2/transpose_1/ReadVariableOp:value:0bl_2/transpose_1/perm:output:0*
T0*
_output_shapes

:x<2
bl_2/transpose_1}
bl_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"x   ÿÿÿÿ2
bl_2/Reshape_1/shape
bl_2/Reshape_1Reshapebl_2/transpose_1:y:0bl_2/Reshape_1/shape:output:0*
T0*
_output_shapes

:x<2
bl_2/Reshape_1
bl_2/MatMulMatMulbl_2/Reshape:output:0bl_2/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
bl_2/MatMulr
bl_2/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
bl_2/Reshape_2/shape/1r
bl_2/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :<2
bl_2/Reshape_2/shape/2»
bl_2/Reshape_2/shapePackbl_2/unstack:output:0bl_2/Reshape_2/shape/1:output:0bl_2/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
bl_2/Reshape_2/shape
bl_2/Reshape_2Reshapebl_2/MatMul:product:0bl_2/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
bl_2/Reshape_2
bl_2/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
bl_2/transpose_2/perm 
bl_2/transpose_2	Transposebl_2/Reshape_2:output:0bl_2/transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
bl_2/transpose_2`
bl_2/Shape_2Shapebl_2/transpose_2:y:0*
T0*
_output_shapes
:2
bl_2/Shape_2q
bl_2/unstack_2Unpackbl_2/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2
bl_2/unstack_2
bl_2/Shape_3/ReadVariableOpReadVariableOp$bl_2_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02
bl_2/Shape_3/ReadVariableOpm
bl_2/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      2
bl_2/Shape_3o
bl_2/unstack_3Unpackbl_2/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2
bl_2/unstack_3}
bl_2/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
bl_2/Reshape_3/shape
bl_2/Reshape_3Reshapebl_2/transpose_2:y:0bl_2/Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bl_2/Reshape_3§
bl_2/transpose_3/ReadVariableOpReadVariableOp$bl_2_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02!
bl_2/transpose_3/ReadVariableOp
bl_2/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
bl_2/transpose_3/perm£
bl_2/transpose_3	Transpose'bl_2/transpose_3/ReadVariableOp:value:0bl_2/transpose_3/perm:output:0*
T0*
_output_shapes

:2
bl_2/transpose_3}
bl_2/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
bl_2/Reshape_4/shape
bl_2/Reshape_4Reshapebl_2/transpose_3:y:0bl_2/Reshape_4/shape:output:0*
T0*
_output_shapes

:2
bl_2/Reshape_4
bl_2/MatMul_1MatMulbl_2/Reshape_3:output:0bl_2/Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bl_2/MatMul_1r
bl_2/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<2
bl_2/Reshape_5/shape/1r
bl_2/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
bl_2/Reshape_5/shape/2½
bl_2/Reshape_5/shapePackbl_2/unstack_2:output:0bl_2/Reshape_5/shape/1:output:0bl_2/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
bl_2/Reshape_5/shape
bl_2/Reshape_5Reshapebl_2/MatMul_1:product:0bl_2/Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
bl_2/Reshape_5¥
bl_2/Reshape_6/ReadVariableOpReadVariableOp&bl_2_reshape_6_readvariableop_resource*
_output_shapes

:<*
dtype02
bl_2/Reshape_6/ReadVariableOp
bl_2/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   <      2
bl_2/Reshape_6/shape
bl_2/Reshape_6Reshape%bl_2/Reshape_6/ReadVariableOp:value:0bl_2/Reshape_6/shape:output:0*
T0*"
_output_shapes
:<2
bl_2/Reshape_6
bl_2/addAddV2bl_2/Reshape_5:output:0bl_2/Reshape_6:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

bl_2/addr
activation_2/ReluRelubl_2/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
activation_2/Relu
dropout_2/IdentityIdentityactivation_2/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout_2/Identity
tabl/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
tabl/transpose/perm
tabl/transpose	Transposedropout_2/Identity:output:0tabl/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
tabl/transposeZ

tabl/ShapeShapetabl/transpose:y:0*
T0*
_output_shapes
:2

tabl/Shapek
tabl/unstackUnpacktabl/Shape:output:0*
T0*
_output_shapes
: : : *	
num2
tabl/unstack
tabl/Shape_1/ReadVariableOpReadVariableOp$tabl_shape_1_readvariableop_resource*
_output_shapes

:<*
dtype02
tabl/Shape_1/ReadVariableOpm
tabl/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"<      2
tabl/Shape_1o
tabl/unstack_1Unpacktabl/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2
tabl/unstack_1y
tabl/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ<   2
tabl/Reshape/shape
tabl/ReshapeReshapetabl/transpose:y:0tabl/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
tabl/Reshape§
tabl/transpose_1/ReadVariableOpReadVariableOp$tabl_shape_1_readvariableop_resource*
_output_shapes

:<*
dtype02!
tabl/transpose_1/ReadVariableOp
tabl/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
tabl/transpose_1/perm£
tabl/transpose_1	Transpose'tabl/transpose_1/ReadVariableOp:value:0tabl/transpose_1/perm:output:0*
T0*
_output_shapes

:<2
tabl/transpose_1}
tabl/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"<   ÿÿÿÿ2
tabl/Reshape_1/shape
tabl/Reshape_1Reshapetabl/transpose_1:y:0tabl/Reshape_1/shape:output:0*
T0*
_output_shapes

:<2
tabl/Reshape_1
tabl/MatMulMatMultabl/Reshape:output:0tabl/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tabl/MatMulr
tabl/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
tabl/Reshape_2/shape/1r
tabl/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
tabl/Reshape_2/shape/2»
tabl/Reshape_2/shapePacktabl/unstack:output:0tabl/Reshape_2/shape/1:output:0tabl/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
tabl/Reshape_2/shape
tabl/Reshape_2Reshapetabl/MatMul:product:0tabl/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tabl/Reshape_2
tabl/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
tabl/transpose_2/perm 
tabl/transpose_2	Transposetabl/Reshape_2:output:0tabl/transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tabl/transpose_2k
tabl/eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2
tabl/eye/onesd
tabl/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
tabl/eye/diag/k{
tabl/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tabl/eye/diag/num_rows{
tabl/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tabl/eye/diag/num_cols
tabl/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
tabl/eye/diag/padding_valueñ
tabl/eye/diagMatrixDiagV3tabl/eye/ones:output:0tabl/eye/diag/k:output:0tabl/eye/diag/num_rows:output:0tabl/eye/diag/num_cols:output:0$tabl/eye/diag/padding_value:output:0*
T0*
_output_shapes

:2
tabl/eye/diag
tabl/ReadVariableOpReadVariableOptabl_readvariableop_resource*
_output_shapes

:*
dtype02
tabl/ReadVariableOpy
tabl/mulMultabl/ReadVariableOp:value:0tabl/eye/diag:output:0*
T0*
_output_shapes

:2

tabl/mul
tabl/ReadVariableOp_1ReadVariableOptabl_readvariableop_resource*
_output_shapes

:*
dtype02
tabl/ReadVariableOp_1q
tabl/subSubtabl/ReadVariableOp_1:value:0tabl/mul:z:0*
T0*
_output_shapes

:2

tabl/subo
tabl/eye_1/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2
tabl/eye_1/onesh
tabl/eye_1/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
tabl/eye_1/diag/k
tabl/eye_1/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tabl/eye_1/diag/num_rows
tabl/eye_1/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tabl/eye_1/diag/num_cols
tabl/eye_1/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
tabl/eye_1/diag/padding_valueÿ
tabl/eye_1/diagMatrixDiagV3tabl/eye_1/ones:output:0tabl/eye_1/diag/k:output:0!tabl/eye_1/diag/num_rows:output:0!tabl/eye_1/diag/num_cols:output:0&tabl/eye_1/diag/padding_value:output:0*
T0*
_output_shapes

:2
tabl/eye_1/diage
tabl/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tabl/truediv/y
tabl/truedivRealDivtabl/eye_1/diag:output:0tabl/truediv/y:output:0*
T0*
_output_shapes

:2
tabl/truedivf
tabl/addAddV2tabl/sub:z:0tabl/truediv:z:0*
T0*
_output_shapes

:2

tabl/add`
tabl/Shape_2Shapetabl/transpose_2:y:0*
T0*
_output_shapes
:2
tabl/Shape_2q
tabl/unstack_2Unpacktabl/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2
tabl/unstack_2m
tabl/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      2
tabl/Shape_3o
tabl/unstack_3Unpacktabl/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2
tabl/unstack_3}
tabl/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
tabl/Reshape_3/shape
tabl/Reshape_3Reshapetabl/transpose_2:y:0tabl/Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tabl/Reshape_3
tabl/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
tabl/transpose_3/perm
tabl/transpose_3	Transposetabl/add:z:0tabl/transpose_3/perm:output:0*
T0*
_output_shapes

:2
tabl/transpose_3}
tabl/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
tabl/Reshape_4/shape
tabl/Reshape_4Reshapetabl/transpose_3:y:0tabl/Reshape_4/shape:output:0*
T0*
_output_shapes

:2
tabl/Reshape_4
tabl/MatMul_1MatMultabl/Reshape_3:output:0tabl/Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tabl/MatMul_1r
tabl/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
tabl/Reshape_5/shape/1r
tabl/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
tabl/Reshape_5/shape/2½
tabl/Reshape_5/shapePacktabl/unstack_2:output:0tabl/Reshape_5/shape/1:output:0tabl/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
tabl/Reshape_5/shape
tabl/Reshape_5Reshapetabl/MatMul_1:product:0tabl/Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tabl/Reshape_5
tabl/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tabl/Max/reduction_indices 
tabl/MaxMaxtabl/Reshape_5:output:0#tabl/Max/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2

tabl/Max

tabl/sub_1Subtabl/Reshape_5:output:0tabl/Max:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

tabl/sub_1a
tabl/ExpExptabl/sub_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

tabl/Exp
tabl/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tabl/Sum/reduction_indices
tabl/SumSumtabl/Exp:y:0#tabl/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2

tabl/Sum
tabl/truediv_1RealDivtabl/Exp:y:0tabl/Sum:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tabl/truediv_1
tabl/ReadVariableOp_2ReadVariableOptabl_readvariableop_2_resource*
_output_shapes
:*
dtype02
tabl/ReadVariableOp_2

tabl/mul_1Multabl/ReadVariableOp_2:value:0tabl/transpose_2:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

tabl/mul_1
tabl/ReadVariableOp_3ReadVariableOptabl_readvariableop_2_resource*
_output_shapes
:*
dtype02
tabl/ReadVariableOp_3a
tabl/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tabl/sub_2/xz

tabl/sub_2Subtabl/sub_2/x:output:0tabl/ReadVariableOp_3:value:0*
T0*
_output_shapes
:2

tabl/sub_2{

tabl/mul_2Multabl/sub_2:z:0tabl/transpose_2:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

tabl/mul_2y

tabl/mul_3Multabl/mul_2:z:0tabl/truediv_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

tabl/mul_3w

tabl/add_1AddV2tabl/mul_1:z:0tabl/mul_3:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

tabl/add_1Z
tabl/Shape_4Shapetabl/add_1:z:0*
T0*
_output_shapes
:2
tabl/Shape_4q
tabl/unstack_4Unpacktabl/Shape_4:output:0*
T0*
_output_shapes
: : : *	
num2
tabl/unstack_4
tabl/Shape_5/ReadVariableOpReadVariableOp$tabl_shape_5_readvariableop_resource*
_output_shapes

:*
dtype02
tabl/Shape_5/ReadVariableOpm
tabl/Shape_5Const*
_output_shapes
:*
dtype0*
valueB"      2
tabl/Shape_5o
tabl/unstack_5Unpacktabl/Shape_5:output:0*
T0*
_output_shapes
: : *	
num2
tabl/unstack_5}
tabl/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
tabl/Reshape_6/shape
tabl/Reshape_6Reshapetabl/add_1:z:0tabl/Reshape_6/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tabl/Reshape_6§
tabl/transpose_4/ReadVariableOpReadVariableOp$tabl_shape_5_readvariableop_resource*
_output_shapes

:*
dtype02!
tabl/transpose_4/ReadVariableOp
tabl/transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
tabl/transpose_4/perm£
tabl/transpose_4	Transpose'tabl/transpose_4/ReadVariableOp:value:0tabl/transpose_4/perm:output:0*
T0*
_output_shapes

:2
tabl/transpose_4}
tabl/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
tabl/Reshape_7/shape
tabl/Reshape_7Reshapetabl/transpose_4:y:0tabl/Reshape_7/shape:output:0*
T0*
_output_shapes

:2
tabl/Reshape_7
tabl/MatMul_2MatMultabl/Reshape_6:output:0tabl/Reshape_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tabl/MatMul_2r
tabl/Reshape_8/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
tabl/Reshape_8/shape/1r
tabl/Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
tabl/Reshape_8/shape/2½
tabl/Reshape_8/shapePacktabl/unstack_4:output:0tabl/Reshape_8/shape/1:output:0tabl/Reshape_8/shape/2:output:0*
N*
T0*
_output_shapes
:2
tabl/Reshape_8/shape
tabl/Reshape_8Reshapetabl/MatMul_2:product:0tabl/Reshape_8/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tabl/Reshape_8
tabl/add_2/ReadVariableOpReadVariableOp"tabl_add_2_readvariableop_resource*"
_output_shapes
:*
dtype02
tabl/add_2/ReadVariableOp

tabl/add_2AddV2tabl/Reshape_8:output:0!tabl/add_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

tabl/add_2
tabl/SqueezeSqueezetabl/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2
tabl/Squeeze
activation_3/SoftmaxSoftmaxtabl/Squeeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_3/Softmaxø
IdentityIdentityactivation_3/Softmax:softmax:0^bl_1/Reshape_6/ReadVariableOp ^bl_1/transpose_1/ReadVariableOp ^bl_1/transpose_3/ReadVariableOp^bl_2/Reshape_6/ReadVariableOp ^bl_2/transpose_1/ReadVariableOp ^bl_2/transpose_3/ReadVariableOp^tabl/ReadVariableOp^tabl/ReadVariableOp_1^tabl/ReadVariableOp_2^tabl/ReadVariableOp_3^tabl/add_2/ReadVariableOp ^tabl/transpose_1/ReadVariableOp ^tabl/transpose_4/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿd(:::::::::::2>
bl_1/Reshape_6/ReadVariableOpbl_1/Reshape_6/ReadVariableOp2B
bl_1/transpose_1/ReadVariableOpbl_1/transpose_1/ReadVariableOp2B
bl_1/transpose_3/ReadVariableOpbl_1/transpose_3/ReadVariableOp2>
bl_2/Reshape_6/ReadVariableOpbl_2/Reshape_6/ReadVariableOp2B
bl_2/transpose_1/ReadVariableOpbl_2/transpose_1/ReadVariableOp2B
bl_2/transpose_3/ReadVariableOpbl_2/transpose_3/ReadVariableOp2*
tabl/ReadVariableOptabl/ReadVariableOp2.
tabl/ReadVariableOp_1tabl/ReadVariableOp_12.
tabl/ReadVariableOp_2tabl/ReadVariableOp_22.
tabl/ReadVariableOp_3tabl/ReadVariableOp_326
tabl/add_2/ReadVariableOptabl/add_2/ReadVariableOp2B
tabl/transpose_1/ReadVariableOptabl/transpose_1/ReadVariableOp2B
tabl/transpose_4/ReadVariableOptabl/transpose_4/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
 
_user_specified_nameinputs
©
F
*__inference_dropout_2_layer_call_fn_443560

inputs
identityÌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_4424992
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
Í\

@__inference_tabl_layer_call_and_return_conditional_losses_442620
x#
shape_1_readvariableop_resource
readvariableop_resource
readvariableop_2_resource#
shape_5_readvariableop_resource!
add_2_readvariableop_resource
identity¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢add_2/ReadVariableOp¢transpose_1/ReadVariableOp¢transpose_4/ReadVariableOpu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permu
	transpose	Transposextranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shape\
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:<*
dtype02
Shape_1/ReadVariableOpc
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"<      2	
Shape_1`
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ<   2
Reshape/shapev
ReshapeReshapetranspose:y:0Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2	
Reshape
transpose_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:<*
dtype02
transpose_1/ReadVariableOpu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

:<2
transpose_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"<   ÿÿÿÿ2
Reshape_1/shapeu
	Reshape_1Reshapetranspose_1:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:<2
	Reshape_1r
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulh
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_2/shape/1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_2/shape/2¢
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_2y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm
transpose_2	TransposeReshape_2:output:0transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
transpose_2a
eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2

eye/onesZ

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

eye/diag/kq
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye/diag/num_rowsq
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye/diag/num_colsu
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye/diag/padding_valueÎ
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*
_output_shapes

:2

eye/diagx
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype02
ReadVariableOpe
mulMulReadVariableOp:value:0eye/diag:output:0*
T0*
_output_shapes

:2
mul|
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype02
ReadVariableOp_1]
subSubReadVariableOp_1:value:0mul:z:0*
T0*
_output_shapes

:2
sube

eye_1/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2

eye_1/ones^
eye_1/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
eye_1/diag/ku
eye_1/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye_1/diag/num_rowsu
eye_1/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye_1/diag/num_colsy
eye_1/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye_1/diag/padding_valueÜ

eye_1/diagMatrixDiagV3eye_1/ones:output:0eye_1/diag/k:output:0eye_1/diag/num_rows:output:0eye_1/diag/num_cols:output:0!eye_1/diag/padding_value:output:0*
T0*
_output_shapes

:2

eye_1/diag[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/yo
truedivRealDiveye_1/diag:output:0truediv/y:output:0*
T0*
_output_shapes

:2	
truedivR
addAddV2sub:z:0truediv:z:0*
T0*
_output_shapes

:2
addQ
Shape_2Shapetranspose_2:y:0*
T0*
_output_shapes
:2	
Shape_2b
	unstack_2UnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
num2
	unstack_2c
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      2	
Shape_3`
	unstack_3UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_3s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Reshape_3/shape~
	Reshape_3Reshapetranspose_2:y:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_3u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/permt
transpose_3	Transposeadd:z:0transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3s
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
Reshape_4/shapeu
	Reshape_4Reshapetranspose_3:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

:2
	Reshape_4x
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1h
Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_5/shape/1h
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_5/shape/2¤
Reshape_5/shapePackunstack_2:output:0Reshape_5/shape/1:output:0Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_5/shape
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_5y
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Max/reduction_indices
MaxMaxReshape_5:output:0Max/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Maxm
sub_1SubReshape_5:output:0Max:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub_1R
ExpExp	sub_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Sum/reduction_indices
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Sumn
	truediv_1RealDivExp:y:0Sum:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	truediv_1z
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype02
ReadVariableOp_2v
mul_1MulReadVariableOp_2:value:0transpose_2:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1z
ReadVariableOp_3ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype02
ReadVariableOp_3W
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_2/xf
sub_2Subsub_2/x:output:0ReadVariableOp_3:value:0*
T0*
_output_shapes
:2
sub_2g
mul_2Mul	sub_2:z:0transpose_2:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
mul_3Mul	mul_2:z:0truediv_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_3c
add_1AddV2	mul_1:z:0	mul_3:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1K
Shape_4Shape	add_1:z:0*
T0*
_output_shapes
:2	
Shape_4b
	unstack_4UnpackShape_4:output:0*
T0*
_output_shapes
: : : *	
num2
	unstack_4
Shape_5/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

:*
dtype02
Shape_5/ReadVariableOpc
Shape_5Const*
_output_shapes
:*
dtype0*
valueB"      2	
Shape_5`
	unstack_5UnpackShape_5:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_5s
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Reshape_6/shapex
	Reshape_6Reshape	add_1:z:0Reshape_6/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_6
transpose_4/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

:*
dtype02
transpose_4/ReadVariableOpu
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transpose"transpose_4/ReadVariableOp:value:0transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4s
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
Reshape_7/shapeu
	Reshape_7Reshapetranspose_4:y:0Reshape_7/shape:output:0*
T0*
_output_shapes

:2
	Reshape_7x
MatMul_2MatMulReshape_6:output:0Reshape_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_2h
Reshape_8/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_8/shape/1h
Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_8/shape/2¤
Reshape_8/shapePackunstack_4:output:0Reshape_8/shape/1:output:0Reshape_8/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_8/shape
	Reshape_8ReshapeMatMul_2:product:0Reshape_8/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_8
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*"
_output_shapes
:*
dtype02
add_2/ReadVariableOp
add_2AddV2Reshape_8:output:0add_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2z
SqueezeSqueeze	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2	
Squeezeÿ
IdentityIdentitySqueeze:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^add_2/ReadVariableOp^transpose_1/ReadVariableOp^transpose_4/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ<:::::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
add_2/ReadVariableOpadd_2/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp28
transpose_4/ReadVariableOptranspose_4/ReadVariableOp:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<

_user_specified_namex
	

&__inference_model_layer_call_fn_442758
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4427332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿd(:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
!
_user_specified_name	input_1
¾
d
H__inference_activation_3_layer_call_and_return_conditional_losses_443688

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯
I
-__inference_activation_2_layer_call_fn_443533

inputs
identityÏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_4424742
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
Æ
d
H__inference_activation_1_layer_call_and_return_conditional_losses_442356

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿx:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
¹.
 
@__inference_bl_2_layer_call_and_return_conditional_losses_443512
x#
shape_1_readvariableop_resource#
shape_3_readvariableop_resource%
!reshape_6_readvariableop_resource
identity¢Reshape_6/ReadVariableOp¢transpose_1/ReadVariableOp¢transpose_3/ReadVariableOpu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permu
	transpose	Transposextranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shape\
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:x<*
dtype02
Shape_1/ReadVariableOpc
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"x   <   2	
Shape_1`
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿx   2
Reshape/shapev
ReshapeReshapetranspose:y:0Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2	
Reshape
transpose_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:x<*
dtype02
transpose_1/ReadVariableOpu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

:x<2
transpose_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"x   ÿÿÿÿ2
Reshape_1/shapeu
	Reshape_1Reshapetranspose_1:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:x<2
	Reshape_1r
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
MatMulh
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_2/shape/1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :<2
Reshape_2/shape/2¢
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
	Reshape_2y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm
transpose_2	TransposeReshape_2:output:0transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
transpose_2Q
Shape_2Shapetranspose_2:y:0*
T0*
_output_shapes
:2	
Shape_2b
	unstack_2UnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
num2
	unstack_2
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:*
dtype02
Shape_3/ReadVariableOpc
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      2	
Shape_3`
	unstack_3UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_3s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Reshape_3/shape~
	Reshape_3Reshapetranspose_2:y:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_3
transpose_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:*
dtype02
transpose_3/ReadVariableOpu
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm
transpose_3	Transpose"transpose_3/ReadVariableOp:value:0transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3s
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
Reshape_4/shapeu
	Reshape_4Reshapetranspose_3:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

:2
	Reshape_4x
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1h
Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<2
Reshape_5/shape/1h
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_5/shape/2¤
Reshape_5/shapePackunstack_2:output:0Reshape_5/shape/1:output:0Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_5/shape
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
	Reshape_5
Reshape_6/ReadVariableOpReadVariableOp!reshape_6_readvariableop_resource*
_output_shapes

:<*
dtype02
Reshape_6/ReadVariableOpw
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   <      2
Reshape_6/shape
	Reshape_6Reshape Reshape_6/ReadVariableOp:value:0Reshape_6/shape:output:0*
T0*"
_output_shapes
:<2
	Reshape_6q
addAddV2Reshape_5:output:0Reshape_6:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
add´
IdentityIdentityadd:z:0^Reshape_6/ReadVariableOp^transpose_1/ReadVariableOp^transpose_3/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿx:::24
Reshape_6/ReadVariableOpReshape_6/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp28
transpose_3/ReadVariableOptranspose_3/ReadVariableOp:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx

_user_specified_namex
©
F
*__inference_dropout_1_layer_call_fn_443457

inputs
identityÌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_4423812
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿx:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
Âä
¨
A__inference_model_layer_call_and_return_conditional_losses_443085

inputs(
$bl_1_shape_1_readvariableop_resource(
$bl_1_shape_3_readvariableop_resource*
&bl_1_reshape_6_readvariableop_resource(
$bl_2_shape_1_readvariableop_resource(
$bl_2_shape_3_readvariableop_resource*
&bl_2_reshape_6_readvariableop_resource(
$tabl_shape_1_readvariableop_resource 
tabl_readvariableop_resource"
tabl_readvariableop_2_resource(
$tabl_shape_5_readvariableop_resource&
"tabl_add_2_readvariableop_resource
identity¢bl_1/Reshape_6/ReadVariableOp¢bl_1/transpose_1/ReadVariableOp¢bl_1/transpose_3/ReadVariableOp¢bl_2/Reshape_6/ReadVariableOp¢bl_2/transpose_1/ReadVariableOp¢bl_2/transpose_3/ReadVariableOp¢tabl/ReadVariableOp¢tabl/ReadVariableOp_1¢tabl/ReadVariableOp_2¢tabl/ReadVariableOp_3¢tabl/add_2/ReadVariableOp¢tabl/transpose_1/ReadVariableOp¢tabl/transpose_4/ReadVariableOp
bl_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
bl_1/transpose/perm
bl_1/transpose	Transposeinputsbl_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(d2
bl_1/transposeZ

bl_1/ShapeShapebl_1/transpose:y:0*
T0*
_output_shapes
:2

bl_1/Shapek
bl_1/unstackUnpackbl_1/Shape:output:0*
T0*
_output_shapes
: : : *	
num2
bl_1/unstack
bl_1/Shape_1/ReadVariableOpReadVariableOp$bl_1_shape_1_readvariableop_resource*
_output_shapes

:dx*
dtype02
bl_1/Shape_1/ReadVariableOpm
bl_1/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"d   x   2
bl_1/Shape_1o
bl_1/unstack_1Unpackbl_1/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2
bl_1/unstack_1y
bl_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2
bl_1/Reshape/shape
bl_1/ReshapeReshapebl_1/transpose:y:0bl_1/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
bl_1/Reshape§
bl_1/transpose_1/ReadVariableOpReadVariableOp$bl_1_shape_1_readvariableop_resource*
_output_shapes

:dx*
dtype02!
bl_1/transpose_1/ReadVariableOp
bl_1/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
bl_1/transpose_1/perm£
bl_1/transpose_1	Transpose'bl_1/transpose_1/ReadVariableOp:value:0bl_1/transpose_1/perm:output:0*
T0*
_output_shapes

:dx2
bl_1/transpose_1}
bl_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"d   ÿÿÿÿ2
bl_1/Reshape_1/shape
bl_1/Reshape_1Reshapebl_1/transpose_1:y:0bl_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:dx2
bl_1/Reshape_1
bl_1/MatMulMatMulbl_1/Reshape:output:0bl_1/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
bl_1/MatMulr
bl_1/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :(2
bl_1/Reshape_2/shape/1r
bl_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :x2
bl_1/Reshape_2/shape/2»
bl_1/Reshape_2/shapePackbl_1/unstack:output:0bl_1/Reshape_2/shape/1:output:0bl_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
bl_1/Reshape_2/shape
bl_1/Reshape_2Reshapebl_1/MatMul:product:0bl_1/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(x2
bl_1/Reshape_2
bl_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
bl_1/transpose_2/perm 
bl_1/transpose_2	Transposebl_1/Reshape_2:output:0bl_1/transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx(2
bl_1/transpose_2`
bl_1/Shape_2Shapebl_1/transpose_2:y:0*
T0*
_output_shapes
:2
bl_1/Shape_2q
bl_1/unstack_2Unpackbl_1/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2
bl_1/unstack_2
bl_1/Shape_3/ReadVariableOpReadVariableOp$bl_1_shape_3_readvariableop_resource*
_output_shapes

:(*
dtype02
bl_1/Shape_3/ReadVariableOpm
bl_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"(      2
bl_1/Shape_3o
bl_1/unstack_3Unpackbl_1/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2
bl_1/unstack_3}
bl_1/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   2
bl_1/Reshape_3/shape
bl_1/Reshape_3Reshapebl_1/transpose_2:y:0bl_1/Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
bl_1/Reshape_3§
bl_1/transpose_3/ReadVariableOpReadVariableOp$bl_1_shape_3_readvariableop_resource*
_output_shapes

:(*
dtype02!
bl_1/transpose_3/ReadVariableOp
bl_1/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
bl_1/transpose_3/perm£
bl_1/transpose_3	Transpose'bl_1/transpose_3/ReadVariableOp:value:0bl_1/transpose_3/perm:output:0*
T0*
_output_shapes

:(2
bl_1/transpose_3}
bl_1/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"(   ÿÿÿÿ2
bl_1/Reshape_4/shape
bl_1/Reshape_4Reshapebl_1/transpose_3:y:0bl_1/Reshape_4/shape:output:0*
T0*
_output_shapes

:(2
bl_1/Reshape_4
bl_1/MatMul_1MatMulbl_1/Reshape_3:output:0bl_1/Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bl_1/MatMul_1r
bl_1/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :x2
bl_1/Reshape_5/shape/1r
bl_1/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
bl_1/Reshape_5/shape/2½
bl_1/Reshape_5/shapePackbl_1/unstack_2:output:0bl_1/Reshape_5/shape/1:output:0bl_1/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
bl_1/Reshape_5/shape
bl_1/Reshape_5Reshapebl_1/MatMul_1:product:0bl_1/Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
bl_1/Reshape_5¥
bl_1/Reshape_6/ReadVariableOpReadVariableOp&bl_1_reshape_6_readvariableop_resource*
_output_shapes

:x*
dtype02
bl_1/Reshape_6/ReadVariableOp
bl_1/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   x      2
bl_1/Reshape_6/shape
bl_1/Reshape_6Reshape%bl_1/Reshape_6/ReadVariableOp:value:0bl_1/Reshape_6/shape:output:0*
T0*"
_output_shapes
:x2
bl_1/Reshape_6
bl_1/addAddV2bl_1/Reshape_5:output:0bl_1/Reshape_6:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2

bl_1/addr
activation_1/ReluRelubl_1/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
activation_1/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_1/dropout/Const®
dropout_1/dropout/MulMulactivation_1/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
dropout_1/dropout/Mul
dropout_1/dropout/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/ShapeÖ
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 dropout_1/dropout/GreaterEqual/yê
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2 
dropout_1/dropout/GreaterEqual¡
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
dropout_1/dropout/Cast¦
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
dropout_1/dropout/Mul_1
bl_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
bl_2/transpose/perm
bl_2/transpose	Transposedropout_1/dropout/Mul_1:z:0bl_2/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
bl_2/transposeZ

bl_2/ShapeShapebl_2/transpose:y:0*
T0*
_output_shapes
:2

bl_2/Shapek
bl_2/unstackUnpackbl_2/Shape:output:0*
T0*
_output_shapes
: : : *	
num2
bl_2/unstack
bl_2/Shape_1/ReadVariableOpReadVariableOp$bl_2_shape_1_readvariableop_resource*
_output_shapes

:x<*
dtype02
bl_2/Shape_1/ReadVariableOpm
bl_2/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"x   <   2
bl_2/Shape_1o
bl_2/unstack_1Unpackbl_2/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2
bl_2/unstack_1y
bl_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿx   2
bl_2/Reshape/shape
bl_2/ReshapeReshapebl_2/transpose:y:0bl_2/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
bl_2/Reshape§
bl_2/transpose_1/ReadVariableOpReadVariableOp$bl_2_shape_1_readvariableop_resource*
_output_shapes

:x<*
dtype02!
bl_2/transpose_1/ReadVariableOp
bl_2/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
bl_2/transpose_1/perm£
bl_2/transpose_1	Transpose'bl_2/transpose_1/ReadVariableOp:value:0bl_2/transpose_1/perm:output:0*
T0*
_output_shapes

:x<2
bl_2/transpose_1}
bl_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"x   ÿÿÿÿ2
bl_2/Reshape_1/shape
bl_2/Reshape_1Reshapebl_2/transpose_1:y:0bl_2/Reshape_1/shape:output:0*
T0*
_output_shapes

:x<2
bl_2/Reshape_1
bl_2/MatMulMatMulbl_2/Reshape:output:0bl_2/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
bl_2/MatMulr
bl_2/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
bl_2/Reshape_2/shape/1r
bl_2/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :<2
bl_2/Reshape_2/shape/2»
bl_2/Reshape_2/shapePackbl_2/unstack:output:0bl_2/Reshape_2/shape/1:output:0bl_2/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
bl_2/Reshape_2/shape
bl_2/Reshape_2Reshapebl_2/MatMul:product:0bl_2/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
bl_2/Reshape_2
bl_2/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
bl_2/transpose_2/perm 
bl_2/transpose_2	Transposebl_2/Reshape_2:output:0bl_2/transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
bl_2/transpose_2`
bl_2/Shape_2Shapebl_2/transpose_2:y:0*
T0*
_output_shapes
:2
bl_2/Shape_2q
bl_2/unstack_2Unpackbl_2/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2
bl_2/unstack_2
bl_2/Shape_3/ReadVariableOpReadVariableOp$bl_2_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02
bl_2/Shape_3/ReadVariableOpm
bl_2/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      2
bl_2/Shape_3o
bl_2/unstack_3Unpackbl_2/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2
bl_2/unstack_3}
bl_2/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
bl_2/Reshape_3/shape
bl_2/Reshape_3Reshapebl_2/transpose_2:y:0bl_2/Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bl_2/Reshape_3§
bl_2/transpose_3/ReadVariableOpReadVariableOp$bl_2_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02!
bl_2/transpose_3/ReadVariableOp
bl_2/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
bl_2/transpose_3/perm£
bl_2/transpose_3	Transpose'bl_2/transpose_3/ReadVariableOp:value:0bl_2/transpose_3/perm:output:0*
T0*
_output_shapes

:2
bl_2/transpose_3}
bl_2/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
bl_2/Reshape_4/shape
bl_2/Reshape_4Reshapebl_2/transpose_3:y:0bl_2/Reshape_4/shape:output:0*
T0*
_output_shapes

:2
bl_2/Reshape_4
bl_2/MatMul_1MatMulbl_2/Reshape_3:output:0bl_2/Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bl_2/MatMul_1r
bl_2/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<2
bl_2/Reshape_5/shape/1r
bl_2/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
bl_2/Reshape_5/shape/2½
bl_2/Reshape_5/shapePackbl_2/unstack_2:output:0bl_2/Reshape_5/shape/1:output:0bl_2/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
bl_2/Reshape_5/shape
bl_2/Reshape_5Reshapebl_2/MatMul_1:product:0bl_2/Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
bl_2/Reshape_5¥
bl_2/Reshape_6/ReadVariableOpReadVariableOp&bl_2_reshape_6_readvariableop_resource*
_output_shapes

:<*
dtype02
bl_2/Reshape_6/ReadVariableOp
bl_2/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   <      2
bl_2/Reshape_6/shape
bl_2/Reshape_6Reshape%bl_2/Reshape_6/ReadVariableOp:value:0bl_2/Reshape_6/shape:output:0*
T0*"
_output_shapes
:<2
bl_2/Reshape_6
bl_2/addAddV2bl_2/Reshape_5:output:0bl_2/Reshape_6:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

bl_2/addr
activation_2/ReluRelubl_2/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
activation_2/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_2/dropout/Const®
dropout_2/dropout/MulMulactivation_2/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout_2/dropout/Mul
dropout_2/dropout/ShapeShapeactivation_2/Relu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/ShapeÖ
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 dropout_2/dropout/GreaterEqual/yê
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2 
dropout_2/dropout/GreaterEqual¡
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout_2/dropout/Cast¦
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout_2/dropout/Mul_1
tabl/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
tabl/transpose/perm
tabl/transpose	Transposedropout_2/dropout/Mul_1:z:0tabl/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
tabl/transposeZ

tabl/ShapeShapetabl/transpose:y:0*
T0*
_output_shapes
:2

tabl/Shapek
tabl/unstackUnpacktabl/Shape:output:0*
T0*
_output_shapes
: : : *	
num2
tabl/unstack
tabl/Shape_1/ReadVariableOpReadVariableOp$tabl_shape_1_readvariableop_resource*
_output_shapes

:<*
dtype02
tabl/Shape_1/ReadVariableOpm
tabl/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"<      2
tabl/Shape_1o
tabl/unstack_1Unpacktabl/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2
tabl/unstack_1y
tabl/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ<   2
tabl/Reshape/shape
tabl/ReshapeReshapetabl/transpose:y:0tabl/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
tabl/Reshape§
tabl/transpose_1/ReadVariableOpReadVariableOp$tabl_shape_1_readvariableop_resource*
_output_shapes

:<*
dtype02!
tabl/transpose_1/ReadVariableOp
tabl/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
tabl/transpose_1/perm£
tabl/transpose_1	Transpose'tabl/transpose_1/ReadVariableOp:value:0tabl/transpose_1/perm:output:0*
T0*
_output_shapes

:<2
tabl/transpose_1}
tabl/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"<   ÿÿÿÿ2
tabl/Reshape_1/shape
tabl/Reshape_1Reshapetabl/transpose_1:y:0tabl/Reshape_1/shape:output:0*
T0*
_output_shapes

:<2
tabl/Reshape_1
tabl/MatMulMatMultabl/Reshape:output:0tabl/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tabl/MatMulr
tabl/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
tabl/Reshape_2/shape/1r
tabl/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
tabl/Reshape_2/shape/2»
tabl/Reshape_2/shapePacktabl/unstack:output:0tabl/Reshape_2/shape/1:output:0tabl/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
tabl/Reshape_2/shape
tabl/Reshape_2Reshapetabl/MatMul:product:0tabl/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tabl/Reshape_2
tabl/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
tabl/transpose_2/perm 
tabl/transpose_2	Transposetabl/Reshape_2:output:0tabl/transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tabl/transpose_2k
tabl/eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2
tabl/eye/onesd
tabl/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
tabl/eye/diag/k{
tabl/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tabl/eye/diag/num_rows{
tabl/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tabl/eye/diag/num_cols
tabl/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
tabl/eye/diag/padding_valueñ
tabl/eye/diagMatrixDiagV3tabl/eye/ones:output:0tabl/eye/diag/k:output:0tabl/eye/diag/num_rows:output:0tabl/eye/diag/num_cols:output:0$tabl/eye/diag/padding_value:output:0*
T0*
_output_shapes

:2
tabl/eye/diag
tabl/ReadVariableOpReadVariableOptabl_readvariableop_resource*
_output_shapes

:*
dtype02
tabl/ReadVariableOpy
tabl/mulMultabl/ReadVariableOp:value:0tabl/eye/diag:output:0*
T0*
_output_shapes

:2

tabl/mul
tabl/ReadVariableOp_1ReadVariableOptabl_readvariableop_resource*
_output_shapes

:*
dtype02
tabl/ReadVariableOp_1q
tabl/subSubtabl/ReadVariableOp_1:value:0tabl/mul:z:0*
T0*
_output_shapes

:2

tabl/subo
tabl/eye_1/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2
tabl/eye_1/onesh
tabl/eye_1/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
tabl/eye_1/diag/k
tabl/eye_1/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tabl/eye_1/diag/num_rows
tabl/eye_1/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tabl/eye_1/diag/num_cols
tabl/eye_1/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
tabl/eye_1/diag/padding_valueÿ
tabl/eye_1/diagMatrixDiagV3tabl/eye_1/ones:output:0tabl/eye_1/diag/k:output:0!tabl/eye_1/diag/num_rows:output:0!tabl/eye_1/diag/num_cols:output:0&tabl/eye_1/diag/padding_value:output:0*
T0*
_output_shapes

:2
tabl/eye_1/diage
tabl/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tabl/truediv/y
tabl/truedivRealDivtabl/eye_1/diag:output:0tabl/truediv/y:output:0*
T0*
_output_shapes

:2
tabl/truedivf
tabl/addAddV2tabl/sub:z:0tabl/truediv:z:0*
T0*
_output_shapes

:2

tabl/add`
tabl/Shape_2Shapetabl/transpose_2:y:0*
T0*
_output_shapes
:2
tabl/Shape_2q
tabl/unstack_2Unpacktabl/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2
tabl/unstack_2m
tabl/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      2
tabl/Shape_3o
tabl/unstack_3Unpacktabl/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2
tabl/unstack_3}
tabl/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
tabl/Reshape_3/shape
tabl/Reshape_3Reshapetabl/transpose_2:y:0tabl/Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tabl/Reshape_3
tabl/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
tabl/transpose_3/perm
tabl/transpose_3	Transposetabl/add:z:0tabl/transpose_3/perm:output:0*
T0*
_output_shapes

:2
tabl/transpose_3}
tabl/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
tabl/Reshape_4/shape
tabl/Reshape_4Reshapetabl/transpose_3:y:0tabl/Reshape_4/shape:output:0*
T0*
_output_shapes

:2
tabl/Reshape_4
tabl/MatMul_1MatMultabl/Reshape_3:output:0tabl/Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tabl/MatMul_1r
tabl/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
tabl/Reshape_5/shape/1r
tabl/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
tabl/Reshape_5/shape/2½
tabl/Reshape_5/shapePacktabl/unstack_2:output:0tabl/Reshape_5/shape/1:output:0tabl/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
tabl/Reshape_5/shape
tabl/Reshape_5Reshapetabl/MatMul_1:product:0tabl/Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tabl/Reshape_5
tabl/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tabl/Max/reduction_indices 
tabl/MaxMaxtabl/Reshape_5:output:0#tabl/Max/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2

tabl/Max

tabl/sub_1Subtabl/Reshape_5:output:0tabl/Max:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

tabl/sub_1a
tabl/ExpExptabl/sub_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

tabl/Exp
tabl/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
tabl/Sum/reduction_indices
tabl/SumSumtabl/Exp:y:0#tabl/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2

tabl/Sum
tabl/truediv_1RealDivtabl/Exp:y:0tabl/Sum:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tabl/truediv_1
tabl/ReadVariableOp_2ReadVariableOptabl_readvariableop_2_resource*
_output_shapes
:*
dtype02
tabl/ReadVariableOp_2

tabl/mul_1Multabl/ReadVariableOp_2:value:0tabl/transpose_2:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

tabl/mul_1
tabl/ReadVariableOp_3ReadVariableOptabl_readvariableop_2_resource*
_output_shapes
:*
dtype02
tabl/ReadVariableOp_3a
tabl/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tabl/sub_2/xz

tabl/sub_2Subtabl/sub_2/x:output:0tabl/ReadVariableOp_3:value:0*
T0*
_output_shapes
:2

tabl/sub_2{

tabl/mul_2Multabl/sub_2:z:0tabl/transpose_2:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

tabl/mul_2y

tabl/mul_3Multabl/mul_2:z:0tabl/truediv_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

tabl/mul_3w

tabl/add_1AddV2tabl/mul_1:z:0tabl/mul_3:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

tabl/add_1Z
tabl/Shape_4Shapetabl/add_1:z:0*
T0*
_output_shapes
:2
tabl/Shape_4q
tabl/unstack_4Unpacktabl/Shape_4:output:0*
T0*
_output_shapes
: : : *	
num2
tabl/unstack_4
tabl/Shape_5/ReadVariableOpReadVariableOp$tabl_shape_5_readvariableop_resource*
_output_shapes

:*
dtype02
tabl/Shape_5/ReadVariableOpm
tabl/Shape_5Const*
_output_shapes
:*
dtype0*
valueB"      2
tabl/Shape_5o
tabl/unstack_5Unpacktabl/Shape_5:output:0*
T0*
_output_shapes
: : *	
num2
tabl/unstack_5}
tabl/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
tabl/Reshape_6/shape
tabl/Reshape_6Reshapetabl/add_1:z:0tabl/Reshape_6/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tabl/Reshape_6§
tabl/transpose_4/ReadVariableOpReadVariableOp$tabl_shape_5_readvariableop_resource*
_output_shapes

:*
dtype02!
tabl/transpose_4/ReadVariableOp
tabl/transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
tabl/transpose_4/perm£
tabl/transpose_4	Transpose'tabl/transpose_4/ReadVariableOp:value:0tabl/transpose_4/perm:output:0*
T0*
_output_shapes

:2
tabl/transpose_4}
tabl/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
tabl/Reshape_7/shape
tabl/Reshape_7Reshapetabl/transpose_4:y:0tabl/Reshape_7/shape:output:0*
T0*
_output_shapes

:2
tabl/Reshape_7
tabl/MatMul_2MatMultabl/Reshape_6:output:0tabl/Reshape_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tabl/MatMul_2r
tabl/Reshape_8/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
tabl/Reshape_8/shape/1r
tabl/Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
tabl/Reshape_8/shape/2½
tabl/Reshape_8/shapePacktabl/unstack_4:output:0tabl/Reshape_8/shape/1:output:0tabl/Reshape_8/shape/2:output:0*
N*
T0*
_output_shapes
:2
tabl/Reshape_8/shape
tabl/Reshape_8Reshapetabl/MatMul_2:product:0tabl/Reshape_8/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tabl/Reshape_8
tabl/add_2/ReadVariableOpReadVariableOp"tabl_add_2_readvariableop_resource*"
_output_shapes
:*
dtype02
tabl/add_2/ReadVariableOp

tabl/add_2AddV2tabl/Reshape_8:output:0!tabl/add_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

tabl/add_2
tabl/SqueezeSqueezetabl/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2
tabl/Squeeze
activation_3/SoftmaxSoftmaxtabl/Squeeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
activation_3/Softmaxø
IdentityIdentityactivation_3/Softmax:softmax:0^bl_1/Reshape_6/ReadVariableOp ^bl_1/transpose_1/ReadVariableOp ^bl_1/transpose_3/ReadVariableOp^bl_2/Reshape_6/ReadVariableOp ^bl_2/transpose_1/ReadVariableOp ^bl_2/transpose_3/ReadVariableOp^tabl/ReadVariableOp^tabl/ReadVariableOp_1^tabl/ReadVariableOp_2^tabl/ReadVariableOp_3^tabl/add_2/ReadVariableOp ^tabl/transpose_1/ReadVariableOp ^tabl/transpose_4/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿd(:::::::::::2>
bl_1/Reshape_6/ReadVariableOpbl_1/Reshape_6/ReadVariableOp2B
bl_1/transpose_1/ReadVariableOpbl_1/transpose_1/ReadVariableOp2B
bl_1/transpose_3/ReadVariableOpbl_1/transpose_3/ReadVariableOp2>
bl_2/Reshape_6/ReadVariableOpbl_2/Reshape_6/ReadVariableOp2B
bl_2/transpose_1/ReadVariableOpbl_2/transpose_1/ReadVariableOp2B
bl_2/transpose_3/ReadVariableOpbl_2/transpose_3/ReadVariableOp2*
tabl/ReadVariableOptabl/ReadVariableOp2.
tabl/ReadVariableOp_1tabl/ReadVariableOp_12.
tabl/ReadVariableOp_2tabl/ReadVariableOp_22.
tabl/ReadVariableOp_3tabl/ReadVariableOp_326
tabl/add_2/ReadVariableOptabl/add_2/ReadVariableOp2B
tabl/transpose_1/ReadVariableOptabl/transpose_1/ReadVariableOp2B
tabl/transpose_4/ReadVariableOptabl/transpose_4/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
 
_user_specified_nameinputs

I
-__inference_activation_3_layer_call_fn_443693

inputs
identityË
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_4426532
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
c
*__inference_dropout_2_layer_call_fn_443555

inputs
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_4424942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
¹.
 
@__inference_bl_2_layer_call_and_return_conditional_losses_442449
x#
shape_1_readvariableop_resource#
shape_3_readvariableop_resource%
!reshape_6_readvariableop_resource
identity¢Reshape_6/ReadVariableOp¢transpose_1/ReadVariableOp¢transpose_3/ReadVariableOpu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permu
	transpose	Transposextranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shape\
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:x<*
dtype02
Shape_1/ReadVariableOpc
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"x   <   2	
Shape_1`
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿx   2
Reshape/shapev
ReshapeReshapetranspose:y:0Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2	
Reshape
transpose_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:x<*
dtype02
transpose_1/ReadVariableOpu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

:x<2
transpose_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"x   ÿÿÿÿ2
Reshape_1/shapeu
	Reshape_1Reshapetranspose_1:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:x<2
	Reshape_1r
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
MatMulh
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_2/shape/1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :<2
Reshape_2/shape/2¢
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
	Reshape_2y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm
transpose_2	TransposeReshape_2:output:0transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
transpose_2Q
Shape_2Shapetranspose_2:y:0*
T0*
_output_shapes
:2	
Shape_2b
	unstack_2UnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
num2
	unstack_2
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:*
dtype02
Shape_3/ReadVariableOpc
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      2	
Shape_3`
	unstack_3UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_3s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Reshape_3/shape~
	Reshape_3Reshapetranspose_2:y:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_3
transpose_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:*
dtype02
transpose_3/ReadVariableOpu
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm
transpose_3	Transpose"transpose_3/ReadVariableOp:value:0transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3s
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
Reshape_4/shapeu
	Reshape_4Reshapetranspose_3:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

:2
	Reshape_4x
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1h
Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<2
Reshape_5/shape/1h
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_5/shape/2¤
Reshape_5/shapePackunstack_2:output:0Reshape_5/shape/1:output:0Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_5/shape
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
	Reshape_5
Reshape_6/ReadVariableOpReadVariableOp!reshape_6_readvariableop_resource*
_output_shapes

:<*
dtype02
Reshape_6/ReadVariableOpw
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   <      2
Reshape_6/shape
	Reshape_6Reshape Reshape_6/ReadVariableOp:value:0Reshape_6/shape:output:0*
T0*"
_output_shapes
:<2
	Reshape_6q
addAddV2Reshape_5:output:0Reshape_6:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
add´
IdentityIdentityadd:z:0^Reshape_6/ReadVariableOp^transpose_1/ReadVariableOp^transpose_3/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿx:::24
Reshape_6/ReadVariableOpReshape_6/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp28
transpose_3/ReadVariableOptranspose_3/ReadVariableOp:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx

_user_specified_namex
¢
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_442494

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¸
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÂ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
	

&__inference_model_layer_call_fn_442819
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4427942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿd(:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
!
_user_specified_name	input_1
Â¥

"__inference__traced_restore_443966
file_prefix
assignvariableop_bl_1_w1
assignvariableop_1_bl_1_w2 
assignvariableop_2_bl_1_bias
assignvariableop_3_bl_2_w1
assignvariableop_4_bl_2_w2 
assignvariableop_5_bl_2_bias
assignvariableop_6_tabl_w1
assignvariableop_7_tabl_w2
assignvariableop_8_tabl_w!
assignvariableop_9_tabl_alpha!
assignvariableop_10_tabl_bias!
assignvariableop_11_adam_iter#
assignvariableop_12_adam_beta_1#
assignvariableop_13_adam_beta_2"
assignvariableop_14_adam_decay*
&assignvariableop_15_adam_learning_rate
assignvariableop_16_total
assignvariableop_17_count&
"assignvariableop_18_adam_bl_1_w1_m&
"assignvariableop_19_adam_bl_1_w2_m(
$assignvariableop_20_adam_bl_1_bias_m&
"assignvariableop_21_adam_bl_2_w1_m&
"assignvariableop_22_adam_bl_2_w2_m(
$assignvariableop_23_adam_bl_2_bias_m&
"assignvariableop_24_adam_tabl_w1_m&
"assignvariableop_25_adam_tabl_w2_m%
!assignvariableop_26_adam_tabl_w_m)
%assignvariableop_27_adam_tabl_alpha_m(
$assignvariableop_28_adam_tabl_bias_m&
"assignvariableop_29_adam_bl_1_w1_v&
"assignvariableop_30_adam_bl_1_w2_v(
$assignvariableop_31_adam_bl_1_bias_v&
"assignvariableop_32_adam_bl_2_w1_v&
"assignvariableop_33_adam_bl_2_w2_v(
$assignvariableop_34_adam_bl_2_bias_v&
"assignvariableop_35_adam_tabl_w1_v&
"assignvariableop_36_adam_tabl_w2_v%
!assignvariableop_37_adam_tabl_w_v)
%assignvariableop_38_adam_tabl_alpha_v(
$assignvariableop_39_adam_tabl_bias_v
identity_41¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¬
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*¸
value®B«)B2layer_with_weights-0/W1/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-0/W2/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/W1/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/W2/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/W1/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/W2/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-2/W/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/alpha/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-0/W1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-0/W2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/W1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/W2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/W1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/W2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-2/W/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-0/W1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-0/W2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/W1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/W2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/W1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/W2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-2/W/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesà
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesû
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*º
_output_shapes§
¤:::::::::::::::::::::::::::::::::::::::::*7
dtypes-
+2)	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_bl_1_w1Identity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOpassignvariableop_1_bl_1_w2Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¡
AssignVariableOp_2AssignVariableOpassignvariableop_2_bl_1_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOpassignvariableop_3_bl_2_w1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOpassignvariableop_4_bl_2_w2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¡
AssignVariableOp_5AssignVariableOpassignvariableop_5_bl_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOpassignvariableop_6_tabl_w1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOpassignvariableop_7_tabl_w2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOpassignvariableop_8_tabl_wIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¢
AssignVariableOp_9AssignVariableOpassignvariableop_9_tabl_alphaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¥
AssignVariableOp_10AssignVariableOpassignvariableop_10_tabl_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_11¥
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_iterIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12§
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13§
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¦
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_decayIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15®
AssignVariableOp_15AssignVariableOp&assignvariableop_15_adam_learning_rateIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¡
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¡
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18ª
AssignVariableOp_18AssignVariableOp"assignvariableop_18_adam_bl_1_w1_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19ª
AssignVariableOp_19AssignVariableOp"assignvariableop_19_adam_bl_1_w2_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¬
AssignVariableOp_20AssignVariableOp$assignvariableop_20_adam_bl_1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21ª
AssignVariableOp_21AssignVariableOp"assignvariableop_21_adam_bl_2_w1_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22ª
AssignVariableOp_22AssignVariableOp"assignvariableop_22_adam_bl_2_w2_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¬
AssignVariableOp_23AssignVariableOp$assignvariableop_23_adam_bl_2_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24ª
AssignVariableOp_24AssignVariableOp"assignvariableop_24_adam_tabl_w1_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25ª
AssignVariableOp_25AssignVariableOp"assignvariableop_25_adam_tabl_w2_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26©
AssignVariableOp_26AssignVariableOp!assignvariableop_26_adam_tabl_w_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27­
AssignVariableOp_27AssignVariableOp%assignvariableop_27_adam_tabl_alpha_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28¬
AssignVariableOp_28AssignVariableOp$assignvariableop_28_adam_tabl_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29ª
AssignVariableOp_29AssignVariableOp"assignvariableop_29_adam_bl_1_w1_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30ª
AssignVariableOp_30AssignVariableOp"assignvariableop_30_adam_bl_1_w2_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31¬
AssignVariableOp_31AssignVariableOp$assignvariableop_31_adam_bl_1_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32ª
AssignVariableOp_32AssignVariableOp"assignvariableop_32_adam_bl_2_w1_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33ª
AssignVariableOp_33AssignVariableOp"assignvariableop_33_adam_bl_2_w2_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34¬
AssignVariableOp_34AssignVariableOp$assignvariableop_34_adam_bl_2_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35ª
AssignVariableOp_35AssignVariableOp"assignvariableop_35_adam_tabl_w1_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36ª
AssignVariableOp_36AssignVariableOp"assignvariableop_36_adam_tabl_w2_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37©
AssignVariableOp_37AssignVariableOp!assignvariableop_37_adam_tabl_w_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38­
AssignVariableOp_38AssignVariableOp%assignvariableop_38_adam_tabl_alpha_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¬
AssignVariableOp_39AssignVariableOp$assignvariableop_39_adam_tabl_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_399
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÎ
Identity_40Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_40Á
Identity_41IdentityIdentity_40:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_41"#
identity_41Identity_41:output:0*·
_input_shapes¥
¢: ::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¹.
 
@__inference_bl_1_layer_call_and_return_conditional_losses_442331
x#
shape_1_readvariableop_resource#
shape_3_readvariableop_resource%
!reshape_6_readvariableop_resource
identity¢Reshape_6/ReadVariableOp¢transpose_1/ReadVariableOp¢transpose_3/ReadVariableOpu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permu
	transpose	Transposextranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(d2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shape\
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:dx*
dtype02
Shape_1/ReadVariableOpc
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"d   x   2	
Shape_1`
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2
Reshape/shapev
ReshapeReshapetranspose:y:0Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
Reshape
transpose_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:dx*
dtype02
transpose_1/ReadVariableOpu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

:dx2
transpose_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"d   ÿÿÿÿ2
Reshape_1/shapeu
	Reshape_1Reshapetranspose_1:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:dx2
	Reshape_1r
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
MatMulh
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :(2
Reshape_2/shape/1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :x2
Reshape_2/shape/2¢
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(x2
	Reshape_2y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm
transpose_2	TransposeReshape_2:output:0transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx(2
transpose_2Q
Shape_2Shapetranspose_2:y:0*
T0*
_output_shapes
:2	
Shape_2b
	unstack_2UnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
num2
	unstack_2
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:(*
dtype02
Shape_3/ReadVariableOpc
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"(      2	
Shape_3`
	unstack_3UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_3s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   2
Reshape_3/shape~
	Reshape_3Reshapetranspose_2:y:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
	Reshape_3
transpose_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:(*
dtype02
transpose_3/ReadVariableOpu
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm
transpose_3	Transpose"transpose_3/ReadVariableOp:value:0transpose_3/perm:output:0*
T0*
_output_shapes

:(2
transpose_3s
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"(   ÿÿÿÿ2
Reshape_4/shapeu
	Reshape_4Reshapetranspose_3:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

:(2
	Reshape_4x
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1h
Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :x2
Reshape_5/shape/1h
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_5/shape/2¤
Reshape_5/shapePackunstack_2:output:0Reshape_5/shape/1:output:0Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_5/shape
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
	Reshape_5
Reshape_6/ReadVariableOpReadVariableOp!reshape_6_readvariableop_resource*
_output_shapes

:x*
dtype02
Reshape_6/ReadVariableOpw
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   x      2
Reshape_6/shape
	Reshape_6Reshape Reshape_6/ReadVariableOp:value:0Reshape_6/shape:output:0*
T0*"
_output_shapes
:x2
	Reshape_6q
addAddV2Reshape_5:output:0Reshape_6:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
add´
IdentityIdentityadd:z:0^Reshape_6/ReadVariableOp^transpose_1/ReadVariableOp^transpose_3/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿd(:::24
Reshape_6/ReadVariableOpReshape_6/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp28
transpose_3/ReadVariableOptranspose_3/ReadVariableOp:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(

_user_specified_namex
¯
I
-__inference_activation_1_layer_call_fn_443430

inputs
identityÏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_4423562
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿx:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
¢
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_442376

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¸
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÂ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿx:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
µ
c
*__inference_dropout_1_layer_call_fn_443452

inputs
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_4423762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿx22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
Ø
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_442381

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿx:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
¢
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_443545

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¸
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÂ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
Æ
d
H__inference_activation_2_layer_call_and_return_conditional_losses_443528

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
Æ
d
H__inference_activation_2_layer_call_and_return_conditional_losses_442474

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
!
ø
A__inference_model_layer_call_and_return_conditional_losses_442794

inputs
bl_1_442763
bl_1_442765
bl_1_442767
bl_2_442772
bl_2_442774
bl_2_442776
tabl_442781
tabl_442783
tabl_442785
tabl_442787
tabl_442789
identity¢bl_1/StatefulPartitionedCall¢bl_2/StatefulPartitionedCall¢tabl/StatefulPartitionedCall
bl_1/StatefulPartitionedCallStatefulPartitionedCallinputsbl_1_442763bl_1_442765bl_1_442767*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_bl_1_layer_call_and_return_conditional_losses_4423312
bl_1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall%bl_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_4423562
activation_1/PartitionedCallÿ
dropout_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_4423812
dropout_1/PartitionedCall´
bl_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0bl_2_442772bl_2_442774bl_2_442776*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_bl_2_layer_call_and_return_conditional_losses_4424492
bl_2/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall%bl_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_4424742
activation_2/PartitionedCallÿ
dropout_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_4424992
dropout_2/PartitionedCallÎ
tabl/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0tabl_442781tabl_442783tabl_442785tabl_442787tabl_442789*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_tabl_layer_call_and_return_conditional_losses_4426202
tabl/StatefulPartitionedCall
activation_3/PartitionedCallPartitionedCall%tabl/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_4426532
activation_3/PartitionedCallÖ
IdentityIdentity%activation_3/PartitionedCall:output:0^bl_1/StatefulPartitionedCall^bl_2/StatefulPartitionedCall^tabl/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿd(:::::::::::2<
bl_1/StatefulPartitionedCallbl_1/StatefulPartitionedCall2<
bl_2/StatefulPartitionedCallbl_2/StatefulPartitionedCall2<
tabl/StatefulPartitionedCalltabl/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
 
_user_specified_nameinputs
Æ
d
H__inference_activation_1_layer_call_and_return_conditional_losses_443425

inputs
identityR
ReluReluinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿx:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
	

&__inference_model_layer_call_fn_443327

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4427332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿd(:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
 
_user_specified_nameinputs
Í\

@__inference_tabl_layer_call_and_return_conditional_losses_443668
x#
shape_1_readvariableop_resource
readvariableop_resource
readvariableop_2_resource#
shape_5_readvariableop_resource!
add_2_readvariableop_resource
identity¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢add_2/ReadVariableOp¢transpose_1/ReadVariableOp¢transpose_4/ReadVariableOpu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permu
	transpose	Transposextranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shape\
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:<*
dtype02
Shape_1/ReadVariableOpc
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"<      2	
Shape_1`
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ<   2
Reshape/shapev
ReshapeReshapetranspose:y:0Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2	
Reshape
transpose_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:<*
dtype02
transpose_1/ReadVariableOpu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

:<2
transpose_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"<   ÿÿÿÿ2
Reshape_1/shapeu
	Reshape_1Reshapetranspose_1:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:<2
	Reshape_1r
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulh
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_2/shape/1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_2/shape/2¢
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_2y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm
transpose_2	TransposeReshape_2:output:0transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
transpose_2a
eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2

eye/onesZ

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

eye/diag/kq
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye/diag/num_rowsq
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye/diag/num_colsu
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye/diag/padding_valueÎ
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*
_output_shapes

:2

eye/diagx
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype02
ReadVariableOpe
mulMulReadVariableOp:value:0eye/diag:output:0*
T0*
_output_shapes

:2
mul|
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype02
ReadVariableOp_1]
subSubReadVariableOp_1:value:0mul:z:0*
T0*
_output_shapes

:2
sube

eye_1/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2

eye_1/ones^
eye_1/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
eye_1/diag/ku
eye_1/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye_1/diag/num_rowsu
eye_1/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
eye_1/diag/num_colsy
eye_1/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
eye_1/diag/padding_valueÜ

eye_1/diagMatrixDiagV3eye_1/ones:output:0eye_1/diag/k:output:0eye_1/diag/num_rows:output:0eye_1/diag/num_cols:output:0!eye_1/diag/padding_value:output:0*
T0*
_output_shapes

:2

eye_1/diag[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/yo
truedivRealDiveye_1/diag:output:0truediv/y:output:0*
T0*
_output_shapes

:2	
truedivR
addAddV2sub:z:0truediv:z:0*
T0*
_output_shapes

:2
addQ
Shape_2Shapetranspose_2:y:0*
T0*
_output_shapes
:2	
Shape_2b
	unstack_2UnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
num2
	unstack_2c
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      2	
Shape_3`
	unstack_3UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_3s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Reshape_3/shape~
	Reshape_3Reshapetranspose_2:y:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_3u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/permt
transpose_3	Transposeadd:z:0transpose_3/perm:output:0*
T0*
_output_shapes

:2
transpose_3s
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
Reshape_4/shapeu
	Reshape_4Reshapetranspose_3:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

:2
	Reshape_4x
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1h
Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_5/shape/1h
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_5/shape/2¤
Reshape_5/shapePackunstack_2:output:0Reshape_5/shape/1:output:0Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_5/shape
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_5y
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Max/reduction_indices
MaxMaxReshape_5:output:0Max/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Maxm
sub_1SubReshape_5:output:0Max:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub_1R
ExpExp	sub_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Expy
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Sum/reduction_indices
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
Sumn
	truediv_1RealDivExp:y:0Sum:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	truediv_1z
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype02
ReadVariableOp_2v
mul_1MulReadVariableOp_2:value:0transpose_2:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1z
ReadVariableOp_3ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype02
ReadVariableOp_3W
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
sub_2/xf
sub_2Subsub_2/x:output:0ReadVariableOp_3:value:0*
T0*
_output_shapes
:2
sub_2g
mul_2Mul	sub_2:z:0transpose_2:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
mul_3Mul	mul_2:z:0truediv_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_3c
add_1AddV2	mul_1:z:0	mul_3:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1K
Shape_4Shape	add_1:z:0*
T0*
_output_shapes
:2	
Shape_4b
	unstack_4UnpackShape_4:output:0*
T0*
_output_shapes
: : : *	
num2
	unstack_4
Shape_5/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

:*
dtype02
Shape_5/ReadVariableOpc
Shape_5Const*
_output_shapes
:*
dtype0*
valueB"      2	
Shape_5`
	unstack_5UnpackShape_5:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_5s
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Reshape_6/shapex
	Reshape_6Reshape	add_1:z:0Reshape_6/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_6
transpose_4/ReadVariableOpReadVariableOpshape_5_readvariableop_resource*
_output_shapes

:*
dtype02
transpose_4/ReadVariableOpu
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transpose"transpose_4/ReadVariableOp:value:0transpose_4/perm:output:0*
T0*
_output_shapes

:2
transpose_4s
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
Reshape_7/shapeu
	Reshape_7Reshapetranspose_4:y:0Reshape_7/shape:output:0*
T0*
_output_shapes

:2
	Reshape_7x
MatMul_2MatMulReshape_6:output:0Reshape_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_2h
Reshape_8/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_8/shape/1h
Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_8/shape/2¤
Reshape_8/shapePackunstack_4:output:0Reshape_8/shape/1:output:0Reshape_8/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_8/shape
	Reshape_8ReshapeMatMul_2:product:0Reshape_8/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Reshape_8
add_2/ReadVariableOpReadVariableOpadd_2_readvariableop_resource*"
_output_shapes
:*
dtype02
add_2/ReadVariableOp
add_2AddV2Reshape_8:output:0add_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2z
SqueezeSqueeze	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2	
Squeezeÿ
IdentityIdentitySqueeze:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^add_2/ReadVariableOp^transpose_1/ReadVariableOp^transpose_4/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ<:::::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
add_2/ReadVariableOpadd_2/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp28
transpose_4/ReadVariableOptranspose_4/ReadVariableOp:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<

_user_specified_namex
å

$__inference_signature_wrapper_442856
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 **
f%R#
!__inference__wrapped_model_4422722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿd(:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
!
_user_specified_name	input_1
¢$
À
A__inference_model_layer_call_and_return_conditional_losses_442733

inputs
bl_1_442702
bl_1_442704
bl_1_442706
bl_2_442711
bl_2_442713
bl_2_442715
tabl_442720
tabl_442722
tabl_442724
tabl_442726
tabl_442728
identity¢bl_1/StatefulPartitionedCall¢bl_2/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢tabl/StatefulPartitionedCall
bl_1/StatefulPartitionedCallStatefulPartitionedCallinputsbl_1_442702bl_1_442704bl_1_442706*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_bl_1_layer_call_and_return_conditional_losses_4423312
bl_1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall%bl_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_4423562
activation_1/PartitionedCall
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_4423762#
!dropout_1/StatefulPartitionedCall¼
bl_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0bl_2_442711bl_2_442713bl_2_442715*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_bl_2_layer_call_and_return_conditional_losses_4424492
bl_2/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall%bl_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_4424742
activation_2/PartitionedCall»
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_4424942#
!dropout_2/StatefulPartitionedCallÖ
tabl/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0tabl_442720tabl_442722tabl_442724tabl_442726tabl_442728*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_tabl_layer_call_and_return_conditional_losses_4426202
tabl/StatefulPartitionedCall
activation_3/PartitionedCallPartitionedCall%tabl/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_4426532
activation_3/PartitionedCall
IdentityIdentity%activation_3/PartitionedCall:output:0^bl_1/StatefulPartitionedCall^bl_2/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall^tabl/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿd(:::::::::::2<
bl_1/StatefulPartitionedCallbl_1/StatefulPartitionedCall2<
bl_2/StatefulPartitionedCallbl_2/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2<
tabl/StatefulPartitionedCalltabl/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
 
_user_specified_nameinputs
û

%__inference_bl_2_layer_call_fn_443523
x
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_bl_2_layer_call_and_return_conditional_losses_4424492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿx:::22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx

_user_specified_namex
	

&__inference_model_layer_call_fn_443354

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4427942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿd(:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
 
_user_specified_nameinputs
¥$
Á
A__inference_model_layer_call_and_return_conditional_losses_442662
input_1
bl_1_442344
bl_1_442346
bl_1_442348
bl_2_442462
bl_2_442464
bl_2_442466
tabl_442637
tabl_442639
tabl_442641
tabl_442643
tabl_442645
identity¢bl_1/StatefulPartitionedCall¢bl_2/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢tabl/StatefulPartitionedCall
bl_1/StatefulPartitionedCallStatefulPartitionedCallinput_1bl_1_442344bl_1_442346bl_1_442348*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_bl_1_layer_call_and_return_conditional_losses_4423312
bl_1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall%bl_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_4423562
activation_1/PartitionedCall
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_4423762#
!dropout_1/StatefulPartitionedCall¼
bl_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0bl_2_442462bl_2_442464bl_2_442466*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_bl_2_layer_call_and_return_conditional_losses_4424492
bl_2/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall%bl_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_4424742
activation_2/PartitionedCall»
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_4424942#
!dropout_2/StatefulPartitionedCallÖ
tabl/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0tabl_442637tabl_442639tabl_442641tabl_442643tabl_442645*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_tabl_layer_call_and_return_conditional_losses_4426202
tabl/StatefulPartitionedCall
activation_3/PartitionedCallPartitionedCall%tabl/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_4426532
activation_3/PartitionedCall
IdentityIdentity%activation_3/PartitionedCall:output:0^bl_1/StatefulPartitionedCall^bl_2/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall^tabl/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿd(:::::::::::2<
bl_1/StatefulPartitionedCallbl_1/StatefulPartitionedCall2<
bl_2/StatefulPartitionedCallbl_2/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2<
tabl/StatefulPartitionedCalltabl/StatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
!
_user_specified_name	input_1
¹.
 
@__inference_bl_1_layer_call_and_return_conditional_losses_443409
x#
shape_1_readvariableop_resource#
shape_3_readvariableop_resource%
!reshape_6_readvariableop_resource
identity¢Reshape_6/ReadVariableOp¢transpose_1/ReadVariableOp¢transpose_3/ReadVariableOpu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permu
	transpose	Transposextranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(d2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shape\
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:dx*
dtype02
Shape_1/ReadVariableOpc
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"d   x   2	
Shape_1`
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2
Reshape/shapev
ReshapeReshapetranspose:y:0Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2	
Reshape
transpose_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:dx*
dtype02
transpose_1/ReadVariableOpu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0*
_output_shapes

:dx2
transpose_1s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"d   ÿÿÿÿ2
Reshape_1/shapeu
	Reshape_1Reshapetranspose_1:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:dx2
	Reshape_1r
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
MatMulh
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :(2
Reshape_2/shape/1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :x2
Reshape_2/shape/2¢
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(x2
	Reshape_2y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm
transpose_2	TransposeReshape_2:output:0transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx(2
transpose_2Q
Shape_2Shapetranspose_2:y:0*
T0*
_output_shapes
:2	
Shape_2b
	unstack_2UnpackShape_2:output:0*
T0*
_output_shapes
: : : *	
num2
	unstack_2
Shape_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:(*
dtype02
Shape_3/ReadVariableOpc
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"(      2	
Shape_3`
	unstack_3UnpackShape_3:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_3s
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   2
Reshape_3/shape~
	Reshape_3Reshapetranspose_2:y:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
	Reshape_3
transpose_3/ReadVariableOpReadVariableOpshape_3_readvariableop_resource*
_output_shapes

:(*
dtype02
transpose_3/ReadVariableOpu
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm
transpose_3	Transpose"transpose_3/ReadVariableOp:value:0transpose_3/perm:output:0*
T0*
_output_shapes

:(2
transpose_3s
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"(   ÿÿÿÿ2
Reshape_4/shapeu
	Reshape_4Reshapetranspose_3:y:0Reshape_4/shape:output:0*
T0*
_output_shapes

:(2
	Reshape_4x
MatMul_1MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1h
Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :x2
Reshape_5/shape/1h
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_5/shape/2¤
Reshape_5/shapePackunstack_2:output:0Reshape_5/shape/1:output:0Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_5/shape
	Reshape_5ReshapeMatMul_1:product:0Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
	Reshape_5
Reshape_6/ReadVariableOpReadVariableOp!reshape_6_readvariableop_resource*
_output_shapes

:x*
dtype02
Reshape_6/ReadVariableOpw
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   x      2
Reshape_6/shape
	Reshape_6Reshape Reshape_6/ReadVariableOp:value:0Reshape_6/shape:output:0*
T0*"
_output_shapes
:x2
	Reshape_6q
addAddV2Reshape_5:output:0Reshape_6:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
add´
IdentityIdentityadd:z:0^Reshape_6/ReadVariableOp^transpose_1/ReadVariableOp^transpose_3/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿd(:::24
Reshape_6/ReadVariableOpReshape_6/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp28
transpose_3/ReadVariableOptranspose_3/ReadVariableOp:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(

_user_specified_namex
Íî

!__inference__wrapped_model_442272
input_1.
*model_bl_1_shape_1_readvariableop_resource.
*model_bl_1_shape_3_readvariableop_resource0
,model_bl_1_reshape_6_readvariableop_resource.
*model_bl_2_shape_1_readvariableop_resource.
*model_bl_2_shape_3_readvariableop_resource0
,model_bl_2_reshape_6_readvariableop_resource.
*model_tabl_shape_1_readvariableop_resource&
"model_tabl_readvariableop_resource(
$model_tabl_readvariableop_2_resource.
*model_tabl_shape_5_readvariableop_resource,
(model_tabl_add_2_readvariableop_resource
identity¢#model/bl_1/Reshape_6/ReadVariableOp¢%model/bl_1/transpose_1/ReadVariableOp¢%model/bl_1/transpose_3/ReadVariableOp¢#model/bl_2/Reshape_6/ReadVariableOp¢%model/bl_2/transpose_1/ReadVariableOp¢%model/bl_2/transpose_3/ReadVariableOp¢model/tabl/ReadVariableOp¢model/tabl/ReadVariableOp_1¢model/tabl/ReadVariableOp_2¢model/tabl/ReadVariableOp_3¢model/tabl/add_2/ReadVariableOp¢%model/tabl/transpose_1/ReadVariableOp¢%model/tabl/transpose_4/ReadVariableOp
model/bl_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model/bl_1/transpose/perm
model/bl_1/transpose	Transposeinput_1"model/bl_1/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(d2
model/bl_1/transposel
model/bl_1/ShapeShapemodel/bl_1/transpose:y:0*
T0*
_output_shapes
:2
model/bl_1/Shape}
model/bl_1/unstackUnpackmodel/bl_1/Shape:output:0*
T0*
_output_shapes
: : : *	
num2
model/bl_1/unstack±
!model/bl_1/Shape_1/ReadVariableOpReadVariableOp*model_bl_1_shape_1_readvariableop_resource*
_output_shapes

:dx*
dtype02#
!model/bl_1/Shape_1/ReadVariableOpy
model/bl_1/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"d   x   2
model/bl_1/Shape_1
model/bl_1/unstack_1Unpackmodel/bl_1/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2
model/bl_1/unstack_1
model/bl_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   2
model/bl_1/Reshape/shape¢
model/bl_1/ReshapeReshapemodel/bl_1/transpose:y:0!model/bl_1/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
model/bl_1/Reshape¹
%model/bl_1/transpose_1/ReadVariableOpReadVariableOp*model_bl_1_shape_1_readvariableop_resource*
_output_shapes

:dx*
dtype02'
%model/bl_1/transpose_1/ReadVariableOp
model/bl_1/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
model/bl_1/transpose_1/perm»
model/bl_1/transpose_1	Transpose-model/bl_1/transpose_1/ReadVariableOp:value:0$model/bl_1/transpose_1/perm:output:0*
T0*
_output_shapes

:dx2
model/bl_1/transpose_1
model/bl_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"d   ÿÿÿÿ2
model/bl_1/Reshape_1/shape¡
model/bl_1/Reshape_1Reshapemodel/bl_1/transpose_1:y:0#model/bl_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:dx2
model/bl_1/Reshape_1
model/bl_1/MatMulMatMulmodel/bl_1/Reshape:output:0model/bl_1/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
model/bl_1/MatMul~
model/bl_1/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :(2
model/bl_1/Reshape_2/shape/1~
model/bl_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :x2
model/bl_1/Reshape_2/shape/2Ù
model/bl_1/Reshape_2/shapePackmodel/bl_1/unstack:output:0%model/bl_1/Reshape_2/shape/1:output:0%model/bl_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
model/bl_1/Reshape_2/shape¯
model/bl_1/Reshape_2Reshapemodel/bl_1/MatMul:product:0#model/bl_1/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(x2
model/bl_1/Reshape_2
model/bl_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model/bl_1/transpose_2/perm¸
model/bl_1/transpose_2	Transposemodel/bl_1/Reshape_2:output:0$model/bl_1/transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx(2
model/bl_1/transpose_2r
model/bl_1/Shape_2Shapemodel/bl_1/transpose_2:y:0*
T0*
_output_shapes
:2
model/bl_1/Shape_2
model/bl_1/unstack_2Unpackmodel/bl_1/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2
model/bl_1/unstack_2±
!model/bl_1/Shape_3/ReadVariableOpReadVariableOp*model_bl_1_shape_3_readvariableop_resource*
_output_shapes

:(*
dtype02#
!model/bl_1/Shape_3/ReadVariableOpy
model/bl_1/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"(      2
model/bl_1/Shape_3
model/bl_1/unstack_3Unpackmodel/bl_1/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2
model/bl_1/unstack_3
model/bl_1/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   2
model/bl_1/Reshape_3/shapeª
model/bl_1/Reshape_3Reshapemodel/bl_1/transpose_2:y:0#model/bl_1/Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
model/bl_1/Reshape_3¹
%model/bl_1/transpose_3/ReadVariableOpReadVariableOp*model_bl_1_shape_3_readvariableop_resource*
_output_shapes

:(*
dtype02'
%model/bl_1/transpose_3/ReadVariableOp
model/bl_1/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
model/bl_1/transpose_3/perm»
model/bl_1/transpose_3	Transpose-model/bl_1/transpose_3/ReadVariableOp:value:0$model/bl_1/transpose_3/perm:output:0*
T0*
_output_shapes

:(2
model/bl_1/transpose_3
model/bl_1/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"(   ÿÿÿÿ2
model/bl_1/Reshape_4/shape¡
model/bl_1/Reshape_4Reshapemodel/bl_1/transpose_3:y:0#model/bl_1/Reshape_4/shape:output:0*
T0*
_output_shapes

:(2
model/bl_1/Reshape_4¤
model/bl_1/MatMul_1MatMulmodel/bl_1/Reshape_3:output:0model/bl_1/Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/bl_1/MatMul_1~
model/bl_1/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :x2
model/bl_1/Reshape_5/shape/1~
model/bl_1/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
model/bl_1/Reshape_5/shape/2Û
model/bl_1/Reshape_5/shapePackmodel/bl_1/unstack_2:output:0%model/bl_1/Reshape_5/shape/1:output:0%model/bl_1/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
model/bl_1/Reshape_5/shape±
model/bl_1/Reshape_5Reshapemodel/bl_1/MatMul_1:product:0#model/bl_1/Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
model/bl_1/Reshape_5·
#model/bl_1/Reshape_6/ReadVariableOpReadVariableOp,model_bl_1_reshape_6_readvariableop_resource*
_output_shapes

:x*
dtype02%
#model/bl_1/Reshape_6/ReadVariableOp
model/bl_1/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   x      2
model/bl_1/Reshape_6/shape¶
model/bl_1/Reshape_6Reshape+model/bl_1/Reshape_6/ReadVariableOp:value:0#model/bl_1/Reshape_6/shape:output:0*
T0*"
_output_shapes
:x2
model/bl_1/Reshape_6
model/bl_1/addAddV2model/bl_1/Reshape_5:output:0model/bl_1/Reshape_6:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
model/bl_1/add
model/activation_1/ReluRelumodel/bl_1/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
model/activation_1/Relu
model/dropout_1/IdentityIdentity%model/activation_1/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
model/dropout_1/Identity
model/bl_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model/bl_2/transpose/perm¶
model/bl_2/transpose	Transpose!model/dropout_1/Identity:output:0"model/bl_2/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
model/bl_2/transposel
model/bl_2/ShapeShapemodel/bl_2/transpose:y:0*
T0*
_output_shapes
:2
model/bl_2/Shape}
model/bl_2/unstackUnpackmodel/bl_2/Shape:output:0*
T0*
_output_shapes
: : : *	
num2
model/bl_2/unstack±
!model/bl_2/Shape_1/ReadVariableOpReadVariableOp*model_bl_2_shape_1_readvariableop_resource*
_output_shapes

:x<*
dtype02#
!model/bl_2/Shape_1/ReadVariableOpy
model/bl_2/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"x   <   2
model/bl_2/Shape_1
model/bl_2/unstack_1Unpackmodel/bl_2/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2
model/bl_2/unstack_1
model/bl_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿx   2
model/bl_2/Reshape/shape¢
model/bl_2/ReshapeReshapemodel/bl_2/transpose:y:0!model/bl_2/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
model/bl_2/Reshape¹
%model/bl_2/transpose_1/ReadVariableOpReadVariableOp*model_bl_2_shape_1_readvariableop_resource*
_output_shapes

:x<*
dtype02'
%model/bl_2/transpose_1/ReadVariableOp
model/bl_2/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
model/bl_2/transpose_1/perm»
model/bl_2/transpose_1	Transpose-model/bl_2/transpose_1/ReadVariableOp:value:0$model/bl_2/transpose_1/perm:output:0*
T0*
_output_shapes

:x<2
model/bl_2/transpose_1
model/bl_2/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"x   ÿÿÿÿ2
model/bl_2/Reshape_1/shape¡
model/bl_2/Reshape_1Reshapemodel/bl_2/transpose_1:y:0#model/bl_2/Reshape_1/shape:output:0*
T0*
_output_shapes

:x<2
model/bl_2/Reshape_1
model/bl_2/MatMulMatMulmodel/bl_2/Reshape:output:0model/bl_2/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
model/bl_2/MatMul~
model/bl_2/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
model/bl_2/Reshape_2/shape/1~
model/bl_2/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :<2
model/bl_2/Reshape_2/shape/2Ù
model/bl_2/Reshape_2/shapePackmodel/bl_2/unstack:output:0%model/bl_2/Reshape_2/shape/1:output:0%model/bl_2/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
model/bl_2/Reshape_2/shape¯
model/bl_2/Reshape_2Reshapemodel/bl_2/MatMul:product:0#model/bl_2/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
model/bl_2/Reshape_2
model/bl_2/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model/bl_2/transpose_2/perm¸
model/bl_2/transpose_2	Transposemodel/bl_2/Reshape_2:output:0$model/bl_2/transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
model/bl_2/transpose_2r
model/bl_2/Shape_2Shapemodel/bl_2/transpose_2:y:0*
T0*
_output_shapes
:2
model/bl_2/Shape_2
model/bl_2/unstack_2Unpackmodel/bl_2/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2
model/bl_2/unstack_2±
!model/bl_2/Shape_3/ReadVariableOpReadVariableOp*model_bl_2_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02#
!model/bl_2/Shape_3/ReadVariableOpy
model/bl_2/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      2
model/bl_2/Shape_3
model/bl_2/unstack_3Unpackmodel/bl_2/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2
model/bl_2/unstack_3
model/bl_2/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
model/bl_2/Reshape_3/shapeª
model/bl_2/Reshape_3Reshapemodel/bl_2/transpose_2:y:0#model/bl_2/Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/bl_2/Reshape_3¹
%model/bl_2/transpose_3/ReadVariableOpReadVariableOp*model_bl_2_shape_3_readvariableop_resource*
_output_shapes

:*
dtype02'
%model/bl_2/transpose_3/ReadVariableOp
model/bl_2/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
model/bl_2/transpose_3/perm»
model/bl_2/transpose_3	Transpose-model/bl_2/transpose_3/ReadVariableOp:value:0$model/bl_2/transpose_3/perm:output:0*
T0*
_output_shapes

:2
model/bl_2/transpose_3
model/bl_2/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
model/bl_2/Reshape_4/shape¡
model/bl_2/Reshape_4Reshapemodel/bl_2/transpose_3:y:0#model/bl_2/Reshape_4/shape:output:0*
T0*
_output_shapes

:2
model/bl_2/Reshape_4¤
model/bl_2/MatMul_1MatMulmodel/bl_2/Reshape_3:output:0model/bl_2/Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/bl_2/MatMul_1~
model/bl_2/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :<2
model/bl_2/Reshape_5/shape/1~
model/bl_2/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
model/bl_2/Reshape_5/shape/2Û
model/bl_2/Reshape_5/shapePackmodel/bl_2/unstack_2:output:0%model/bl_2/Reshape_5/shape/1:output:0%model/bl_2/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
model/bl_2/Reshape_5/shape±
model/bl_2/Reshape_5Reshapemodel/bl_2/MatMul_1:product:0#model/bl_2/Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
model/bl_2/Reshape_5·
#model/bl_2/Reshape_6/ReadVariableOpReadVariableOp,model_bl_2_reshape_6_readvariableop_resource*
_output_shapes

:<*
dtype02%
#model/bl_2/Reshape_6/ReadVariableOp
model/bl_2/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   <      2
model/bl_2/Reshape_6/shape¶
model/bl_2/Reshape_6Reshape+model/bl_2/Reshape_6/ReadVariableOp:value:0#model/bl_2/Reshape_6/shape:output:0*
T0*"
_output_shapes
:<2
model/bl_2/Reshape_6
model/bl_2/addAddV2model/bl_2/Reshape_5:output:0model/bl_2/Reshape_6:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
model/bl_2/add
model/activation_2/ReluRelumodel/bl_2/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
model/activation_2/Relu
model/dropout_2/IdentityIdentity%model/activation_2/Relu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
model/dropout_2/Identity
model/tabl/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model/tabl/transpose/perm¶
model/tabl/transpose	Transpose!model/dropout_2/Identity:output:0"model/tabl/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
model/tabl/transposel
model/tabl/ShapeShapemodel/tabl/transpose:y:0*
T0*
_output_shapes
:2
model/tabl/Shape}
model/tabl/unstackUnpackmodel/tabl/Shape:output:0*
T0*
_output_shapes
: : : *	
num2
model/tabl/unstack±
!model/tabl/Shape_1/ReadVariableOpReadVariableOp*model_tabl_shape_1_readvariableop_resource*
_output_shapes

:<*
dtype02#
!model/tabl/Shape_1/ReadVariableOpy
model/tabl/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"<      2
model/tabl/Shape_1
model/tabl/unstack_1Unpackmodel/tabl/Shape_1:output:0*
T0*
_output_shapes
: : *	
num2
model/tabl/unstack_1
model/tabl/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ<   2
model/tabl/Reshape/shape¢
model/tabl/ReshapeReshapemodel/tabl/transpose:y:0!model/tabl/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2
model/tabl/Reshape¹
%model/tabl/transpose_1/ReadVariableOpReadVariableOp*model_tabl_shape_1_readvariableop_resource*
_output_shapes

:<*
dtype02'
%model/tabl/transpose_1/ReadVariableOp
model/tabl/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
model/tabl/transpose_1/perm»
model/tabl/transpose_1	Transpose-model/tabl/transpose_1/ReadVariableOp:value:0$model/tabl/transpose_1/perm:output:0*
T0*
_output_shapes

:<2
model/tabl/transpose_1
model/tabl/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"<   ÿÿÿÿ2
model/tabl/Reshape_1/shape¡
model/tabl/Reshape_1Reshapemodel/tabl/transpose_1:y:0#model/tabl/Reshape_1/shape:output:0*
T0*
_output_shapes

:<2
model/tabl/Reshape_1
model/tabl/MatMulMatMulmodel/tabl/Reshape:output:0model/tabl/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/tabl/MatMul~
model/tabl/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
model/tabl/Reshape_2/shape/1~
model/tabl/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
model/tabl/Reshape_2/shape/2Ù
model/tabl/Reshape_2/shapePackmodel/tabl/unstack:output:0%model/tabl/Reshape_2/shape/1:output:0%model/tabl/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
model/tabl/Reshape_2/shape¯
model/tabl/Reshape_2Reshapemodel/tabl/MatMul:product:0#model/tabl/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/tabl/Reshape_2
model/tabl/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model/tabl/transpose_2/perm¸
model/tabl/transpose_2	Transposemodel/tabl/Reshape_2:output:0$model/tabl/transpose_2/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/tabl/transpose_2w
model/tabl/eye/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2
model/tabl/eye/onesp
model/tabl/eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
model/tabl/eye/diag/k
model/tabl/eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
model/tabl/eye/diag/num_rows
model/tabl/eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
model/tabl/eye/diag/num_cols
!model/tabl/eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!model/tabl/eye/diag/padding_value
model/tabl/eye/diagMatrixDiagV3model/tabl/eye/ones:output:0model/tabl/eye/diag/k:output:0%model/tabl/eye/diag/num_rows:output:0%model/tabl/eye/diag/num_cols:output:0*model/tabl/eye/diag/padding_value:output:0*
T0*
_output_shapes

:2
model/tabl/eye/diag
model/tabl/ReadVariableOpReadVariableOp"model_tabl_readvariableop_resource*
_output_shapes

:*
dtype02
model/tabl/ReadVariableOp
model/tabl/mulMul!model/tabl/ReadVariableOp:value:0model/tabl/eye/diag:output:0*
T0*
_output_shapes

:2
model/tabl/mul
model/tabl/ReadVariableOp_1ReadVariableOp"model_tabl_readvariableop_resource*
_output_shapes

:*
dtype02
model/tabl/ReadVariableOp_1
model/tabl/subSub#model/tabl/ReadVariableOp_1:value:0model/tabl/mul:z:0*
T0*
_output_shapes

:2
model/tabl/sub{
model/tabl/eye_1/onesConst*
_output_shapes
:*
dtype0*
valueB*  ?2
model/tabl/eye_1/onest
model/tabl/eye_1/diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
model/tabl/eye_1/diag/k
model/tabl/eye_1/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
model/tabl/eye_1/diag/num_rows
model/tabl/eye_1/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
model/tabl/eye_1/diag/num_cols
#model/tabl/eye_1/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#model/tabl/eye_1/diag/padding_value©
model/tabl/eye_1/diagMatrixDiagV3model/tabl/eye_1/ones:output:0 model/tabl/eye_1/diag/k:output:0'model/tabl/eye_1/diag/num_rows:output:0'model/tabl/eye_1/diag/num_cols:output:0,model/tabl/eye_1/diag/padding_value:output:0*
T0*
_output_shapes

:2
model/tabl/eye_1/diagq
model/tabl/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
model/tabl/truediv/y
model/tabl/truedivRealDivmodel/tabl/eye_1/diag:output:0model/tabl/truediv/y:output:0*
T0*
_output_shapes

:2
model/tabl/truediv~
model/tabl/addAddV2model/tabl/sub:z:0model/tabl/truediv:z:0*
T0*
_output_shapes

:2
model/tabl/addr
model/tabl/Shape_2Shapemodel/tabl/transpose_2:y:0*
T0*
_output_shapes
:2
model/tabl/Shape_2
model/tabl/unstack_2Unpackmodel/tabl/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num2
model/tabl/unstack_2y
model/tabl/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      2
model/tabl/Shape_3
model/tabl/unstack_3Unpackmodel/tabl/Shape_3:output:0*
T0*
_output_shapes
: : *	
num2
model/tabl/unstack_3
model/tabl/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
model/tabl/Reshape_3/shapeª
model/tabl/Reshape_3Reshapemodel/tabl/transpose_2:y:0#model/tabl/Reshape_3/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/tabl/Reshape_3
model/tabl/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
model/tabl/transpose_3/perm 
model/tabl/transpose_3	Transposemodel/tabl/add:z:0$model/tabl/transpose_3/perm:output:0*
T0*
_output_shapes

:2
model/tabl/transpose_3
model/tabl/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
model/tabl/Reshape_4/shape¡
model/tabl/Reshape_4Reshapemodel/tabl/transpose_3:y:0#model/tabl/Reshape_4/shape:output:0*
T0*
_output_shapes

:2
model/tabl/Reshape_4¤
model/tabl/MatMul_1MatMulmodel/tabl/Reshape_3:output:0model/tabl/Reshape_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/tabl/MatMul_1~
model/tabl/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
model/tabl/Reshape_5/shape/1~
model/tabl/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
model/tabl/Reshape_5/shape/2Û
model/tabl/Reshape_5/shapePackmodel/tabl/unstack_2:output:0%model/tabl/Reshape_5/shape/1:output:0%model/tabl/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
model/tabl/Reshape_5/shape±
model/tabl/Reshape_5Reshapemodel/tabl/MatMul_1:product:0#model/tabl/Reshape_5/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/tabl/Reshape_5
 model/tabl/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 model/tabl/Max/reduction_indices¸
model/tabl/MaxMaxmodel/tabl/Reshape_5:output:0)model/tabl/Max/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
model/tabl/Max
model/tabl/sub_1Submodel/tabl/Reshape_5:output:0model/tabl/Max:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/tabl/sub_1s
model/tabl/ExpExpmodel/tabl/sub_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/tabl/Exp
 model/tabl/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 model/tabl/Sum/reduction_indices­
model/tabl/SumSummodel/tabl/Exp:y:0)model/tabl/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(2
model/tabl/Sum
model/tabl/truediv_1RealDivmodel/tabl/Exp:y:0model/tabl/Sum:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/tabl/truediv_1
model/tabl/ReadVariableOp_2ReadVariableOp$model_tabl_readvariableop_2_resource*
_output_shapes
:*
dtype02
model/tabl/ReadVariableOp_2¢
model/tabl/mul_1Mul#model/tabl/ReadVariableOp_2:value:0model/tabl/transpose_2:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/tabl/mul_1
model/tabl/ReadVariableOp_3ReadVariableOp$model_tabl_readvariableop_2_resource*
_output_shapes
:*
dtype02
model/tabl/ReadVariableOp_3m
model/tabl/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
model/tabl/sub_2/x
model/tabl/sub_2Submodel/tabl/sub_2/x:output:0#model/tabl/ReadVariableOp_3:value:0*
T0*
_output_shapes
:2
model/tabl/sub_2
model/tabl/mul_2Mulmodel/tabl/sub_2:z:0model/tabl/transpose_2:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/tabl/mul_2
model/tabl/mul_3Mulmodel/tabl/mul_2:z:0model/tabl/truediv_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/tabl/mul_3
model/tabl/add_1AddV2model/tabl/mul_1:z:0model/tabl/mul_3:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/tabl/add_1l
model/tabl/Shape_4Shapemodel/tabl/add_1:z:0*
T0*
_output_shapes
:2
model/tabl/Shape_4
model/tabl/unstack_4Unpackmodel/tabl/Shape_4:output:0*
T0*
_output_shapes
: : : *	
num2
model/tabl/unstack_4±
!model/tabl/Shape_5/ReadVariableOpReadVariableOp*model_tabl_shape_5_readvariableop_resource*
_output_shapes

:*
dtype02#
!model/tabl/Shape_5/ReadVariableOpy
model/tabl/Shape_5Const*
_output_shapes
:*
dtype0*
valueB"      2
model/tabl/Shape_5
model/tabl/unstack_5Unpackmodel/tabl/Shape_5:output:0*
T0*
_output_shapes
: : *	
num2
model/tabl/unstack_5
model/tabl/Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
model/tabl/Reshape_6/shape¤
model/tabl/Reshape_6Reshapemodel/tabl/add_1:z:0#model/tabl/Reshape_6/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/tabl/Reshape_6¹
%model/tabl/transpose_4/ReadVariableOpReadVariableOp*model_tabl_shape_5_readvariableop_resource*
_output_shapes

:*
dtype02'
%model/tabl/transpose_4/ReadVariableOp
model/tabl/transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
model/tabl/transpose_4/perm»
model/tabl/transpose_4	Transpose-model/tabl/transpose_4/ReadVariableOp:value:0$model/tabl/transpose_4/perm:output:0*
T0*
_output_shapes

:2
model/tabl/transpose_4
model/tabl/Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ2
model/tabl/Reshape_7/shape¡
model/tabl/Reshape_7Reshapemodel/tabl/transpose_4:y:0#model/tabl/Reshape_7/shape:output:0*
T0*
_output_shapes

:2
model/tabl/Reshape_7¤
model/tabl/MatMul_2MatMulmodel/tabl/Reshape_6:output:0model/tabl/Reshape_7:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/tabl/MatMul_2~
model/tabl/Reshape_8/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
model/tabl/Reshape_8/shape/1~
model/tabl/Reshape_8/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
model/tabl/Reshape_8/shape/2Û
model/tabl/Reshape_8/shapePackmodel/tabl/unstack_4:output:0%model/tabl/Reshape_8/shape/1:output:0%model/tabl/Reshape_8/shape/2:output:0*
N*
T0*
_output_shapes
:2
model/tabl/Reshape_8/shape±
model/tabl/Reshape_8Reshapemodel/tabl/MatMul_2:product:0#model/tabl/Reshape_8/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/tabl/Reshape_8¯
model/tabl/add_2/ReadVariableOpReadVariableOp(model_tabl_add_2_readvariableop_resource*"
_output_shapes
:*
dtype02!
model/tabl/add_2/ReadVariableOp«
model/tabl/add_2AddV2model/tabl/Reshape_8:output:0'model/tabl/add_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/tabl/add_2
model/tabl/SqueezeSqueezemodel/tabl/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ2
model/tabl/Squeeze
model/activation_3/SoftmaxSoftmaxmodel/tabl/Squeeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/activation_3/SoftmaxÌ
IdentityIdentity$model/activation_3/Softmax:softmax:0$^model/bl_1/Reshape_6/ReadVariableOp&^model/bl_1/transpose_1/ReadVariableOp&^model/bl_1/transpose_3/ReadVariableOp$^model/bl_2/Reshape_6/ReadVariableOp&^model/bl_2/transpose_1/ReadVariableOp&^model/bl_2/transpose_3/ReadVariableOp^model/tabl/ReadVariableOp^model/tabl/ReadVariableOp_1^model/tabl/ReadVariableOp_2^model/tabl/ReadVariableOp_3 ^model/tabl/add_2/ReadVariableOp&^model/tabl/transpose_1/ReadVariableOp&^model/tabl/transpose_4/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿd(:::::::::::2J
#model/bl_1/Reshape_6/ReadVariableOp#model/bl_1/Reshape_6/ReadVariableOp2N
%model/bl_1/transpose_1/ReadVariableOp%model/bl_1/transpose_1/ReadVariableOp2N
%model/bl_1/transpose_3/ReadVariableOp%model/bl_1/transpose_3/ReadVariableOp2J
#model/bl_2/Reshape_6/ReadVariableOp#model/bl_2/Reshape_6/ReadVariableOp2N
%model/bl_2/transpose_1/ReadVariableOp%model/bl_2/transpose_1/ReadVariableOp2N
%model/bl_2/transpose_3/ReadVariableOp%model/bl_2/transpose_3/ReadVariableOp26
model/tabl/ReadVariableOpmodel/tabl/ReadVariableOp2:
model/tabl/ReadVariableOp_1model/tabl/ReadVariableOp_12:
model/tabl/ReadVariableOp_2model/tabl/ReadVariableOp_22:
model/tabl/ReadVariableOp_3model/tabl/ReadVariableOp_32B
model/tabl/add_2/ReadVariableOpmodel/tabl/add_2/ReadVariableOp2N
%model/tabl/transpose_1/ReadVariableOp%model/tabl/transpose_1/ReadVariableOp2N
%model/tabl/transpose_4/ReadVariableOp%model/tabl/transpose_4/ReadVariableOp:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
!
_user_specified_name	input_1
³
¢
%__inference_tabl_layer_call_fn_443683
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_tabl_layer_call_and_return_conditional_losses_4426202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ<:::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<

_user_specified_namex
Ø
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_443550

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<2

Identity_1"!

identity_1Identity_1:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<
 
_user_specified_nameinputs
û

%__inference_bl_1_layer_call_fn_443420
x
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_bl_1_layer_call_and_return_conditional_losses_4423312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿd(:::22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(

_user_specified_namex"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*³
serving_default
?
input_14
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿd(@
activation_30
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:çÙ
É,
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+&call_and_return_all_conditional_losses
__call__
_default_save_signature"¸)
_tf_keras_network){"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 40]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "BL", "config": {"name": "bl_1", "trainable": true, "dtype": "float32", "output_dim": [120, 5], "kernel_regularizer": null, "kernel_constraint": null}, "name": "bl_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["bl_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "BL", "config": {"name": "bl_2", "trainable": true, "dtype": "float32", "output_dim": [60, 2], "kernel_regularizer": null, "kernel_constraint": null}, "name": "bl_2", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["bl_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "TABL", "config": {"name": "tabl", "trainable": true, "dtype": "float32", "output_dim": [2, 1], "projection_regularizer": null, "projection_constraint": null, "attention_regularizer": null, "attention_constraint": null}, "name": "tabl", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "softmax"}, "name": "activation_3", "inbound_nodes": [[["tabl", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["activation_3", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 100, 40]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 40]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 40]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "BL", "config": {"name": "bl_1", "trainable": true, "dtype": "float32", "output_dim": [120, 5], "kernel_regularizer": null, "kernel_constraint": null}, "name": "bl_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["bl_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "BL", "config": {"name": "bl_2", "trainable": true, "dtype": "float32", "output_dim": [60, 2], "kernel_regularizer": null, "kernel_constraint": null}, "name": "bl_2", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["bl_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "TABL", "config": {"name": "tabl", "trainable": true, "dtype": "float32", "output_dim": [2, 1], "projection_regularizer": null, "projection_constraint": null, "attention_regularizer": null, "attention_constraint": null}, "name": "tabl", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "softmax"}, "name": "activation_3", "inbound_nodes": [[["tabl", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["activation_3", 0, 0]]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0001250000059371814, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
õ"ò
_tf_keras_input_layerÒ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 40]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 40]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
ð

output_dim
W1
W2
bias
trainable_variables
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"µ
_tf_keras_layer{"class_name": "BL", "name": "bl_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bl_1", "trainable": true, "dtype": "float32", "output_dim": [120, 5], "kernel_regularizer": null, "kernel_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 40]}}
×
trainable_variables
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Æ
_tf_keras_layer¬{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}
ç
trainable_variables
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
î
 
output_dim
!W1
"W2
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
+&call_and_return_all_conditional_losses
__call__"³
_tf_keras_layer{"class_name": "BL", "name": "bl_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bl_2", "trainable": true, "dtype": "float32", "output_dim": [60, 2], "kernel_regularizer": null, "kernel_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 120, 5]}}
×
(trainable_variables
)regularization_losses
*	variables
+	keras_api
+&call_and_return_all_conditional_losses
__call__"Æ
_tf_keras_layer¬{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}
ç
,trainable_variables
-regularization_losses
.	variables
/	keras_api
+&call_and_return_all_conditional_losses
__call__"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
Å
0
output_dim
1W1
2W2
3W
	4alpha
5bias
6trainable_variables
7regularization_losses
8	variables
9	keras_api
+&call_and_return_all_conditional_losses
__call__"ø
_tf_keras_layerÞ{"class_name": "TABL", "name": "tabl", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "tabl", "trainable": true, "dtype": "float32", "output_dim": [2, 1], "projection_regularizer": null, "projection_constraint": null, "attention_regularizer": null, "attention_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 2]}}
Ú
:trainable_variables
;regularization_losses
<	variables
=	keras_api
+&call_and_return_all_conditional_losses
__call__"É
_tf_keras_layer¯{"class_name": "Activation", "name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "softmax"}}
¤
>iter

?beta_1

@beta_2
	Adecay
Blearning_ratemumvmw!mx"my#mz1m{2m|3m}4m~5mvvv!v"v#v1v2v3v4v5v"
	optimizer
n
0
1
2
!3
"4
#5
16
27
38
49
510"
trackable_list_wrapper
 "
trackable_list_wrapper
n
0
1
2
!3
"4
#5
16
27
38
49
510"
trackable_list_wrapper
Î
trainable_variables
Clayer_metrics
regularization_losses

Dlayers
Emetrics
Flayer_regularization_losses
Gnon_trainable_variables
	variables
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
 "
trackable_list_wrapper
:dx2bl_1/W1
:(2bl_1/W2
:x2	bl_1/bias
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
°
trainable_variables
Hlayer_metrics
regularization_losses

Ilayers
Jmetrics
Klayer_regularization_losses
Lnon_trainable_variables
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
trainable_variables
Mlayer_metrics
regularization_losses

Nlayers
Ometrics
Player_regularization_losses
Qnon_trainable_variables
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
trainable_variables
Rlayer_metrics
regularization_losses

Slayers
Tmetrics
Ulayer_regularization_losses
Vnon_trainable_variables
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:x<2bl_2/W1
:2bl_2/W2
:<2	bl_2/bias
5
!0
"1
#2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
°
$trainable_variables
Wlayer_metrics
%regularization_losses

Xlayers
Ymetrics
Zlayer_regularization_losses
[non_trainable_variables
&	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
(trainable_variables
\layer_metrics
)regularization_losses

]layers
^metrics
_layer_regularization_losses
`non_trainable_variables
*	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
,trainable_variables
alayer_metrics
-regularization_losses

blayers
cmetrics
dlayer_regularization_losses
enon_trainable_variables
.	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:<2tabl/W1
:2tabl/W2
:2tabl/W
:2
tabl/alpha
:2	tabl/bias
C
10
21
32
43
54"
trackable_list_wrapper
 "
trackable_list_wrapper
C
10
21
32
43
54"
trackable_list_wrapper
°
6trainable_variables
flayer_metrics
7regularization_losses

glayers
hmetrics
ilayer_regularization_losses
jnon_trainable_variables
8	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
:trainable_variables
klayer_metrics
;regularization_losses

llayers
mmetrics
nlayer_regularization_losses
onon_trainable_variables
<	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
'
p0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
»
	qtotal
	rcount
s	variables
t	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
q0
r1"
trackable_list_wrapper
-
s	variables"
_generic_user_object
:dx2Adam/bl_1/W1/m
:(2Adam/bl_1/W2/m
 :x2Adam/bl_1/bias/m
:x<2Adam/bl_2/W1/m
:2Adam/bl_2/W2/m
 :<2Adam/bl_2/bias/m
:<2Adam/tabl/W1/m
:2Adam/tabl/W2/m
:2Adam/tabl/W/m
:2Adam/tabl/alpha/m
$:"2Adam/tabl/bias/m
:dx2Adam/bl_1/W1/v
:(2Adam/bl_1/W2/v
 :x2Adam/bl_1/bias/v
:x<2Adam/bl_2/W1/v
:2Adam/bl_2/W2/v
 :<2Adam/bl_2/bias/v
:<2Adam/tabl/W1/v
:2Adam/tabl/W2/v
:2Adam/tabl/W/v
:2Adam/tabl/alpha/v
$:"2Adam/tabl/bias/v
Ò2Ï
A__inference_model_layer_call_and_return_conditional_losses_443300
A__inference_model_layer_call_and_return_conditional_losses_442662
A__inference_model_layer_call_and_return_conditional_losses_442696
A__inference_model_layer_call_and_return_conditional_losses_443085À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
&__inference_model_layer_call_fn_442758
&__inference_model_layer_call_fn_443354
&__inference_model_layer_call_fn_442819
&__inference_model_layer_call_fn_443327À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ã2à
!__inference__wrapped_model_442272º
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"
input_1ÿÿÿÿÿÿÿÿÿd(
å2â
@__inference_bl_1_layer_call_and_return_conditional_losses_443409
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ê2Ç
%__inference_bl_1_layer_call_fn_443420
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_1_layer_call_and_return_conditional_losses_443425¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_activation_1_layer_call_fn_443430¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
È2Å
E__inference_dropout_1_layer_call_and_return_conditional_losses_443447
E__inference_dropout_1_layer_call_and_return_conditional_losses_443442´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
*__inference_dropout_1_layer_call_fn_443452
*__inference_dropout_1_layer_call_fn_443457´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
å2â
@__inference_bl_2_layer_call_and_return_conditional_losses_443512
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ê2Ç
%__inference_bl_2_layer_call_fn_443523
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_2_layer_call_and_return_conditional_losses_443528¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_activation_2_layer_call_fn_443533¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
È2Å
E__inference_dropout_2_layer_call_and_return_conditional_losses_443545
E__inference_dropout_2_layer_call_and_return_conditional_losses_443550´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
*__inference_dropout_2_layer_call_fn_443555
*__inference_dropout_2_layer_call_fn_443560´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
å2â
@__inference_tabl_layer_call_and_return_conditional_losses_443668
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ê2Ç
%__inference_tabl_layer_call_fn_443683
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_activation_3_layer_call_and_return_conditional_losses_443688¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_activation_3_layer_call_fn_443693¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ËBÈ
$__inference_signature_wrapper_442856input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ¦
!__inference__wrapped_model_442272!"#134254¢1
*¢'
%"
input_1ÿÿÿÿÿÿÿÿÿd(
ª ";ª8
6
activation_3&#
activation_3ÿÿÿÿÿÿÿÿÿ¬
H__inference_activation_1_layer_call_and_return_conditional_losses_443425`3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿx
ª ")¢&

0ÿÿÿÿÿÿÿÿÿx
 
-__inference_activation_1_layer_call_fn_443430S3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿx
ª "ÿÿÿÿÿÿÿÿÿx¬
H__inference_activation_2_layer_call_and_return_conditional_losses_443528`3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ<
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ<
 
-__inference_activation_2_layer_call_fn_443533S3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ<
ª "ÿÿÿÿÿÿÿÿÿ<¤
H__inference_activation_3_layer_call_and_return_conditional_losses_443688X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
-__inference_activation_3_layer_call_fn_443693K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
@__inference_bl_1_layer_call_and_return_conditional_losses_443409`.¢+
$¢!

xÿÿÿÿÿÿÿÿÿd(
ª ")¢&

0ÿÿÿÿÿÿÿÿÿx
 |
%__inference_bl_1_layer_call_fn_443420S.¢+
$¢!

xÿÿÿÿÿÿÿÿÿd(
ª "ÿÿÿÿÿÿÿÿÿx¤
@__inference_bl_2_layer_call_and_return_conditional_losses_443512`!"#.¢+
$¢!

xÿÿÿÿÿÿÿÿÿx
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ<
 |
%__inference_bl_2_layer_call_fn_443523S!"#.¢+
$¢!

xÿÿÿÿÿÿÿÿÿx
ª "ÿÿÿÿÿÿÿÿÿ<­
E__inference_dropout_1_layer_call_and_return_conditional_losses_443442d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿx
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿx
 ­
E__inference_dropout_1_layer_call_and_return_conditional_losses_443447d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿx
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿx
 
*__inference_dropout_1_layer_call_fn_443452W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿx
p
ª "ÿÿÿÿÿÿÿÿÿx
*__inference_dropout_1_layer_call_fn_443457W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿx
p 
ª "ÿÿÿÿÿÿÿÿÿx­
E__inference_dropout_2_layer_call_and_return_conditional_losses_443545d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ<
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ<
 ­
E__inference_dropout_2_layer_call_and_return_conditional_losses_443550d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ<
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ<
 
*__inference_dropout_2_layer_call_fn_443555W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ<
p
ª "ÿÿÿÿÿÿÿÿÿ<
*__inference_dropout_2_layer_call_fn_443560W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ<
p 
ª "ÿÿÿÿÿÿÿÿÿ<·
A__inference_model_layer_call_and_return_conditional_losses_442662r!"#13425<¢9
2¢/
%"
input_1ÿÿÿÿÿÿÿÿÿd(
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ·
A__inference_model_layer_call_and_return_conditional_losses_442696r!"#13425<¢9
2¢/
%"
input_1ÿÿÿÿÿÿÿÿÿd(
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¶
A__inference_model_layer_call_and_return_conditional_losses_443085q!"#13425;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿd(
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¶
A__inference_model_layer_call_and_return_conditional_losses_443300q!"#13425;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿd(
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
&__inference_model_layer_call_fn_442758e!"#13425<¢9
2¢/
%"
input_1ÿÿÿÿÿÿÿÿÿd(
p

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_model_layer_call_fn_442819e!"#13425<¢9
2¢/
%"
input_1ÿÿÿÿÿÿÿÿÿd(
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_model_layer_call_fn_443327d!"#13425;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿd(
p

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_model_layer_call_fn_443354d!"#13425;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿd(
p 

 
ª "ÿÿÿÿÿÿÿÿÿ´
$__inference_signature_wrapper_442856!"#13425?¢<
¢ 
5ª2
0
input_1%"
input_1ÿÿÿÿÿÿÿÿÿd(";ª8
6
activation_3&#
activation_3ÿÿÿÿÿÿÿÿÿ¢
@__inference_tabl_layer_call_and_return_conditional_losses_443668^13425.¢+
$¢!

xÿÿÿÿÿÿÿÿÿ<
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
%__inference_tabl_layer_call_fn_443683Q13425.¢+
$¢!

xÿÿÿÿÿÿÿÿÿ<
ª "ÿÿÿÿÿÿÿÿÿ