#!/bin/bash
INSTRUCT_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
BASE_MODEL="meta-llama/Meta-Llama-3.1-8B"

##################################
# GENERALIZATION ACROSS DATASETS #
##################################

# Train llama3 instruct on BaseFakepedia and eval on {Base, Multihop, Arithmetic}
# Train method = FT, Eval method = ZS, Eval Dataset = {Base, Multihop, Arithmetic}, Eval IF = Point
python main.py BaseFakepedia -M ${INSTRUCT_MODEL} -S 3 -TS 2048 -TSS 1000 -P -BS 8 -GA 2 -CWF instruction -O \
-EV '[{"dataset_name": "BaseFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": 0, "context_weight_format": "instruction"}, {"dataset_name": "MultihopFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": 0, "context_weight_format": "instruction"},{"dataset_name": "Arithmetic", "subsplit": "base", "k_demonstrations": 0, "context_weight_format": "instruction"}]'
# Train method = ICL, Eval method = ICL, Eval Dataset = Base, Eval IF = Point
python main.py BaseFakepedia -NT -M ${INSTRUCT_MODEL} -S 3 -TSS 1000 -BS 8 -GA 2 -CWF instruction \
-EV '[{"dataset_name": "BaseFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": 10, "context_weight_format": "instruction"}, {"dataset_name": "MultihopFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": 10, "context_weight_format": "instruction"},{"dataset_name": "Arithmetic", "subsplit": "base", "k_demonstrations": 10, "context_weight_format": "instruction"}]'
# Train method = N/A, Eval method = ZS, Eval Dataset = Base, Eval IF = Point
python main.py BaseFakepedia -NT -M ${INSTRUCT_MODEL} -S 3 -TSS 1000 -BS 8 -GA 2 -CWF instruction \
-EV '[{"dataset_name": "BaseFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": 0, "context_weight_format": "instruction"}, {"dataset_name": "MultihopFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": 0, "context_weight_format": "instruction"},{"dataset_name": "Arithmetic", "subsplit": "base", "k_demonstrations": 0, "context_weight_format": "instruction"}]'

# Train llama3 base on BaseFakepedia and eval on {Base, Multihop, Arithmetic}
# Train method = FT, Eval method = ZS, Eval Dataset = {Base, Multihop, Arithmetic}, Eval IF = Point
python main.py BaseFakepedia -M ${BASE_MODEL} -S 3 -TS 2048 -TSS 1000 -P -BS 8 -GA 2 -CWF instruction -O \
-EV '[{"dataset_name": "BaseFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": 0, "context_weight_format": "instruction"}, {"dataset_name": "MultihopFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": 0, "context_weight_format": "instruction"},{"dataset_name": "Arithmetic", "subsplit": "base", "k_demonstrations": 0, "context_weight_format": "instruction"}]'
# Train method = ICL, Eval method = ICL, Eval Dataset = Base, Eval IF = Point
python main.py BaseFakepedia -NT -M ${BASE_MODEL} -S 3 -TSS 1000 -BS 8 -GA 2 -CWF instruction \
-EV '[{"dataset_name": "BaseFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": 10, "context_weight_format": "instruction"}, {"dataset_name": "MultihopFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": 10, "context_weight_format": "instruction"},{"dataset_name": "Arithmetic", "subsplit": "base", "k_demonstrations": 10, "context_weight_format": "instruction"}]'
# Train method = N/A, Eval method = ZS, Eval Dataset = Base, Eval IF = Point
python main.py BaseFakepedia -NT -M ${BASE_MODEL} -S 3 -TSS 1000 -BS 8 -GA 2 -CWF instruction \
-EV '[{"dataset_name": "BaseFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": 0, "context_weight_format": "instruction"}, {"dataset_name": "MultihopFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": 0, "context_weight_format": "instruction"},{"dataset_name": "Arithmetic", "subsplit": "base", "k_demonstrations": 0, "context_weight_format": "instruction"}]'

#############################################
# GENERALIZATION ACROSS Instruction Formats #
#############################################
# Train llama3 instruct on BaseFakepedia and eval on {Base, Multihop, Arithmetic}
# Train method = FT, Train IF = instruction, Eval method = ZS, Eval Dataset = {Base}, Eval IF = instruction and float
python main.py BaseFakepedia -M ${INSTRUCT_MODEL} -S 3 -TS 2048 -TSS 1000 -P -BS 8 -GA 2 -CWF instruction -O \
-EV '[{"dataset_name": "BaseFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": 0, "context_weight_format": "instruction"}, {"dataset_name": "BaseFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": 0, "context_weight_format": "float"}]'
# Train method = FT, Train IF = float, Eval method = ZS, Eval Dataset = {Base}, Eval IF = instruction and float
python main.py BaseFakepedia -M ${INSTRUCT_MODEL} -S 3 -TS 2048 -TSS 1000 -P -BS 8 -GA 2 -CWF float -O \
-EV '[{"dataset_name": "BaseFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": 0, "context_weight_format": "instruction"}, {"dataset_name": "BaseFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": 0, "context_weight_format": "float"}]'
# Train llama3 base on BaseFakepedia and eval on {Base, Multihop, Arithmetic}

# Train method = FT, Train IF = instruction, Eval method = ZS, Eval Dataset = {Base}, Eval IF = instruction and float
python main.py BaseFakepedia -M ${BASE_MODEL} -S 3 -TS 2048 -TSS 1000 -P -BS 8 -GA 2 -CWF instruction -O \
-EV '[{"dataset_name": "BaseFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": 0, "context_weight_format": "instruction"}, {"dataset_name": "BaseFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": 0, "context_weight_format": "float"}]'
# Train method = FT, Train IF = float, Eval method = ZS, Eval Dataset = {Base}, Eval IF = instruction and float
python main.py BaseFakepedia -M ${BASE_MODEL} -S 3 -TS 2048 -TSS 1000 -P -BS 8 -GA 2 -CWF float -O \
-EV '[{"dataset_name": "BaseFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": 0, "context_weight_format": "instruction"}, {"dataset_name": "BaseFakepedia", "subsplit": "nodup_relpid", "k_demonstrations": 0, "context_weight_format": "float"}]'
