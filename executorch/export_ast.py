from torch._export import capture_pre_autograd_graph
from torch.export import export, ExportedProgram

from executorch import exir
from executorch.exir import EdgeCompileConfig
from executorch.exir.passes import MemoryPlanningPass

from transformers import AutoFeatureExtractor, ASTForAudioClassification
from transformers import AutoProcessor, ASTModel

feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
# feature_extractor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
# model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

from datasets import load_dataset
dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True)
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
example_args = (inputs.input_values,)
pre_autograd_aten_dialect = capture_pre_autograd_graph(model, example_args)
print("Pre-Autograd ATen Dialect Graph")
# print(pre_autograd_aten_dialect)

# Quantization (in-between torch.capture_pre_autograd_graph and torch.export)
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

quantizer = XNNPACKQuantizer()
# if we set is_per_channel to True, we also need to add out_variant of quantize_per_channel/dequantize_per_channel
operator_config = get_symmetric_quantization_config(is_per_channel=False)
quantizer.set_global(operator_config)
m = prepare_pt2e(pre_autograd_aten_dialect, quantizer)
# calibration
m(*example_args)
m = convert_pt2e(m)
# print(f"Quantized model: {m}")

aten_dialect: ExportedProgram = export(pre_autograd_aten_dialect, example_args)
print("ATen Dialect Graph")
# print(aten_dialect)

edge_program: exir.EdgeProgramManager = exir.to_edge(
    aten_dialect,
    # This is needed if quantizing
    compile_config=EdgeCompileConfig(
        _check_ir_validity=False
    )
)
print("Edge Dialect Graph")

# TODO: Delegate to backend (`to_backend`)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
edge_program = edge_program.to_backend(XnnpackPartitioner)

executorch_program: exir.ExecutorchProgramManager = edge_program.to_executorch(
    exir.ExecutorchBackendConfig(
        passes=[],  # User-defined passes
        memory_planning_pass=MemoryPlanningPass(
            "greedy"
        ),  # Default memory planning pass
    )
)

# TODO: Save as executable
with open("quantized.pte", "wb") as file:
    file.write(executorch_program.buffer)
