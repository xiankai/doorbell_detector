import torch
from torch._export import capture_pre_autograd_graph
from torch.export import export, ExportedProgram

from BEATs import BEATs, BEATsConfig
model_path='BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'

checkpoint = torch.load(model_path)
cfg = BEATsConfig(checkpoint['cfg'])
BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint['model'])
BEATs_model.eval()

audio_input_16khz = torch.randn(1, 10000)
padding_mask = torch.zeros(1, 10000).bool()
example_args = (audio_input_16khz,padding_mask)
pre_autograd_aten_dialect = capture_pre_autograd_graph(BEATs_model.extract_features, example_args)
print("Pre-Autograd ATen Dialect Graph")
print(pre_autograd_aten_dialect)

aten_dialect: ExportedProgram = export(pre_autograd_aten_dialect, example_args)
print("ATen Dialect Graph")
print(aten_dialect)