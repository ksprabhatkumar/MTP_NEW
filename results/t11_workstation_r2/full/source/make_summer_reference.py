"""Run only with the recovered original PyG environment to create a parity fixture."""
from pathlib import Path
import importlib.util
import torch

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('summer_reference', root/'baselines/summer_original/models/mamba_whar.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.manual_seed(20261017)
model = module.GAE_Mamba_WHAR(num_nodes=9, in_features=9, hidden_dim=16, num_classes=33, dropout=.5).eval()
x = torch.randn(3, 12, 9, 9, requires_grad=True)
logits, reconstruction = model(x)
loss = logits.square().mean() + reconstruction.square().mean()
loss.backward()
target = root/'baselines/summer_original/parity_reference.pt'
torch.save({'state':model.state_dict(),'x':x.detach(),'logits':logits.detach(),
            'reconstruction':reconstruction.detach(),'input_gradient':x.grad,
            'parameter_gradients':{n:p.grad for n,p in model.named_parameters()},
            'torch':torch.__version__}, target)
print(target)
