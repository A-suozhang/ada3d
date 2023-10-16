import torch
import sys
import os

if len(sys.argv) < 2:
    x = torch.load(os.path.join('./exported_intermediates.pth'),'cpu')
else:
    x = torch.load(sys.argv[1],'cpu')

print(x['model_3d'])
for k,v in x.items():
    if '3d_' in k and ('pre' in k or 'post' in k):
        print(k, v[0].features.shape)

print(x['model_2d'])
for k,v in x.items():
    if 'block' in k:
        sparse_rate = (v[0]==0).int().sum() / v[0].nelement()
        print(k, sparse_rate)
