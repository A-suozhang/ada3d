import torch

checkpoint = torch.load('./latest_model.pth','cpu')

bn_op_names = []
for k in checkpoint['model_state'].keys():
    if 'running_mean' in k and 'backbone_2d' in k:
        bn_op_name = k.replace('.running_mean','')
        bn_op_names.append(bn_op_name)

print(bn_op_names)
if 'optimizer_state' in checkpoint.keys():
    del checkpoint['optimizer_state']

# drop running_mean, beta, number_batches_tracked
# keep running_var and weight
name_to_drop = ['running_mean','bias']  # also drop 'weight' if affine=False
# name_to_modify = ['weight']  # change the bn.weight
for n1 in bn_op_names:
    for n2 in name_to_drop:
        del checkpoint['model_state'][n1+'.'+n2]

torch.save(checkpoint, 'latest_model_for_masked_bn.pth')













