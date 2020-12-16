import torch as th
import torchlib as tl
import matplotlib.pyplot as plt

lr = 1e-2
lr = 1e-3
lr = 1e-1
num_epochs = 600
batch_size = 8
num_batch = 625


params = {th.nn.parameter.Parameter(th.zeros(128), requires_grad=True),
          th.nn.parameter.Parameter(th.zeros(128), requires_grad=True),
          }

# optimizer = th.optim.Adam(params, lr=lr)
optimizer = th.optim.SGD(params, lr=lr, momentum=0.9)
# optimizer = th.optim.Adadelta(params, lr=lr, rho=0.9, eps=1e-06, weight_decay=0)
# optimizer = th.optim.Adagrad(params, lr=lr, lr_decay=0.3, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
scheduler = tl.optim.lr_scheduler.DoubleGaussianKernelLR(optimizer, t_eta_max=50, sigma1=10, sigma2=100, eta_start=1e-6, eta_stop=1e-5, last_epoch=-1)
# scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1, last_epoch=-1)
# scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6, last_epoch=-1)
# scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9988, last_epoch=-1)
# scheduler = th.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=None, epochs=num_epochs, steps_per_epoch=num_batch, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=1000.0, last_epoch=-1)

print(optimizer)
# print(scheduler)

lrs = []
for n in range(num_epochs):

    for b in range(num_batch):
        optimizer.step()
        # print(optimizer.param_groups)
        lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()
    # print(optimizer)

plt.figure()
plt.plot(lrs)
plt.show()