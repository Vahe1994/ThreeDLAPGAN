import numpy as np
import time
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from ThreeDLAPGAN.src.pytorch_generator_disscriminator import sample_noise,Generator,Discriminator,UnitNormClipper,\
                                        g_loss,d_loss,iterate_minibatches


class GAN(nn.Module):
    def __init__(self,generator,discriminator):
        super(GAN, self).__init__()
        self.clipper = UnitNormClipper()
        self.generator = generator
        self.discriminator = discriminator


    def train(self,data,inform = None,use_cuda=True,TASK = 2,num_epochs = 200,batch_size = 50,k_d=1, k_g = 1,lr = 0.0001):
        g_optimizer = optim.Adam(self.generator.parameters(),lr=lr)
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)
        try:
            for epoch in range(num_epochs):
                ls_g=[]
                ls_d=[]
                for input_data,info in iterate_minibatches(data, batch_size,inform):
                    
                    # Optimize D
                    
                    for _ in range(k_d):
                        # Sample noise
                        if not (info is None):
                            noise = Variable(torch.cat((torch.Tensor(sample_noise(len(input_data))),torch.Tensor(info)),1).cuda())
                        else:
                            noise = Variable(torch.Tensor(sample_noise(len(input_data))).cuda())
                        
                        # Do an update
                    
                        inp_data = Variable(torch.Tensor(input_data).cuda())
                        data_gen = self.generator(noise)
                        if(TASK==4):
                            #COde from here https://github.com/EmilienDupont/wgan-gp
                            alpha = torch.rand(inp_data.size()[0], 1)
                            alpha = alpha.expand_as(inp_data)
                            if use_cuda:
                                alpha = alpha.cuda()
                            interpolated = alpha * inp_data.data + (1 - alpha) * data_gen.data
                            interpolated = Variable(interpolated, requires_grad=True)
                            if use_cuda:
                                interpolated = interpolated.cuda()
                            prob_interpolated = self.discriminator(interpolated,TASK=TASK)
                            gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                                grad_outputs=torch.ones(prob_interpolated.size()).cuda() if use_cuda else torch.ones(
                                                prob_interpolated.size()),
                                                create_graph=True, retain_graph=True)[0]
                            gradients = gradients.view(inp_data.size()[0], -1)
                            gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
                            penalty = 10 * ((gradients_norm - 1) ** 2).mean()
                            if not (info is None):
                                loss = d_loss(self.discriminator(data_gen + Variable(torch.Tensor(info).cuda(), requires_grad=False),TASK = TASK), self.discriminator(inp_data,TASK = TASK),TASK,penalty)
                            else:
                                loss = d_loss(self.discriminator(data_gen,TASK = TASK), self.discriminator(inp_data,TASK = TASK),TASK,penalty)
                        else:
                            if not (info is None):
                                loss = d_loss(self.discriminator(data_gen + Variable(torch.Tensor(info).cuda(), requires_grad=False),TASK = TASK),self.discriminator(inp_data,TASK = TASK),TASK)
                            else:
                                loss = d_loss(self.discriminator(data_gen,TASK = TASK), self.discriminator(inp_data,TASK = TASK),TASK)
                        ls_d.append(loss.data.cpu().numpy())
                        d_optimizer.zero_grad()
                        loss.backward()
                        d_optimizer.step()
                        if TASK == 3:
                            self.discriminator.apply(self.clipper,TASK = TASK)

            
                    # Optimize G
                    for _ in range(k_g):
                        # Sample noise
                        if not (info is None):
                            noise = Variable(torch.cat((torch.Tensor(sample_noise(len(input_data))),torch.Tensor(info)),1).cuda())
                        else:
                            noise = Variable(torch.Tensor(sample_noise(len(input_data))).cuda())
                        
                        # Do an update
                        data_gen = self.generator(noise)
                        if not (info is None):
                             loss = g_loss(self.discriminator(data_gen + Variable(torch.Tensor(info).cuda(), requires_grad=False),TASK = TASK),TASK)
                        else:
                            loss = g_loss(self.discriminator(data_gen,TASK = TASK),TASK)
                        ls_g.append(loss.data.cpu().numpy())
                        g_optimizer.zero_grad()
                        loss.backward()
                        g_optimizer.step()
                if(epoch%10==0):
                    print('generator_loss:',np.mean(ls_g),'discriminator_loss',np.mean(ls_d))
        except KeyboardInterrupt:
            pass