import numpy as np
import time
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from ThreeDLAPGAN.src.pytorch_generator_disscriminator import sample_noise,Generator,Discriminator,UnitNormClipper,\
                                        g_loss,d_loss,iterate_minibatches,iterate_minibatches,iterate_minibatches_with_inf


class GAN(nn.Module):
    def __init__(self,generator,discriminator,num_epochs = 200,batch_size = 50,learning_rate=0.0001,k_d=1, k_g = 1):
        self.num_epochs=num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.k_d = k_d
        self.k_g = k_g
        self.clipper = UnitNormClipper()
        self.generator = generator
        self.discriminator = discriminator


    def train(data,use_cuda=True,TASK = 2 batch_size = self.batch_size,lr = self.learning_rate,k_d = self.k_d, k_g = self.k_g,num_epochs =self.num_epochs ):
        g_optimizer = optim.Adam(self.generator.parameters(),lr=lr)
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)
        try:
            for epoch in range(num_epochs):
                ls_g=[]
                ls_d=[]
                for input_data in iterate_minibatches(data, batch_size):
                    
                    # Optimize D
                    
                    for _ in range(k_d):
                        # Sample noise
                        noise = Variable(torch.Tensor(sample_noise(len(input_data))).cuda())
                        
                        # Do an update
                    
                        inp_data = Variable(torch.Tensor(input_data).cuda())
                        data_gen = self.generator(noise)
                        if(TASK==4):
                            #COde from here https://github.com/EmilienDupont/wgan-gp
                            alpha = torch.rand(batch_size, 1,)
                            alpha = alpha.expand_as(inp_data)
                            if use_cuda:
                                alpha = alpha.cuda()
                            interpolated = alpha * inp_data.data + (1 - alpha) * data_gen.data
                            interpolated = Variable(interpolated, requires_grad=True)
                            if use_cuda:
                                interpolated = interpolated.cuda()
                            prob_interpolated = self.discriminator(interpolated)
                            gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                                grad_outputs=torch.ones(prob_interpolated.size()).cuda() if use_cuda else torch.ones(
                                                prob_interpolated.size()),
                                                create_graph=True, retain_graph=True)[0]
                            gradients = gradients.view(batch_size, -1)
                            gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
                            penalty = 10 * ((gradients_norm - 1) ** 2).mean()
                            loss = d_loss(self.discriminator(data_gen), self.discriminator(inp_data),penalty,TASK)
                        else:
                            loss = d_loss(self.discriminator(data_gen), self.discriminator(inp_data),TASK)
                        ls_d.append(loss.data.cpu().numpy()[0])
                        d_optimizer.zero_grad()
                        loss.backward()
                        d_optimizer.step()
                        if TASK == 3:
                            self.discriminator.apply(clipper)

            
                    # Optimize G
                    for _ in range(k_g):
                        # Sample noise
                        noise = Variable(torch.Tensor(sample_noise(len(input_data))).cuda())
                        
                        # Do an update
                        data_gen = self.generator(noise)
                        loss = g_loss(self.discriminator(data_gen),TASK)
                        ls_g.append(loss.data.cpu().numpy()[0])
                        g_optimizer.zero_grad()
                        loss.backward()
                        g_optimizer.step()
                if(epoch%10==0):
                    print('generator_loss:',np.mean(ls_g),'discriminator_loss',np.mean(ls_d))
        except KeyboardInterrupt:
            pass