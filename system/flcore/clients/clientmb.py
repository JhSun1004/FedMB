# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
from collections import defaultdict

class clientMB(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.protos = None
        self.global_protos = None
        self.loss_mse = nn.MSELoss()

        self.lamda = args.lamda


    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

        for i, (x, y) in enumerate(trainloader):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            rep = self.model.base(x)
            output = self.model.head(rep)
            loss = self.loss(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.optimizer_head.step(self.global_head, self.device)

        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = False
            
        # protos = defaultdict(list)
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]):
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.collect_protos()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_protos(self, global_protos):
        self.global_protos = copy.deepcopy(global_protos)

    def set_head(self, head):
        for new_param, old_param in zip(head.parameters(), self.model.head.parameters()):
            old_param.data = new_param.data.clone()

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        self.protos = agg_func(protos)

# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos