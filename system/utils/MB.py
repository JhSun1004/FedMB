import torch
import copy
import numpy as np
import torch.nn as nn
import copy
class MB:
    def __init__(self, global_model: torch.nn.Module, sigma_lr: float = 0.01, num_clients: int = 20, p = 2, eta = 0):
        """
        Initialize the Bias Correction module.

        Args:
            global_model: The global model.
            sigma_lr: The learning rate for the sigma. Default: 0.01.

        Returns:
            None.
        """
        self.sigma_lr = sigma_lr
        self.global_model = copy.deepcopy(global_model)
        self.model_sigma = None
        self.bias_layer = [0, 1]
        print(f"bias_layer: {self.bias_layer}")
        self.local_bias = []
        self.mean_sigma = []
        self.mean_bias = []
        self.eta = eta
        self.max_para = []
        local_bias = []
        self.p = p
        for i, param in enumerate(global_model.parameters()):
            if i in self.bias_layer:
                new_layer = copy.deepcopy(param.detach())
                local_bias.append(copy.deepcopy(new_layer))
        for _ in range(num_clients):
            self.local_bias.append(copy.deepcopy(local_bias))
    
    def generate_sigma(self, global_model: torch.nn.Module, uploaded_models: list, uploaded_weights: list):
        assert(len(uploaded_models) > 0)
        # Calculate sd of uploaded parameters
        if self.model_sigma == None:
            return
        for sigma in self.model_sigma:
            sigma.data.zero_()
        params_g = list(global_model.parameters())
        for w, client_model in zip(uploaded_weights, uploaded_models):
            params_l = list(client_model.parameters())
            for i, sigma in enumerate(self.model_sigma):
                sigma.data += pow((params_l[self.bias_layer[i]].data - params_g[self.bias_layer[i]].data), 2) * w
        for sigma in self.model_sigma:
            sigma.data = torch.sqrt(sigma.data)
        
    def update_bias(self, uploaded_models: list, uploaded_ids: list):
        """
        Update the bias of the local model.

        Args:
            uploaded_models: The models uploaded by the clients.
            uploaded_ids: The ids of the clients.

        Returns:
            None.
        """
        params_g = list(self.global_model.parameters())
        for uploaded_model, id in zip(uploaded_models, uploaded_ids):
            params_l = list(uploaded_model.parameters())
            for i, (bias, sigma) in enumerate(zip(self.local_bias[id], self.model_sigma)):
                sigma_lr = self.sigma_lr
                if self.p == 1: 
                    bias.data = sigma_lr * (params_l[self.bias_layer[i]].data - params_g[self.bias_layer[i]].data) + (1 - self.sigma_lr * sigma.data) * bias.data
                elif self.p == 2:
                    bias_update = (params_l[self.bias_layer[i]].data - params_g[self.bias_layer[i]].data) / sigma.data
                    bias_update[torch.isnan(bias_update)] = 0.0
                    clamp_val = 1
                    torch.clamp(bias_update, -clamp_val, clamp_val)
                    bias.data = sigma_lr * bias_update + (1 - self.sigma_lr * (1 + self.eta)) * bias.data
                else:
                    bias.data = sigma_lr * (params_l[self.bias_layer[i]].data - params_g[self.bias_layer[i]].data) * sigma.data + (1 - self.sigma_lr * sigma.data * sigma.data) * bias.data

    def update(self, global_model: torch.nn.Module, uploaded_models: list, uploaded_ids: list, uploaded_weights: list):
        if self.model_sigma == None:
            self.model_sigma = []
            for i, param in enumerate(self.global_model.parameters()):
                if i in self.bias_layer:
                    new_layer = copy.deepcopy(param.detach())
                    self.model_sigma.append(new_layer) 
        self.global_model = copy.deepcopy(global_model)     
        self.generate_sigma(global_model, uploaded_models, uploaded_weights)                     
        self.update_bias(uploaded_models, uploaded_ids)
        # self.global_model = copy.deepcopy(global_model)

    def distribute_model(self, client):
        """
        Distribute the model to the client.

        Args:
            client: The client to distribute the model to.
            local_model: The local model to distribute.

        Returns:
            None.
        """
        if self.model_sigma == None:
        # if True:
            client.set_head(self.global_model)
        else:  
            local_model = copy.deepcopy(self.global_model)
            params = list(local_model.parameters())
            for i, (bias, sigma) in enumerate(zip(self.local_bias[client.id], self.model_sigma)):
                params[self.bias_layer[i]].data += bias.data * sigma.data
            client.set_head(local_model)
        