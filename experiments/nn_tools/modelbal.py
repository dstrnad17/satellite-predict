#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:52:11 2024

@author: dunnchadnstrnad
"""

import torch
import torch.nn as nn

def calculate_params(n_in, n_out):
    return (n_in * n_out) + n_out

def create_model(num_layers, input_dim, output_dim, total_params):
    layers = []
    remaining_params = total_params
    
    # Initialize input dimension for the first layer
    n_in = input_dim
    
    for i in range(num_layers):
        if i == num_layers - 1:  # Last layer must match output_dim
            n_out = output_dim
        else:
            # Estimate the number of neurons for the current layer
            # The calculation is adjusted to account for remaining layers and params
            n_out = int((remaining_params - output_dim) / (n_in + num_layers - i))
        
        # Create layer and append to layers list
        layer = nn.Linear(n_in, n_out)
        layers.append(layer)
        
        # Calculate and subtract the used parameters
        params_used = calculate_params(n_in, n_out)
        remaining_params -= params_used
        
        # Update input dimension for next layer
        n_in = n_out
    
    return nn.Sequential(*layers)

# Example: Create a network with 3 layers and a total of 1000 parameters
input_dim = 3
output_dim = 1
total_params = 300
num_layers = 2

model = create_model(num_layers=num_layers, input_dim=input_dim, output_dim=output_dim, total_params=total_params)
print(model)

# Verify the total number of parameters
total_calculated_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_calculated_params}")
