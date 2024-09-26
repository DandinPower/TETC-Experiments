import torch
import torch.nn as nn
import torch.optim as optim

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()

# Dummy data
x = torch.randn(100, 10)
y = torch.randn(100, 1)

optimizer = optim.Adam(model.parameters())

import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                     model_parameters=model.parameters(),
                                                     config="config.json")

criterion = nn.MSELoss()

for epoch in range(10):
    for i in range(0, len(x), 32):
        batch_x = x[i:i+32]
        batch_y = y[i:i+32]
        
        outputs = model_engine(batch_x)
        loss = criterion(outputs, batch_y)
        
        model_engine.backward(loss)
        model_engine.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
