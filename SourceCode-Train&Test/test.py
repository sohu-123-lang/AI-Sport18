import torch
import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import matplotlib.patches as patches

class MultiModalModel(torch.nn.Module):
    def __init__(self):
        super(MultiModalModel, self).__init__()
        self.rnn = torch.nn.LSTM(input_size=3, hidden_size=64, num_layers=2, batch_first=True)
        self.fc_team = torch.nn.Linear(2 * 2, 64)
        self.rnn_gesture = torch.nn.LSTM(input_size=12, hidden_size=64, num_layers=2, batch_first=True)
        self.fc_fusion = torch.nn.Linear(64 * 3, 128)
        self.fc_output = torch.nn.Linear(128, 2)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x1, x2, x3):
        _, (hn, _) = self.rnn(x1)
        x1_out = hn[-1]
        x1_out = self.dropout(x1_out)
        x2_out = F.relu(self.fc_team(x2.view(x2.size(0), -1)))
        _, (hn_gesture, _) = self.rnn_gesture(x3)
        x3_out = hn_gesture[-1]
        x3_out = self.dropout(x3_out)
        combined = torch.cat((x1_out, x2_out, x3_out), dim=1)
        fused = F.relu(self.fc_fusion(combined))
        output = self.fc_output(fused)
        return output

model = MultiModalModel()
#$model.load_state_dict(torch.load('multi_modal_model_best2-dropout.pth'))
model.load_state_dict(torch.load('multi_modal_model_best2.pth'))

def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

enable_dropout(model)

X1 = torch.load('combined_X1.pt')
X2 = torch.load('combined_X2.pt')
X3 = torch.load('combined_X3.pt')
y_true = torch.load('combined_y.pt')  

all_outputs = []

num_samples = X1.shape[0]  
with torch.no_grad():
    for i in range(num_samples):
        x1_test = X1[i].unsqueeze(0)  
        x2_test = X2[i].unsqueeze(0)
        x3_test = X3[i].unsqueeze(0)
        output = model(x1_test, x2_test, x3_test)
        all_outputs.append(output.numpy())


outputs = np.concatenate(all_outputs, axis=0)

errors = np.linalg.norm(outputs - y_true.numpy(), axis=1)


num_top_errors = int(len(errors) * 0.1)  # 取前 10%
top_error_indices = np.argsort(errors)[-num_top_errors:]

print("Error indices of the top 10%:")
for idx in top_error_indices:
    print(f"Sample {idx}: Predicted {outputs[idx]}, Ground Truth {y_true[idx]}, Error {errors[idx]}")

# average error
mean_error = np.mean(errors)
print(f"\nAverage Error: {mean_error:.4f}")

# visual
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, color='blue', alpha=0.7)
plt.axvline(np.percentile(errors, 90), color='red', linestyle='--', label='Top 10% Threshold')
plt.title("Error Distribution")
plt.xlabel("Euclidean Distance Error")
plt.ylabel("Frequency")
plt.legend()
plt.show()
