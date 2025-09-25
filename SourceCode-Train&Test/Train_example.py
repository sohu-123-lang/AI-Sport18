import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 32
NUM_EPOCHS = 10000
LEARNING_RATE = 0.001
KEYPOINT_FEATURES = 12
HIDDEN_DIM = 64 

class MultiModalModel(nn.Module):
    def __init__(self):
        super(MultiModalModel, self).__init__()
        
        self.x1_projection = nn.Linear(3, HIDDEN_DIM)
        
        # X1
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=HIDDEN_DIM, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)

        # X2
        self.fc_team = nn.Sequential(
            nn.Linear(4, HIDDEN_DIM), 
            nn.ReLU(),
        )
        

        self.x3_projection = nn.Linear(KEYPOINT_FEATURES, HIDDEN_DIM) 
        self.transformer_gesture_layer = nn.TransformerEncoderLayer(d_model=HIDDEN_DIM, nhead=4)
        self.transformer_gesture = nn.TransformerEncoder(self.transformer_gesture_layer, num_layers=2)
        
        self.fc_additional = nn.Sequential(
            nn.Linear(1, HIDDEN_DIM),
            nn.ReLU(),
        )
        
        self.fc_fusion = nn.Linear(HIDDEN_DIM * 4, 128)
        self.fc_output = nn.Linear(128, 2)  
    
    def forward(self, x1, x2, x3, x4):

        x1 = self.x1_projection(x1)  
        x1_transformed = self.transformer_encoder(x1.permute(1, 0, 2))  
        x1_out = x1_transformed[-1]  

        # X2
        x2_out = self.fc_team(x2)   

        # X3 
        x3 = self.x3_projection(x3)  
        x3_transformed = self.transformer_gesture(x3.permute(1, 0, 2)) 
        x3_out = x3_transformed[-1]  # 

        # X4
        x4_out = self.fc_additional(x4.view(-1, 1))  

        
        combined = torch.cat((x1_out, x2_out, x3_out, x4_out), dim=1)  #
        fused = F.relu(self.fc_fusion(combined))  
        output = self.fc_output(fused)  # 
        return output

def evaluate_model(model, test_loader):
    model.eval()  
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for X1, X2, X3, X4, y in test_loader:
            outputs = model(X1, X2, X3, X4)
            all_outputs.append(outputs)
            all_targets.append(y)
    
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    mae = F.l1_loss(all_outputs, all_targets)  # MAE computing
    print(f' MAE: {mae.item():.4f}')
    return all_outputs, all_targets, mae

def train_model(model, criterion, optimizer, train_loader, val_loader, scheduler=None):
    model.train()
    for epoch in range(NUM_EPOCHS):
        for X1, X2, X3, X4, y in train_loader:
            optimizer.zero_grad() 
            outputs = model(X1, X2, X3, X4) 
            loss = criterion(outputs, y)  
            loss.backward()  
            optimizer.step() 
        if scheduler:
            scheduler.step()
        
        _, _, val_loss = evaluate_model(model, val_loader)
        if (epoch + 1) % 100 == 0:  
            print(f'第 {epoch + 1} 轮, 损失: {loss.item():.4f}, 验证损失: {val_loss.item():.4f}')

# initizationizing
model = MultiModalModel()
criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# lr setting
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)  


num_samples = 1656
X1 = torch.load('shuffled_X1.pt') 
X2 = torch.load('shuffled_X2.pt')
X3 = torch.load('shuffled_X3.pt')
X4 = torch.load('shuffled_X4.pt')

y = torch.load('shuffled_y.pt')

# create TensorDataset and DataLoader
dataset = TensorDataset(X1, X2, X3, X4, y)
total_size = len(dataset)
train_size = int(0.9 * total_size)  # 90% for train
val_size = int(0.05 * total_size)    # 5% for evaluation
test_size = total_size - train_size - val_size  # left for test

train_dataset, temp_dataset = random_split(dataset, [train_size, total_size - train_size])
val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_model(model, criterion, optimizer, train_loader, val_loader, scheduler)

evaluate_model(model, test_loader)

torch.save(model.state_dict(), 'multi_modal_model_transformer.pth')
print("模型已保存到 'multi_modal_model_transformer.pth'")
