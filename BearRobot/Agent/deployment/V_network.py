import torch
import torch.nn as nn
import torchvision.models as models

class V_model(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(V_model,self).__init__()
        
        self.resnet1 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.resnet2 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        self.resnet1 = nn.Sequential(*list(self.resnet1.children())[:-2])
        self.resnet2 = nn.Sequential(*list(self.resnet2.children())[:-2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        resnet_output_dim = 512
        self.fc1 = nn.Linear(resnet_output_dim * 2 + state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,50)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
    
    def forward(self, image1, image2, state):
        img_emb1 = self.resnet1(image1)
        img_emb2 = self.resnet2(image2)
        
        img_emb1 = self.avgpool(img_emb1)
        img_emb2 = self.avgpool(img_emb2)
        
        img_emb1 = img_emb1.view(img_emb1.size(0), -1)
        img_emb2 = img_emb2.view(img_emb2.size(0), -1)
        
        combined = torch.cat((img_emb1, img_emb2, state), dim=1)
        
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.softmax(x)
        
        return x

if __name__ == "__main__":
    image1 = torch.randn(2, 3, 224, 224) 
    image2 = torch.randn(2, 3, 224, 224)
    state = torch.randn(2, 9)
    
    model = V_model(state_dim=9)
    output = model(image1, image2, state)
    
    print(output)
    print(torch.softmax(output,dim=-1))  
