import torch
import torch.nn as nn
from torch.autograd import grad
from torchvision.models import resnet34


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def gradient_penalty(critic, h_s, h_t):
    ''' Gradeitnt penalty for Wasserstein GAN'''
    alpha = torch.rand(h_s.size(0), 1).cuda()
    # alpha = torch.rand(h_s.size(0), 1)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.cat([interpolates, h_s, h_t]).requires_grad_()
    # interpolates.requires_grad_()
    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()

    return gradient_penalty


class GradReverse(torch.autograd.Function):
    '''
    Gradient Reversal Layer
    '''
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg()*ctx.constant
        return grad_output, None

    # pylint raise E0213 warning here
    def grad_reverse(x, constant):
        '''
        Extension of grad reverse layer
        '''
        return GradReverse.apply(x, constant)

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()

        self.resnet = resnet34(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.resnet.fc.in_features
        # Remove the last layer of the resnet
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # Reshape to [batch_size, 512]
        return x

class Classifier(nn.Module):
    ''' Task Classifier '''

    def __init__(self):
        super(Classifier, self).__init__()

        # self.classify = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 5),
        # )

        self.centerLoss = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2, 5)
        )

    def forward(self, x):
    
        x = self.centerLoss(x)
        y = self.classifier(x)

        return x, y




import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    ''' Self-Attention Module '''

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()

        self.query = nn.Linear(in_dim, in_dim, bias=False)
        self.key = nn.Linear(in_dim, in_dim, bias=False)
        self.value = nn.Linear(in_dim, in_dim, bias=False)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        scores = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = torch.softmax(scores / torch.sqrt(torch.tensor(x.size(-1))), dim=-1)

        attended_values = torch.matmul(attention_weights, value)
        output = x + attended_values

        return output


class Discriminator_WGAN(nn.Module):
    ''' Domain Discriminator '''

    def __init__(self):
        super(Discriminator_WGAN, self).__init__()

        self.classify = nn.Sequential(
            nn.Linear(512, 16),  # Decreased hidden size
            nn.LeakyReLU(0.2),  # LeakyReLU activation
            nn.Dropout(0.2),  # Dropout regularization
            # nn.Linear(64, 16),  # Decreased hidden size
            # nn.LeakyReLU(0.2),  # LeakyReLU activation
            # nn.Dropout(0.2),  # Dropout regularization
            SelfAttention(16),  # Apply self-attention here
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.classify(x)
    

# class Discriminator_WGAN(nn.Module):
#     ''' Domain Discriminator '''

#     def __init__(self):
#         super(Discriminator_WGAN, self).__init__()

#         self.classify = nn.Sequential(
#             nn.Linear(512, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(128, 32),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             SmallAttention(32),  # Apply attention here
#             nn.Linear(32, 1)
#         )

#     def forward(self, x):
#         return self.classify(x)




# class Discriminator_WGAN(nn.Module):
#     ''' Domain Discriminator '''

#     def __init__(self):
#         super(Discriminator_WGAN, self).__init__()

#         self.classify = nn.Sequential(
#             nn.Linear(512, 64),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(64, 1),
#             # nn.ReLU(inplace=True),
#             # nn.Dropout(0.5),
#             # nn.Linear(64, 1)
#         )

#     def forward(self, x):
#         return self.classify(x)










# import torch
# import torch.nn as nn

# class Attention(nn.Module):
#     def __init__(self, in_features):
#         super(Attention, self).__init__()
        
#         self.query = nn.Linear(in_features, in_features // 4)
#         self.key = nn.Linear(in_features, in_features // 4)
#         self.value = nn.Linear(in_features, in_features // 4)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         query = self.query(x)
#         key = self.key(x)
#         value = self.value(x)
        
#         attn_weights = self.softmax(torch.bmm(query.unsqueeze(2), key.unsqueeze(1)))
#         attended_features = torch.bmm(attn_weights, value.unsqueeze(2)).squeeze(2)
        
#         return attended_features

# class ResidualBlock(nn.Module):
#     def __init__(self, in_features):
#         super(ResidualBlock, self).__init__()
        
#         self.block = nn.Sequential(
#             nn.Linear(in_features, in_features // 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features // 2, in_features),
#         )
        
#     def forward(self, x):
#         return x + self.block(x)

# class Discriminator_WGAN(nn.Module):
#     ''' Domain Discriminator '''

#     def __init__(self, in_features=512):
#         super(Discriminator_WGAN, self).__init__()

#         self.attention = Attention(in_features)
#         self.residual_block = ResidualBlock(in_features // 4)
#         self.classify = nn.Linear(in_features // 4, 1)  # Adjusted the output size to in_features // 2
        
#     def forward(self, x):
#         attended_features = self.attention(x)
#         residual_features = self.residual_block(attended_features)
#         output = self.classify(residual_features)

#         return output