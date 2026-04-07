import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 定义双模态编码器（对应论文中的 f_ψ 和 f_ξ）
# ==========================================
class SimpleEncoder(nn.Module):
    """将输入映射到共享的意图空间（2维，方便可视化）"""
    def __init__(self, input_dim, embed_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, embed_dim)
        )
    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. 对比损失函数（InfoNCE / 对称交叉熵）
# ==========================================
def contrastive_loss(z1, z2, temperature=0.1):
    """
    z1, z2: [batch_size, embed_dim] 两个模态的嵌入
    temperature: 控制正负样本分离的锐度
    """
    # L2归一化（余弦相似度的前提）
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)

    # 计算相似度矩阵 [batch, batch]
    # 对角线 = 正样本对（同一轨迹与其描述）
    # 非对角线 = 负样本对（轨迹与其他随机描述）
    sim_matrix = torch.matmul(z1, z2.T) / temperature

    # 标签：对角线索引即为正样本
    labels = torch.arange(z1.size(0), device=z1.device)
    
    # 交叉熵损失：鼓励对角线最大，其余最小
    loss = nn.CrossEntropyLoss()(sim_matrix, labels)
    return loss

# ==========================================
# 3. 构造演示数据
# ==========================================
torch.manual_seed(42)
batch_size = 8
traj_dim = 10  # 机器人轨迹特征维度
text_dim = 10  # 语言标注特征维度

# 正样本对：模拟“同一意图的不同表达”
# 轨迹和文本共享一个潜在意图向量，加上各自模态的噪声
shared_intent = torch.randn(batch_size, 5)

# 构造意图到各模态的投影矩阵 [意图维度, 模态维度]
proj_to_traj = torch.randn(5, traj_dim)   # [5, 10]
proj_to_text = torch.randn(5, text_dim)   # [5, 10]

# 生成正样本对：模态特有噪声 + 共享意图的线性投影
trajectories = torch.randn(batch_size, traj_dim) + 0.3 * (shared_intent @ proj_to_traj)
texts = torch.randn(batch_size, text_dim) + 0.3 * (shared_intent @ proj_to_text)

# ==========================================
# 4. 初始化模型与优化器
# ==========================================
motion_encoder = SimpleEncoder(traj_dim)   # 论文中的 f_ψ
lang_encoder   = SimpleEncoder(text_dim)   # 论文中的 f_ξ
optimizer = optim.Adam(list(motion_encoder.parameters()) + list(lang_encoder.parameters()), lr=0.02)

# ==========================================
# 5. 训练循环（极简版）
# ==========================================
print("🔄 开始对比学习训练...")
for epoch in range(60):
    optimizer.zero_grad()
    
    z_traj = motion_encoder(trajectories)
    z_text = lang_encoder(texts)
    
    loss = contrastive_loss(z_traj, z_text, temperature=0.1)
    loss.backward()
    optimizer.step()
    
    if epoch % 15 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

# ==========================================
# 6. 可视化：意图空间对齐效果
# ==========================================
with torch.no_grad():
    z_traj_vis = nn.functional.normalize(motion_encoder(trajectories), dim=1).numpy()
    z_text_vis = nn.functional.normalize(lang_encoder(texts), dim=1).numpy()

plt.figure(figsize=(7, 6))
plt.scatter(z_traj_vis[:, 0], z_traj_vis[:, 1], c='steelblue', marker='o', s=80, label='Robot Motions')
plt.scatter(z_text_vis[:, 0], z_text_vis[:, 1], c='coral', marker='x', s=80, label='Language Annotations')

# 绘制正样本对连线
for i in range(batch_size):
    plt.plot([z_traj_vis[i, 0], z_text_vis[i, 0]], 
             [z_traj_vis[i, 1], z_text_vis[i, 1]], 'k--', alpha=0.4)

plt.title('Shared Intention Space After Contrastive Learning')
plt.xlabel('Intention Dim 1')
plt.ylabel('Intention Dim 2')
plt.legend()
plt.grid(alpha=0.3)
plt.axis('equal')
plt.show()