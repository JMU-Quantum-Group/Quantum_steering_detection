import torch


def get_common_perpendicular(line1, line2):
    # 确定两条直线的参数方程
    x1, y1, z1 = line1[0]
    m1, n1, p1 = line1[1]
    x2, y2, z2 = line2[0]
    m2, n2, p2 = line2[1]

    # 构造系数矩阵和常数向量
    A = torch.tensor([[m1, -m2], [n1, -n2], [p1, -p2]], dtype=torch.float)
    b = torch.tensor([x2 - x1, y2 - y1, z2 - z1], dtype=torch.float)

    # 解线性方程组
    t0, s0 = torch.lstsq(b.unsqueeze(1), A)[0][:2]

    # 计算公垂线两端点坐标
    A = (x1 + m1 * t0.item(), y1 + n1 * t0.item(), z1 + p1 * t0.item())
    B = (x2 + m2 * s0.item(), y2 + n2 * s0.item(), z2 + p2 * s0.item())

    return A, B


# 示例
line1 = ((0, 0, 0), (1, 0, 0))
line2 = ((1, 1, 0), (0, 1, 0))
A, B = get_common_perpendicular(line1, line2)
print(f'Point A: {A}')
print(f'Point B: {B}')
