from skimage import io
from skimage.morphology import skeletonize, thin
from skimage import measure
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from skimage.filters import threshold_otsu

# 读取并转换为灰度图像
image = io.imread("MaskRDD-140_Last\China_MotorBike_000176_masked_1.png", as_gray=True)


# 二值化图像
thresh = threshold_otsu(image)
binary = image > thresh

# # 应用 skeletonize 和 thin
# skeleton = skeletonize(image)
# thinned = thin(image)
# thinned_partial = thin(image, max_num_iter=5)

# 应用 skeletonize 和 thin
skeleton = skeletonize(binary)
thinned = thin(binary)
thinned_partial = thin(binary, max_num_iter=5)

# 生成图像网格的邻接图
def create_graph(binary_image):
    G = nx.Graph()
    rows, cols = binary_image.shape
    for r in range(rows):
        for c in range(cols):
            if binary_image[r, c] == 1:  # 只考虑白色像素
                G.add_node((r, c))
                # 添加邻接节点
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and binary_image[nr, nc] == 1:
                        G.add_edge((r, c), (nr, nc))
    return G

# 计算最长路径并返回路径像素
def find_longest_path(binary_image):
    G = create_graph(binary_image)
    longest_length = 0
    longest_path = []
    
    # 寻找连通组件并计算每个组件的最长路径
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        if len(subgraph.nodes) > 1:
            for source in subgraph.nodes:
                lengths = nx.single_source_shortest_path_length(subgraph, source)
                max_node = max(lengths, key=lengths.get)
                path = nx.shortest_path(subgraph, source=source, target=max_node)
                if len(path) > longest_length:
                    longest_length = len(path)
                    longest_path = path
    return longest_path, longest_length

# 计算三种方法下的裂缝最长路径和长度
skeleton_path, skeleton_length = find_longest_path(skeleton)
thinned_path, thinned_length = find_longest_path(thinned)
thinned_partial_path, thinned_partial_length = find_longest_path(thinned_partial)

# 创建彩色图像用于可视化
def visualize_path(image, path):
    color_image = np.stack([image, image, image], axis=-1)  # 将灰度图像转换为RGB
    for r, c in path:
        color_image[r, c] = [1, 0, 0]  # 将路径像素标记为红色
    return color_image

# 可视化三种方法下的最长路径
skeleton_colored = visualize_path(image, skeleton_path)
thinned_colored = visualize_path(image, thinned_path)
thinned_partial_colored = visualize_path(image, thinned_partial_path)

# 输出长度结果
print(f"Skeleton longest path length: {skeleton_length}")
print(f"Thinned longest path length: {thinned_length}")
print(f"Partially thinned longest path length: {thinned_partial_length}")

# 可视化原始图像及处理结果
fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(skeleton_colored)
ax[1].set_title(f'Skeleton (Longest Path Length: {skeleton_length})')
ax[1].axis('off')

ax[2].imshow(thinned_colored)
ax[2].set_title(f'Thinned (Longest Path Length: {thinned_length})')
ax[2].axis('off')

ax[3].imshow(thinned_partial_colored)
ax[3].set_title(f'Partially Thinned (Longest Path Length: {thinned_partial_length})')
ax[3].axis('off')

fig.tight_layout()
plt.show()
