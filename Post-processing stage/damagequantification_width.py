# 找到了 在 2 中修改

'''
算法中的x和y完全来自skeleton方法



skeleton = skeletonize(binary )  # 完全骨架化
thin_result = thin(binary )  # 完全细化
partially_thinned = thin(binary  , max_num_iter=25)  # 部分细化

x, y = np.where(skeleton > 0)
centers = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))


--------
修改如下

skeleton = skeletonize(binary )  # 完全骨架化
thin_result = thin(binary )  # 完全细化
partially_thinned = thin(binary  , max_num_iter=25)  # 部分细化

x, y = np.where(skeleton > 0)
centers = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))

x1, y1 = np.where(thin_result > 0)
centers = np.hstack((x1.reshape(-1, 1), y1.reshape(-1, 1)))
...
部分细化的方法也应该是这样的

'''

import numpy as np
from skimage import io
from skimage.morphology import medial_axis
from skimage import measure
from skimage import data
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from skimage.morphology import skeletonize, thin
from skimage.filters import threshold_otsu

def show_2dpoints(pointcluster, s=None, quivers=None, qscale=1):
    # pointcluster should be a list of numpy ndarray
    # This function shows a list of point clouds in different colors
    n = len(pointcluster)
    nmax = n
    if quivers is not None:
        nq = len(quivers)
        nmax = max(n, nq)

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'tomato', 'gold']
    if nmax < 10:
        colors = np.array(colors[0:nmax])
    else:
        colors = np.random.rand(nmax, 3)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # Set up a 2x2 grid of subplots
    fig.suptitle('Crack Analysis Results', fontsize=16)  # Optional: Add a title for all subplots

    if s is None:
        s = np.ones(n) * 2

    for i, ax in enumerate(axes.flat):  # Iterate over axes in the grid
        if i < n:
            ax.scatter(pointcluster[i][:, 0], pointcluster[i][:, 1], s=s[i], c=[colors[i]], alpha=0.6)
            ax.set_title(f'Point Cluster {i + 1}', fontsize=12)
        ax.axis('equal')  # Ensure equal scaling on both axes

    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to prevent overlap with the title
    plt.show()


def SVD(points):
    # 二维，三维均适用
    # 二维直线，三维平面
    pts = points.copy()
    # 奇异值分解
    c = np.mean(pts, axis=0)
    A = pts - c # shift the points
    A = A.T #3*n
    u, s, vh = np.linalg.svd(A, full_matrices=False, compute_uv=True) # A=u*s*vh
    normal = u[:,-1]

    # 法向量归一化
    nlen = np.sqrt(np.dot(normal,normal))
    normal = normal / nlen
    # normal 是主方向的方向向量 与PCA最小特征值对应的特征向量是垂直关系
    # u 每一列是一个方向
    # s 是对应的特征值
    # c >>> 点的中心
    # normal >>> 拟合的方向向量
    return u,s,c,normal


def calcu_dis_from_ctrlpts(ctrlpts):
    if ctrlpts.shape[1]==4:
        return np.sqrt(np.sum((ctrlpts[:,0:2]-ctrlpts[:,2:4])**2,axis=1))
    else:
        return np.sqrt(np.sum((ctrlpts[:,[0,2]]-ctrlpts[:,[3,5]])**2,axis=1))


def estimate_normal_for_pos(pos,points,n):
    """
    计算pos处的法向量.
    
    Input：
    ------
    pos: nx2 ndarray 需要计算法向量的位置.
    points: 骨架线的点集
    n: 用到的近邻点的个数
    
    Output：
    ------
    normals: nx2 ndarray 在pos位置处的法向量.
    """
    
    # estimate normal vectors at a given point
    pts = np.copy(points)
    tree = KDTree(pts, leaf_size=2)
    idx = tree.query(pos, k=n, return_distance=False, dualtree=False, breadth_first=False)
    #pts = np.concatenate((np.concatenate((pts[0].reshape(1,-1),pts),axis=0),pts[-1].reshape(1,-1)),axis=0)
    normals = []
    for i in range(0,pos.shape[0]):
        pts_for_normals = pts[idx[i,:],:]
        _,_,_,normal = SVD(pts_for_normals)
        normals.append(normal)
    normals = np.array(normals)
    return normals


def estimate_normals(points,n):
    """
    计算points表示的曲线上的每一个点法向量.
    等同于 estimate_normal_for_pos(points,points,n)

    Input：
    ------
    points: nx2 ndarray 曲线点集.
    n: 用到的近邻点的个数
    
    Output：
    ------
    normals: nx2 ndarray 在points曲线上的每一处的法向量.
    """
    
    pts = np.copy(points)
    tree = KDTree(pts, leaf_size=2)
    idx = tree.query(pts, k=n, return_distance=False, dualtree=False, breadth_first=False)
    #pts = np.concatenate((np.concatenate((pts[0].reshape(1,-1),pts),axis=0),pts[-1].reshape(1,-1)),axis=0)
    normals = []
    for i in range(0,pts.shape[0]):
        pts_for_normals = pts[idx[i,:],:]
        _,_,_,normal = SVD(pts_for_normals)
        normals.append(normal)
    normals = np.array(normals)
    return normals


def get_crack_ctrlpts(centers,normals,bpoints,hband=5,vband=2,est_width=0):
    # main algorithm to obtain crack width
    cpoints = np.copy(centers)
    cnormals = np.copy(normals)

    xmatrix = np.array([[0,1],[-1,0]])
    cnormalsx = np.dot(xmatrix,cnormals.T).T # the normal of x axis
    N = cpoints.shape[0]

    interp_segm = []
    widths = []
    for i in range(N):
        try:
            ny = cnormals[i]
            nx = cnormalsx[i]
            tform = np.array([nx,ny])
            bpoints_loc = np.dot(tform,bpoints.T).T
            cpoints_loc = np.dot(tform,cpoints.T).T
            ci = cpoints_loc[i]

            bl_ind = (bpoints_loc[:,0]-(ci[0]-hband))*(bpoints_loc[:,0]-ci[0])<0
            br_ind = (bpoints_loc[:,0]-ci[0])*(bpoints_loc[:,0]-(ci[0]+hband))<=0
            bl = bpoints_loc[bl_ind] # left points
            br = bpoints_loc[br_ind] # right points

            if est_width>0:
                # 下面的数值 est_width 是预估计的裂缝宽度
                half_est_width = est_width / 2
                blt = bl[(bl[:,1]-(ci[1]+half_est_width))*(bl[:,1]-ci[1])<0]
                blb = bl[(bl[:,1]-(ci[1]-half_est_width))*(bl[:,1]-ci[1])<0]
                brt = br[(br[:,1]-(ci[1]+half_est_width))*(br[:,1]-ci[1])<0]
                brb = br[(br[:,1]-(ci[1]-half_est_width))*(br[:,1]-ci[1])<0]
            else:
                blt = bl[bl[:,1]>np.mean(bl[:,1])]
                if np.ptp(blt[:,1])>vband:
                    blt = blt[blt[:,1]>np.mean(blt[:,1])]

                blb = bl[bl[:,1]<np.mean(bl[:,1])]
                if np.ptp(blb[:,1])>vband:
                    blb = blb[blb[:,1]<np.mean(blb[:,1])]

                brt = br[br[:,1]>np.mean(br[:,1])]
                if np.ptp(brt[:,1])>vband:
                    brt = brt[brt[:,1]>np.mean(brt[:,1])]

                brb = br[br[:,1]<np.mean(br[:,1])]
                if np.ptp(brb[:,1])>vband:
                    brb = brb[brb[:,1]<np.mean(brb[:,1])]



            t1 = blt[np.argsort(blt[:,0])[-1]]
            t2 = brt[np.argsort(brt[:,0])[0]]

            b1 = blb[np.argsort(blb[:,0])[-1]]
            b2 = brb[np.argsort(brb[:,0])[0]]


            interp1 = (ci[0]-t1[0])*((t2[1]-t1[1])/(t2[0]-t1[0]))+t1[1]
            interp2 = (ci[0]-b1[0])*((b2[1]-b1[1])/(b2[0]-b1[0]))+b1[1]

            if interp1-ci[1]>0 and interp2-ci[1]<0:
                widths.append([i,interp1-ci[1],interp2-ci[1]])

                interps = np.array([[ci[0],interp1],[ci[0],interp2]])

                interps_rec = np.dot(np.linalg.inv(tform),interps.T).T

                #show_2dpoints([bpointsxl_loc1,bpointsxl_loc2,bpointsxr_loc1,bpointsxr_loc2,np.array([ptsl_1,ptsl_2]),np.array([ptsr_1,ptsr_2]),interps,ci.reshape(1,-1)],s=[1,1,1,1,20,20,20,20])
                interps_rec = interps_rec.reshape(1,-1)[0,:]
                interp_segm.append(interps_rec)
        except:
            print("the %d-th was wrong" % i)
            continue
    interp_segm = np.array(interp_segm)
    widths = np.array(widths)
    # check
    # show_2dpoints([np.array([[ci[0],interp1],[ci[0],interp2]]),np.array([t1,t2,b1,b2]),cpoints_loc,bl,br],[10,20,15,2,2])
    return interp_segm, widths

# 新增找到裂缝最宽处的函数
def find_max_width(widths):
    """
    查找裂缝最宽处的测量点.
    
    Input:
    ------
    widths: 宽度数组, 形状为 (n, 3)，其中每一行的结构为 [测量点索引，正上方宽度，正下方宽度].
    
    Output:
    ------
    max_idx: 最宽测量点的索引.
    max_width: 最大的宽度值.
    """
    total_widths = np.abs(widths[:, 1]) + np.abs(widths[:, 2])
    max_idx = np.argmax(total_widths)
    max_width = total_widths[max_idx]
    return max_idx, max_width


def detect_branches(skeleton):
    """
    检测裂缝骨架上的分叉点。假设分叉点是有3个或更多邻居的点。
    
    Input:
    ------
    skeleton: 二值图像, 裂缝骨架
    
    Output:
    ------
    branch_points: 分叉点的坐标列表
    """
    from scipy.ndimage import label
    from skimage.morphology import skeletonize
    
    # 使用8邻域的标记算法检测分叉点
    struct = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])
    
    labeled_array, num_features = label(skeleton, structure=struct)
    branch_points = []
    
    for region in range(1, num_features + 1):
        region_points = np.argwhere(labeled_array == region)
        neighbors = 0
        
        # 统计邻居个数，判断是否为分叉点
        for point in region_points:
            y, x = point
            neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2]) - skeleton[y, x]
            if neighbors >= 3:  # 三个或更多邻居
                branch_points.append((y, x))
    
    return np.array(branch_points)



# 主代码部分
image = io.imread("MaskRDD-140_Last\China_MotorBike_000176_masked_1.png", as_gray=True)
iw, ih = image.shape
# 二值化图像
thresh = threshold_otsu(image)
binary = image > thresh

# 骨架提取
# 使用三种方法
skeleton = skeletonize(binary )  # 完全骨架化
thin_result = thin(binary )  # 完全细化
partially_thinned = thin(binary  , max_num_iter=25)  # 部分细化

x, y = np.where(skeleton > 0)
centers = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))

x1, y1 = np.where(thin_result > 0)
centers1 = np.hstack((x1.reshape(-1, 1), y1.reshape(-1, 1)))

x2, y2 = np.where(partially_thinned > 0)
centers2 = np.hstack((x2.reshape(-1, 1), y2.reshape(-1, 1)))


# 估计法向量
normals = estimate_normals(centers, 9)
normals1 = estimate_normals(centers1, 9)
normals2 = estimate_normals(centers2, 9)

# 检测分叉点
branch_points = detect_branches(skeleton)
branch_points1 = detect_branches(thin_result)
branch_points2 = detect_branches(partially_thinned)

# 获取裂缝轮廓
contours = measure.find_contours(binary , 0.8)
bpoints = np.vstack(contours)

# 骨架上的控制点估计法向量
centers = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
normals = estimate_normals(centers, 9)

centers1 = np.hstack((x1.reshape(-1, 1), y1.reshape(-1, 1)))
normals1 = estimate_normals(centers1, 9)

centers2 = np.hstack((x2.reshape(-1, 1), y2.reshape(-1, 1)))
normals2 = estimate_normals(centers2, 9)



# 计算裂缝控制点和宽度，处理分叉区域
interps, widths = get_crack_ctrlpts(centers, normals, bpoints, hband=5, vband=2, est_width=30)
interps1, widths1 = get_crack_ctrlpts(centers1, normals1, bpoints, hband=5, vband=2, est_width=30)
interps2, widths2 = get_crack_ctrlpts(centers2, normals2, bpoints, hband=5, vband=2, est_width=30)

# 找到裂缝最宽处
max_idx, max_width = find_max_width(widths)
max_point = interps[max_idx]

max_idx1, max_width1 = find_max_width(widths1)
max_point1 = interps1[max_idx1]

max_idx2, max_width2 = find_max_width(widths2)
max_point2 = interps2[max_idx2]

# 绘制分叉裂缝骨架、轮廓、细化结果、最宽处和分叉点
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
ax = axes.ravel()

# 显示原始裂缝图像
ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('Original Crack Image')

# 显示骨架和分叉点
bpixel_and_skeleton = np.zeros((iw, ih, 3), dtype=np.uint8)
bpixel_and_skeleton[bpoints[:, 0].astype(int), bpoints[:, 1].astype(int), 0] = 255  # 轮廓显示为红色
bpixel_and_skeleton[skeleton > 0, 1] = 255  # 骨架显示为绿色
for branch in branch_points:
    bpixel_and_skeleton[branch[0], branch[1], 2] = 255  # 分叉点显示为蓝色
ax[1].imshow(bpixel_and_skeleton)


interps_show = interps[np.random.choice(interps.shape[0], 30, replace=False),:] 
# 在图中绘制裂缝控制点和标记最宽处
for i in range(interps_show.shape[0]):
    ax[1].plot([interps_show[i, 1], interps_show[i, 3]], [interps_show[i, 0], interps_show[i, 2]], 
               c='c', ls='-', lw=2, marker='o', ms=6, mec='c', mfc='c', alpha=0.9)  # 增加点的大小和透明度


# for i in range(interps_show.shape[0]):
#     ax[1].plot([interps_show[i,1],interps_show[i,3]],[interps_show[i,0],interps_show[i,2]],c='c', ls='-', lw=2, marker='o',ms=4,mec='c',mfc='c')

ax[1].plot([max_point[1], max_point[3]], [max_point[0], max_point[2]], 
           c='r', ls='-', lw=3, marker='x', ms=8, mec='r', mfc='r')
ax[1].set_title(f'Skeleton Crack (Max Width: {max_width:.2f} pixels)')

# 显示分叉点并标注
for branch in branch_points:
    ax[1].plot(branch[1], branch[0], 'bo', markersize=8, label="Branch Point")
# ax[1].set_title('Skeleton')




# 显示thin细化结果
bpixel_and_thin = np.zeros((iw, ih, 3), dtype=np.uint8)
bpixel_and_thin[bpoints[:, 0].astype(int), bpoints[:, 1].astype(int), 0] = 255  # 轮廓显示为红色
bpixel_and_thin[thin_result > 0, 1] = 255  # 细化结果显示为绿色
ax[2].imshow(bpixel_and_thin)


interps_show = interps[np.random.choice(interps.shape[0], 30, replace=False),:] 
# 在图中绘制裂缝控制点和标记最宽处
for i in range(interps_show.shape[0]):
    ax[2].plot([interps_show[i, 1], interps_show[i, 3]], [interps_show[i, 0], interps_show[i, 2]], 
               c='c', ls='-', lw=2, marker='o', ms=6, mec='c', mfc='c', alpha=0.9)  # 增加点的大小和透明度


# for i in range(interps_show.shape[0]):
#     ax[1].plot([interps_show[i,1],interps_show[i,3]],[interps_show[i,0],interps_show[i,2]],c='c', ls='-', lw=2, marker='o',ms=4,mec='c',mfc='c')

ax[2].plot([max_point1[1], max_point1[3]], [max_point1[0], max_point1[2]], 
           c='r', ls='-', lw=3, marker='x', ms=8, mec='r', mfc='r')
ax[2].set_title(f'Thinned Crack (Max Width: {max_width1:.2f} pixels)')

# 显示分叉点并标注
for branch in branch_points1:
    ax[2].plot(branch[1], branch[0], 'bo', markersize=8, label="Branch Point")
# ax[2].set_title('Thinned Crack')




# 显示部分细化结果和最宽处
bpixel_and_partial = np.zeros((iw, ih, 3), dtype=np.uint8)
bpixel_and_partial[bpoints[:, 0].astype(int), bpoints[:, 1].astype(int), 0] = 255  # 轮廓显示为红色
bpixel_and_partial[partially_thinned > 0, 1] = 255  # 部分细化结果显示为绿色
ax[3].imshow(bpixel_and_partial)

interps_show = interps[np.random.choice(interps.shape[0], 30, replace=False),:] 
# 在图中绘制裂缝控制点和标记最宽处
for i in range(interps_show.shape[0]):
    ax[3].plot([interps_show[i, 1], interps_show[i, 3]], [interps_show[i, 0], interps_show[i, 2]], 
               c='c', ls='-', lw=2, marker='o', ms=6, mec='c', mfc='c', alpha=0.9)  # 增加点的大小和透明度


# for i in range(interps_show.shape[0]):
#     ax[1].plot([interps_show[i,1],interps_show[i,3]],[interps_show[i,0],interps_show[i,2]],c='c', ls='-', lw=2, marker='o',ms=4,mec='c',mfc='c')

ax[3].plot([max_point2[1], max_point2[3]], [max_point2[0], max_point2[2]], 
           c='r', ls='-', lw=3, marker='x', ms=8, mec='r', mfc='r')
ax[3].set_title(f'Partially Thinned Crack (Max Width: {max_width2:.2f} pixels)')

# 显示分叉点并标注
for branch in branch_points2:
    ax[3].plot(branch[1], branch[0], 'bo', markersize=8, label="Branch Point")

# 布局调整
fig.tight_layout()

# 显示结果
plt.show()

# 输出最宽处的坐标和宽度
print(f"Most Wide Point Coordinates: Start ({max_point[0]:.2f}, {max_point[1]:.2f}), End ({max_point[2]:.2f}, {max_point[3]:.2f})")
print(f"Max Width: {max_width:.2f} pixels")
# print(f"Branch Points: {branch_points}")
