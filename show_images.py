import torch
from matplotlib import pyplot as plt


def show_images(images, title=""):
    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy() #该命令主要用于数据类型的转换

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))    #即展示的图像的规模是8英寸乘以8英寸的
    rows = int(len(images) ** (1 / 2))  #len命令应当用于表示数据集的大小，如图片一共有64张，那么可以将其按照8行乘以8列的排布方式处理
    cols = round(len(images) / rows)

    # Populating figure with sub-plots  该命令用于在大图上绘制小图
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1) #第一第二个指标用于表示子图是几行几列的，第三个指标用于表示子图中当前图的位置

            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")  # matplotlib 显示灰度图像，设置 Gray 参数
                idx += 1 #不足图像数目的时候，指标依然需要加一的操作,用于显示所在的位置
    fig.suptitle(title, fontsize=30)
    # Showing the figure
    plt.show()