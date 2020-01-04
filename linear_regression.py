# 梯度是函数在各个自变量处的偏导数构成的向量
# 梯度的方向永远指向函数在当前位置 函数值增大最快的方向
# 梯度的反方向即指向函数在当前位置 函数值减少最快的方向
# 梯度下降法 运用的就是梯度的反向方
# x1 = x0 - a*梯度  该公式：在a的步幅条件下，当前位置梯度下降最快的下个位置点的求法
# 不断迭代下个位置，以求出函数的极小值点
#
# 在给出loss函数的约束条件下，运用梯度下降法，将相关参数代入以求出loss函数的最小值


import numpy as np 

# 基于 y = 1.477x + 0.089 线性模型采样，并添加正态分布误差自变量
# 采样 100 个样本
def generate_simple():
    data = []

    for i in range(100):
        x = np.random.uniform(-10.,10.)
        eps = np.random.normal(0.,0.01)
        y = 1.477 * x + 0.089 + eps
        data.append([x,y])

    data = np.array(data)
    return data


# 计算在梯度下降时，不同的w,b 对应的loss函数值是多少，以便后续确定最优w,b
def mse(b, w, points):
    totalError = 0
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        totalError += (y - (w * x + b)) ** 2
    return totalError/float(len(points))


# 梯度下降 函数实现
def step_gradient(b_current, w_current, points, lr):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))

    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient += (2/N) * ((w_current * x + b_current) - y)      # 文献公式
        w_gradient += (2/N) * x * ((w_current * x + b_current) - y)  # 文献公式
    
    new_b = b_current - lr * b_gradient         # x1 = x0 - a*梯度
    new_w = w_current - lr * w_gradient
    return [new_b, new_w]


# 梯度下降训练
def gradient_descent(points, starting_b, starting_w, lr, num_iterations):
    b = starting_b
    w = starting_w

    for step in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), lr)
        loss = mse(b, w, points)
        if step % 50 == 0:
            print(f"iterations:{step}, loss:{loss}, w:{w}, b:{b}")
    
    return [b,w]


def main():
    lr = 0.01
    init_b = 0
    init_w = 0
    num_iterations = 10000

    points = generate_simple()
    [b,w] = gradient_descent(points, init_b, init_w, lr, num_iterations)
    loss = mse(b,w,points)
    print(f'Final loss:{loss}, w:{w}, b:{b}')

if __name__ == '__main__':
    main()