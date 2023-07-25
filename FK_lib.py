import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        # 跳过前面的行
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    # 注意 offset是相对于离自己最近的父节点的偏移
    
    joint_name = []
    joint_parent = []
    joint_offset = None

    offset_list = []


    stack = []  # 由于关节是递归定义的 所以我们用一个栈处理数据 里面的数据为一个列表 为[name,index,offset]
    index = -1   # 关于索引的整数 我们设定根节点的父节点索引为-1  根节点作为父节点时的索引为0
    currentJoint = [0,0,0]  # 用于放于stack中的全局列表 [名字,自身索引,父关节索引,偏移]

    """" 处理文件 """
    with open(bvh_file_path) as bvh_file:
        for line in bvh_file:
            line = line.lstrip()
            line = line.rstrip()
            if line == 'MOTION':
                break
            if line == 'HIERARCHY' or line == "{" or line.startswith('CHANNELS'):
                continue

            # 节点定义行
            if line.startswith('ROOT'):
                joint_name.append("RootJoint")

                currentJoint[0] = "RootJoint"                
                # 父关节索引
                joint_parent.append(index)
                currentJoint[2] = index
                index += 1
                # 自身索引
                currentJoint[1] = index
                continue

            if line.startswith('JOINT'):
                name = line.lstrip("JOINT ")
                
                joint_name.append(name)
                currentJoint[0] = name
                # 父关节索引 即是栈顶的关节的自身索引
                joint_parent.append(stack[-1][1])
                currentJoint[2] = stack[-1][1]
                index += 1
                # 自身索引
                currentJoint[1] = index
                continue

            if line.startswith('End Site'):
                name = str(joint_name[-1] + "_end")
                joint_name.append(name)
                currentJoint[0] = name
                # 父关节索引 即是栈顶的关节的自身索引
                joint_parent.append(stack[-1][1])
                currentJoint[2] = stack[-1][1]
                index += 1
                # 自身索引
                currentJoint[1] = index
                continue

            # 偏移行 完成独读取操作后入栈
            if line.startswith('OFFSET'):
                if currentJoint[0] == 0:
                    raise Exception("文件结构错误!")
                offsets = line.lstrip("OFFSET") # 去除OFFSET字样

                # 得到偏移数组xyz放入偏移列表 这与出现的名字顺序是对应的
                xyz = [float(x) for x in offsets.split()]
                offset_list.append(xyz)
                stack.append(currentJoint)
                currentJoint = [0,0,0]

            # 处理完成 出栈
            if line == "}":
                stack.pop()
    """" 处理文件结束 """

    joint_offset = np.array(offset_list)    
    
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = np.zeros([len(joint_name),3])
    joint_orientations = np.zeros([len(joint_name),4])
    # 如何计算全局位置和全局朝向 每计算一个 放一个进joint_positions和joint_orientations数组
    # 关节的全局旋转 Qn = R0R1...Rn R为旋转矩阵
    # 我们思考 关节的全局位置如何计算 p_n+1 = p_n + Qn * ln 其中l为偏移量 Qn为全局旋转矩阵

    # 切片是左闭右开 设置根节点的信息
    joint_positions[0,:] = motion_data[frame_id,0:3]
    joint_orientations[0,:] = R.from_euler('XYZ', motion_data[frame_id,3:6], degrees=True).as_quat()
    # 通道数有几个 为1个位置加M个欧拉角 注意因末端点没有旋转 过为 1+M-E

    # 跳过第一和二个通道 前方已经进行初始化 分别为根节点的位置和朝向
    channel_num = 2
    # 如何处理末端点的问题? 按关节序号进行遍历
    for i in range(len(joint_name)): # n = 25 有25个关节
        # 根关节的位置和朝向已设置
        if i == 0:
            continue
        # 计算关节位置
        parent_position = joint_positions[joint_parent[i],:]
        parent_rot = R.from_quat(joint_orientations[joint_parent[i],:])
        offset = joint_offset[i,:]
        joint_positions[i,0:3] = parent_position + parent_rot.apply(offset)

        # 计算全局朝向
        if joint_name[i].endswith("_end"):
            joint_orientations[i,0:4] = R.from_euler('XYZ', [0, 0, 0], degrees=True).as_quat()
            continue
        # 获得相对旋转的四元数
        rot = R.from_euler('XYZ', motion_data[frame_id,channel_num*3 : channel_num*3+3], degrees=True)
        joint_orientations[i,0:4] = (parent_rot * rot).as_quat()
        channel_num += 1

    
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    # 我们要得到T-Pose的motion_data 我们使用的是A_pose_run的motion数据 它的初始姿态是A_Pose的 
    # 把它重定向为T_Pose
    # 为此 我们需要T_Pose和A_Pose的骨架 和A_pose的动作数据 T_Pose的骨架数据在walk60中
    # 即我们最终要得到符合T_Pose骨架顺序的motion_data
    T_pose_joint_name, T_pose_joint_parent, T_pose_joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    A_pose_joint_name, A_pose_joint_parent, A_pose_joint_offset = part1_calculate_T_pose(A_pose_bvh_path)
    A_pose_motion_data = load_motion_data(A_pose_bvh_path)
    # 注意 这里的joint name顺序不一致 但是父子关系是一致的 也就是说 是一个父节点的下的几个子节点排序不一样
    """
        A       A
       /|\     /|\ 
      B C D   C B D
      所以只需按名字读取即可
    """
    # 双方文件的差别 在offset不一样 对应的偏差角度不同 长度一致 我们要做的 就是把不同offset之间的旋转矩阵写出来

    """"先计算不同pose之间offset的旋转变换"""
    # 旋转变换 是每个关节都有一一对应的 Q_i^A=Q_pi^A * R_i^A 我们要存储的 即是Q_i^A
    # joint_transform: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的旋转变换(四元数) 按照T_pose的顺序存储
    joint_transform = np.zeros([len(T_pose_joint_name),4])
    # 初始化 赋值为单位四元数
    for i in range(len(A_pose_joint_name)):
        joint_transform[i,:] = R.identity().as_quat()
    # 注意 因为偏移的关系是在A_Pose中定义的 所以我们要按A_Pose的顺序遍历

    index_A = 0 # 根节点的赋值即为单位旋转 直接从1开始
    while index_A != len(A_pose_joint_name):
        
        if A_pose_joint_name[index_A].endswith("_end") or A_pose_joint_name[index_A] == "RootJoint":
            # 不做处理 因初始化为单元四元数
            index_A += 1
            continue
        
        index_T = T_pose_joint_name.index(A_pose_joint_name[index_A])
        # 两者偏移相等 用单位旋转 即不旋转
        if (T_pose_joint_offset[index_T,:] == A_pose_joint_offset[index_A,:]).all():
            # 不做处理 因初始化为单元四元数
            index_A += 1
        else:
            # 这里我们需要解决一个问题 已知两个向量 如何求一个对应其变换的旋转矩阵 注意 答案是不唯一的

            if A_pose_joint_name[index_A] == "lElbow":
                joint_transform[index_T,:] = R.from_euler('XYZ', [0.0, 0.0, -45.0], degrees=True).as_quat()
            elif A_pose_joint_name[index_A] == "rElbow":
                joint_transform[index_T,:] = R.from_euler('XYZ', [0.0, 0.0, 45.0], degrees=True).as_quat()
            else:
                parent_index = T_pose_joint_parent[index_T]
                joint_transform[index_T,:] = joint_transform[parent_index,:]
            
            index_A += 1



    """
        joint_transform已确定 现在确定motion_data
        R_i^B = Q_pi^A->B * R_i^A * (Q_i^A->B)^T
        R_pi^B = R_pi^A * (Q_pi^A->B)^T
        我们读取A_pose_motion_data 直接用这个做为计算数据再计算填充一个T_pose_motion_data 并返回

        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
    """
    print(joint_transform)
    T_pose_motion_data = np.zeros(A_pose_motion_data.shape)
    rows = A_pose_motion_data.shape[0]
    for row in range(rows):
        print(row)
        # 先放入位置数据
        T_pose_motion_data[row,0:3] = A_pose_motion_data[row,0:3]
        col = 3
        for i in range(len(T_pose_joint_name)):
            # 根节点可以无脑做
            if T_pose_joint_name[i] == "RootJoint":
                Q_AtoB_pi = R.from_quat(joint_transform[0,:])
                R_A_pi = R.from_euler('XYZ', A_pose_motion_data[row,3:6], degrees=True)
                T_pose_motion_data[row,3:6] = (R_A_pi * Q_AtoB_pi.inv()).as_euler("XYZ", degrees=True)
                col += 3
                continue
            elif T_pose_joint_name[i].endswith("_end"):
                # 注意 末端点在motion里没有数据 故不加3
                continue
            else:
                # 非根节点 要注意顺序是不一致的
                A_pose_list_index = A_pose_joint_name.index(T_pose_joint_name[i])
                # 问题 怎么找到其在A_pose_motion_data里的索引?
                end_num = 0
                for i in range(A_pose_list_index):
                    if A_pose_joint_name[i].endswith("_end"):
                        end_num += 1
                A_pose_motion_index = (A_pose_list_index - end_num) * 3 + 3
                R_A_i = R.from_euler('XYZ', A_pose_motion_data[row,A_pose_motion_index:A_pose_motion_index+3], degrees=True)

                Q_AtoB_i = R.from_quat(joint_transform[i,:])
                parent_index = T_pose_joint_parent[i]
                Q_AtoB_pi = R.from_quat(joint_transform[parent_index,:])
                R_B_i = (Q_AtoB_pi * R_A_i * Q_AtoB_i.inv()).as_euler("XYZ",degrees=True)
                end_num = 0
                for i in range(A_pose_list_index):
                    if A_pose_joint_name[i].endswith("_end"):
                        end_num += 1
                T_pose_motion_index = (i - end_num) * 3 + 3
                T_pose_motion_data[row,T_pose_motion_index:T_pose_motion_index+3] = R_B_i

                col += 3

    return T_pose_motion_data
