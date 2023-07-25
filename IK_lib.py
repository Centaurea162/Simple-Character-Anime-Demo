import numpy as np
from scipy.spatial.transform import Rotation as R
import math

def get_cos(vec_n,vec_m):
    return np.dot(vec_n,vec_m) / (np.linalg.norm(vec_n) * np.linalg.norm(vec_m))

def get_parent_from_path(meta_data):
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()

    joint_parent_form_fixed = [-2 for n in range(len(meta_data.joint_parent))]
    if 0 not in path or 0 == path[0]:
        joint_parent_form_fixed = meta_data.joint_parent
        return joint_parent_form_fixed
    else:
        # 父节点为固定关节 则固定关节的原父节点和原子节点均为子节点
        queue_parent = [path[0]]
        joint_parent_form_fixed[path[0]] = -1
        while len(queue_parent) != 0:
            cur_index = queue_parent.pop()
            # 找原父节点
            # 问题出在 当遍历到原根关节时 index = -1 这个怎么处理 答 不处理它的父节点 只处理它的子节点
            if cur_index != -1:
                parent = meta_data.joint_parent[cur_index]
                if joint_parent_form_fixed[parent] == -2:
                    joint_parent_form_fixed[parent] = cur_index
                    queue_parent = [parent] + queue_parent
            # 找原子节点
            children = [i for i in range(len(meta_data.joint_parent)) if meta_data.joint_parent[i] == cur_index]
            for child in children:
                if joint_parent_form_fixed[child] != -2:
                    children.remove(child)
                else:
                    joint_parent_form_fixed[child] = cur_index
            queue_parent = children + queue_parent
        return joint_parent_form_fixed


def getJointPathInfo(meta_data, joint_positions, joint_orientations):
    """
    辅助函数，返回从固定节点到end节点的路径的数据信息

    path: 各个关节的索引
    path_name: 各个关节的名字

    输出：
        path_offsets:各个节点以固定节点为根节点的偏移
        path_positions:各个节点在原始Pose的位置
        path_orientations:各个节点在原始Pose的朝向
    """
    # calculate joint position
    # calculate joint orientation
    path_positions = []
    path_orientations = []
    for joint in meta_data.path:
        path_positions.append(joint_positions[joint])
        path_orientations.append(R.from_quat(joint_orientations[joint]))

    # calculate joint offset
    path_offsets = []
    path_offsets.append(np.array([0., 0., 0.]))
    for i in range(len(meta_data.path) - 1):
        path_offsets.append(meta_data.joint_initial_position[meta_data.path[i + 1]] - meta_data.joint_initial_position[meta_data.path[i]])

    return path_offsets, path_positions, path_orientations



def cyclicCoordinateDescent(meta_data, joint_positions, joint_orientations, target_pose):
    # 计算 inverse_kinematics 链的信息
    path_offsets, path_positions, path_orientations = getJointPathInfo(meta_data, joint_positions, joint_orientations)
    # CCD 循环
    cnt = 0

    while (np.linalg.norm(joint_positions[meta_data.path[-1]] - target_pose) >= 1e-2 and cnt <= 10):
        for index in reversed(meta_data.path):
            # 跳过固定关节和路径末端点
            if index == meta_data.path[-1]:
                continue
            # 当前关节在path_XXX的索引
            path_index = meta_data.path.index(index)
            # CCD的思想 让l_mn与目标位置共线 l_mn = posi_n - posi_m 即是与l_mt共线
            l_mn = path_positions[-1] - path_positions[path_index]
            l_mp = target_pose - path_positions[path_index]
            l_mn = l_mn / np.linalg.norm(l_mn)
            l_mp = l_mp / np.linalg.norm(l_mp)


            # 得到旋转对齐矩阵 两种方式
            
            # 调用scipy库 但是数值不是非常稳定
            # rots = R.align_vectors(l_mp.reshape(1,3),l_mn.reshape(1,3))
            # rot = rots[0]

            # 更加数值稳定的方法
            rotation_radius = np.arccos(np.clip(np.dot(l_mn, l_mp), -1, 1))
            current_axis = np.cross(l_mn, l_mp)
            rotation_axis = current_axis / np.linalg.norm(current_axis)
            rotation_vector = R.from_rotvec(rotation_radius * rotation_axis)


            # 应用旋转
            for j in range(path_index,len(meta_data.path) - 1):
                # 更新 path_orientations
                path_orientations[j] = rotation_vector * path_orientations[j]
                # 更新 path_positions
                path_positions[j + 1] = path_positions[j] + path_orientations[j].apply(path_offsets[j + 1])
            # 记得把最后一个节点的更新补上
            path_orientations[-1] = rotation_vector * path_orientations[-1]
        cnt += 1
    return path_positions, path_orientations



def calculateJointAngle(path_orientations):
    # joint_angle = Ri
    joint_angle = []
    joint_angle.append(path_orientations[0].as_euler('XYZ', degrees=True))
    for i in range(len(path_orientations) - 1):
        joint_angle.append((R.inv(path_orientations[i]) * path_orientations[i + 1]).as_euler('XYZ', degrees=True))

    return joint_angle

def calculateJacobian(end_position, joint_angle, path_positions, path_orientations):
    jacobian = []

    for i in range(len(path_orientations)):
        r_i = end_position - path_positions[i]
        current_joint = joint_angle[i]

        R_ix = R.from_euler('XYZ', [current_joint[0], 0, 0], degrees=True)
        R_ixy = R.from_euler('XYZ', [current_joint[0], current_joint[1], 0], degrees=True)

        e_x = np.array([1., 0., 0.]).reshape(-1,3)
        e_y = np.array([0., 1., 0.]).reshape(-1,3)
        e_z = np.array([0., 0., 1.]).reshape(-1,3)

        Q_prev = None
        if i == 0:
            Q_prev = R.identity()
        else:
            Q_prev = path_orientations[i - 1]

        a_ix = Q_prev.apply(e_x) 
        a_iy = (Q_prev * R_ix).apply(e_y)
        a_iz = (Q_prev * R_ixy).apply(e_z)

        jacobian.append(np.cross(a_ix,r_i))
        jacobian.append(np.cross(a_iy,r_i))
        jacobian.append(np.cross(a_iz,r_i))

    jacobian = np.concatenate(jacobian, axis=0).transpose()
    return jacobian

def calculateJointPathInJacobian(theta, end_index, path_offsets, path_positions, path_orientations):
    path_rotations = []
    theta = theta.reshape(-1,3)
    for i in range(len(theta)):
        eula = theta[i]
        path_rotations.append(R.from_euler('XYZ', eula, degrees=True))

    # update joint rotations R_{i} = Q_{i-1}^T Q_{i}
    path_orientations[0] = path_rotations[0]
    for j in range(len(path_positions) - 1):
        path_positions[j + 1] = path_positions[j] + path_orientations[j].apply(path_offsets[j + 1])
        if j + 1 < end_index:
            path_orientations[j + 1] = path_orientations[j] * path_rotations[j + 1]
        else:
            path_orientations[j + 1] = path_orientations[j]
    return path_positions, path_orientations


def gradientDescent(meta_data, joint_positions, joint_orientations, target_pose):
    # 计算 inverse_kinematics 链的信息
    path_offsets, path_positions, path_orientations = getJointPathInfo(meta_data, joint_positions, joint_orientations)

    end_index = meta_data.path_name.index(meta_data.end_joint)
    count = 0
    while (np.linalg.norm(path_positions[-1] - target_pose) >= 0.01 and count <= 10):
        end_position = path_positions[-1]
        joint_angle = calculateJointAngle(path_orientations)
        jacobian = calculateJacobian(end_position, joint_angle, path_positions, path_orientations)

        # get all path rotations, convert to XYZ euler angle
        theta = np.concatenate(joint_angle, axis=0).transpose().reshape(-1, 1)

        distance = path_positions[-1] - target_pose
        alpha = 100

        # theta_i+1 = theta_i - alpha J^T * distance 
        delta = alpha * np.dot(jacobian.transpose(), distance.reshape(3,1))
        
        theta = theta - delta
        
    # convert theta back to rotations
        path_positions, path_orientations = calculateJointPathInJacobian(theta, end_index, path_offsets, path_positions, path_orientations)

        count += 1
        pass
    

    return path_positions, path_orientations


def applyFullBodyIK(meta_data, joint_positions, joint_orientations, path_positions, path_orientations):

    # 计算 path_joints 在原始Pose下的 rotation
    joint_rotations = R.identity(len(meta_data.joint_name))
    for i in range(len(meta_data.joint_parent)):
        if meta_data.joint_parent[i] == -1:
            joint_rotations[i] = R.from_quat(joint_orientations[i])
        else:
            joint_rotations[i] = R.inv(R.from_quat(joint_orientations[meta_data.joint_parent[i]])) * R.from_quat(joint_orientations[i])

    # # apply IK rotation result
    if len(meta_data.path2) > 1: # we have locked sub chain
        # path_joints 的 forward_kinematicspath2返回从脚到根节点的路径
        # have to set {i}'s joint orientation with {i-1}' orientation
        # because end's joint is not a valid joint
        # path2返回从脚到根节点的路径

        for i in range(len(meta_data.path2) - 1):
            # 注意 我们跳过了path2[0]固定关节 不作旋转 所以是range(len(meta_data.path2) - 1) 且path_orientations[i]赋给了meta_data.path2[i + 1]
            joint_orientations[meta_data.path2[i + 1]] = path_orientations[i].as_quat()

        for i in range(len(meta_data.path1) - 1):
            # 我们跳过了end关节 不作旋转 所以是range(len(meta_data.path1) - 1)
            joint_orientations[meta_data.path[i + len(meta_data.path2)]] = path_orientations[i + len(meta_data.path2)].as_quat()


    else: # we don't have locked sub chain
        # path_joints 的 forward_kinematics
        for j in range(len(meta_data.path)):
            joint_orientations[meta_data.path[j]] = path_orientations[j].as_quat()

    # apply IK position result
    for i in range(len(meta_data.path)):
        joint_positions[meta_data.path[i]] = path_positions[i]

    # 其余 joints 的 forward_kinematics
    # 直接按列表顺序更新 即能保证 在关节更新时 父关节已经更新完成
    for i in range(len(meta_data.joint_parent)):
        if meta_data.joint_parent[i] == -1:
            continue
        if meta_data.joint_name[i] not in meta_data.path_name:
            # print(joint_orientations[meta_data.joint_parent[i]])
            joint_positions[i] = joint_positions[meta_data.joint_parent[i]] + \
                R.from_quat(joint_orientations[meta_data.joint_parent[i]]).apply(meta_data.joint_initial_position[i] - \
                meta_data.joint_initial_position[meta_data.joint_parent[i]])
            joint_orientations[i] = (R.from_quat(joint_orientations[meta_data.joint_parent[i]]) * joint_rotations[i]).as_quat()
    return joint_positions, joint_orientations

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    path_positions, path_orientations = gradientDescent(meta_data, joint_positions, joint_orientations, target_pose)
    joint_positions, joint_orientations = applyFullBodyIK(meta_data, joint_positions, joint_orientations, path_positions, path_orientations)

    return joint_positions, joint_orientations



            
    

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    target_pose = np.array([joint_positions[0][0] + relative_x, target_height, joint_positions[0][2] + relative_z])
    path_positions, path_orientations = gradientDescent(meta_data, joint_positions, joint_orientations, target_pose)
    joint_positions, joint_orientations = applyFullBodyIK(meta_data, joint_positions, joint_orientations, path_positions, path_orientations)

    return joint_positions, joint_orientations
    


def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations