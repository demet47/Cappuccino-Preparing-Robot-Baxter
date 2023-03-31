import pybullet as p
import pybullet_data
from manipulator import Manipulator
import pandas as pd
import torch
import os
import pdb
import numpy as np
import sys

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)
p.setRealTimeSimulation(0)
planeId=p.loadURDF("plane.urdf")

baxter = Manipulator(p, "./baxter_common/baxter_description/urdf/toms_baxter.urdf", position=(0,0,0.9), ik_idx=20)
baxter.add_debug_text()

joint_names = [joint_name.decode() for joint_name in baxter.names]


# labels for ease! !!NOTE!!!: demos from baxter contains time values, predictons do not!!!
labels = ["time","left_s0","left_s1","left_e0","left_e1","left_w0","left_w1",\
            "left_w2","left_gripper","right_s0","right_s1","right_e0","right_e1",\
            "right_w0","right_w1", "right_w2","right_gripper"]


debug_line_ids = [] # to remove them when needed


# visualize the demo trajectory:
    
while debug_line_ids:   p.removeUserDebugItem(debug_line_ids.pop())



#triangular
trajectory1 = [(0,-1.4,1),(0.1,-1.3,1),(0.2,-1.2,1),(0.3, -1.1,1),(0.4,-1.0,1),(0.5,-0.9,1),
               (0.6,-0.8,1),(0.7,-0.7,1),(0.75,-0.6,1),(0.8,-0.5,1),(0.85,-0.3,1),(0.9,-0.1,1),(1,0,1),
               (0.9,0.05,1),(0.85,0.1,1),(0.8,0.15,1),(0.75,0.2,1),(0.7,0.25,1),(0.6,0.3,1),
               (0.5,0.35,1),(0.4,0.4,1),(0.3,0.45,1),(0.25,0.5,1),(0.2,0.55,1),(0.1,0.6,1)]
trajectory1 = [baxter.convert_to_joint_data(p, None) for p in trajectory1]

#polinomial
trajectory2 = [(0, -1.2, 1.1424743488113782), (0.005929,-1.3,1.1) , (0.023716,-1.4,1),(0.053361,-1.2833,1),(0.094864,-1.1666,1),(0.148225, -1.05,1),(0.213444,-0.9333,1),(0.290521,-0.8166,1),
                (0.379456,-0.7,1),(0.480249, -0.5833,1),(0.5929,-0.4666,1),(0.717409, -0.35,1),(0.853776,-0.2333,1),(1,-0.1166,1),
               (1,0,1),(0.843,0.05,1),(0.6995,0.1,1),(0.5693,0.15, 1),(0.45256,0.2,1),(0.3492,0.25,1),
               (0.259, 0.3,1),(0.18256,0.35,1),(0.1194,0.4,1),(0.0695,0.45,1),(0.033,0.5,1),(0.1,0.55,1),(0.1,0.6,1)]
trajectory2 = [baxter.convert_to_joint_data(p, None) for p in trajectory2]
    
#impuls-like 
trajectory3 = [(0,-1.4,1),(0,-1.2833,1),(0,-1.1666,1),(0, -1.05,1),(0,-0.9333,1),(0, -0.8166,1),
                (0,-0.7,1),(0,-0.5833,1),(0.5,-0.4666,1),(0.5, -0.35, 1),(0.5,-0.2333,1),(0.5,-0.1166,1),(1,0,1),
               (0.5,0.05,1),(0.5,0.1,1),(0.5,0.15,1),(0.5,0.2,1),(0.1,0.25,1),(0.1,0.3,1),
               (0.1,0.35,1),(0.1,0.4,1),(0.1,0.45,1),(0.1,0.5,1),(0.1,0.55,1),(0.1,0.6,1)]
trajectory3 = [baxter.convert_to_joint_data(p, None) for p in trajectory3]

#step-by-step
trajectory4 = [(0,-1.4,1),(0.2,-1.4,1),(0.2,-1.2,1),(0.4,-1.2,1),(0.4,-1,1),(0.6,-1,1),
                (0.6,-0.8,1),(0.8,-0.8,1),(0.8,-0.6,1),(1,-0.6,1),(1,-0.4,1),(1,-0.4,1),(1,0,1),
               (0.8,0,1),(0.8,0.1,1),(0.6,0.1,1),(0.6,0.2,1),(0.4,0.2,1),(0.4,0.3,1),
               (0.3,0.3,1),(0.3,0.4,1),(0.2,0.4,1),(0.2,0.5,1),(0.1,0.5,1),(0.1,0.6,1)]
trajectory4 = [baxter.convert_to_joint_data(p, None) for p in trajectory4]
    
trajectory5 = [(0.4, 0.0, 1), (0.3999989899244969, 0.0010050251256281408, 1), (0.39999595969798746, 0.0020100502512562816, 1), (0.3999909093204717, 0.0030150753768844224, 1), (0.3999838387919497, 0.004020100502512563, 1), (0.39997474811242145, 0.0050251256281407045, 1), (0.39996363728188683, 0.006030150753768845, 1), (0.39995050630034595, 0.007035175879396985, 1), (0.3999353551677988, 0.008040201005025126, 1), (0.3999181838842454, 0.009045226130653268, 1), (0.39989899244968563, 0.010050251256281409, 1), (0.3998777808641196, 0.011055276381909548, 1), (0.3998545491275473, 0.01206030150753769, 1), (0.3998292972399687, 0.013065326633165831, 1), (0.3998020252013838, 0.01407035175879397, 1), (0.39977273301179267, 0.015075376884422112, 1), (0.39974142067119517, 0.016080402010050253, 1), (0.39970808817959147, 0.017085427135678392, 1), (0.3996727355369814, 0.018090452261306535, 1), (0.3996353627433651, 0.019095477386934675, 1), (0.39959596979874246, 0.020100502512562818, 1), (0.3995545567031136, 0.021105527638190957, 1), (0.3995111234564784, 0.022110552763819097, 1), (0.39946567005883693, 0.02311557788944724, 1), (0.39941819651018917, 0.02412060301507538, 1), (0.3993687028105351, 0.02512562814070352, 1), (0.3993171889598748, 0.026130653266331662, 1), (0.39926365495820815, 0.0271356783919598, 1), (0.39920810080553526, 0.02814070351758794, 1), (0.39915052650185606, 0.029145728643216084, 1), (0.39909093204717055, 0.030150753768844223, 1), (0.3990293174414788, 0.031155778894472366, 1), (0.3989656826847807, 0.032160804020100506, 1), (0.39890002777707634, 0.033165829145728645, 1), (0.3988323527183657, 0.034170854271356785, 1), (0.3987626575086488, 0.03517587939698493, 1), (0.39869094214792555, 0.03618090452261307, 1), (0.3986172066361961, 0.03718592964824121, 1), (0.3985414509734603, 0.03819095477386935, 1), (0.3984636751597182, 0.03919597989949749, 1), (0.39838387919496987, 0.040201005025125636, 1), (0.3983020630792152, 0.041206030150753775, 1), (0.3982182268124543, 0.042211055276381915, 1), (0.398132370394687, 0.043216080402010054, 1), (0.3980444938259135, 0.044221105527638194, 1), (0.3979545971061337, 0.04522613065326633, 1), (0.3978626802353476, 0.04623115577889448, 1), (0.3977687432135552, 0.04723618090452262, 1), (0.39767278604075657, 0.04824120603015076, 1), (0.3975748087169516, 0.0492462311557789, 1), (0.39747481124214035, 0.05025125628140704, 1), (0.39737279361632283, 0.051256281407035184, 1), (0.397268755839499, 0.052261306532663324, 1), (0.3971626979116689, 0.05326633165829146, 1), (0.3970546198328325, 0.0542713567839196, 1), (0.3969445216029898, 0.05527638190954774, 1), (0.39683240322214086, 0.05628140703517588, 1), (0.3967182646902856, 0.05728643216080403, 1), (0.39660210600742407, 0.05829145728643217, 1), (0.39648392717355624, 0.05929648241206031, 1), (0.39636372818868215, 0.06030150753768845, 1), (0.3962415090528017, 0.061306532663316586, 1), (0.39611726976591505, 0.06231155778894473, 1), (0.39599101032802203, 0.06331658291457287, 1), (0.39586273073912276, 0.06432160804020101, 1), (0.39573243099921723, 0.06532663316582915, 1), (0.39560011110830534, 0.06633165829145729, 1), (0.39546577106638725, 0.06733668341708543, 1), (0.3953294108734628, 0.06834170854271357, 1), (0.3951910305295321, 0.06934673366834171, 1), (0.3950506300345951, 0.07035175879396986, 1), (0.39490820938865184, 0.071356783919598, 1), (0.39476376859170226, 0.07236180904522614, 1), (0.3946173076437464, 0.07336683417085428, 1), (0.39446882654478427, 0.07437185929648242, 1), (0.3943183252948158, 0.07537688442211056, 1), (0.3941658038938411, 0.0763819095477387, 1), (0.39401126234186007, 0.07738693467336684, 1), (0.3938547006388728, 0.07839195979899498, 1), (0.3936961187848792, 0.07939698492462312, 1), (0.3935355167798793, 0.08040201005025127, 1), (0.39337289462387315, 0.08140703517587941, 1), (0.3932082523168607, 0.08241206030150755, 1), (0.39304158985884197, 0.08341708542713569, 1), (0.39287290724981694, 0.08442211055276383, 1), (0.39270220448978566, 0.08542713567839197, 1), (0.392529481578748, 0.08643216080402011, 1), (0.39235473851670416, 0.08743718592964825, 1), (0.39217797530365395, 0.08844221105527639, 1), (0.3919991919395975, 0.08944723618090453, 1), (0.39181838842453476, 0.09045226130653267, 1), (0.3916355647584657, 0.09145728643216082, 1), (0.3914507209413904, 0.09246231155778896, 1), (0.3912638569733088, 0.0934673366834171, 1), (0.3910749728542209, 0.09447236180904524, 1), (0.39088406858412666, 0.09547738693467338, 1), (0.3906911441630262, 0.09648241206030152, 1), (0.39049619959091947, 0.09748743718592966, 1), (0.3902992348678064, 0.0984924623115578, 1), (0.39010024999368703, 0.09949748743718594, 1), (0.38989924496856143, 0.10050251256281408, 1), (0.3896962197924295, 0.10150753768844221, 1), (0.3894911744652913, 0.10251256281407037, 1), (0.3892841089871468, 0.10351758793969851, 1), (0.38907502335799604, 0.10452261306532665, 1), (0.38886391757783895, 0.10552763819095479, 1), (0.3886507916466756, 0.10653266331658293, 1), (0.388435645564506, 0.10753768844221107, 1), (0.38821847933133, 0.1085427135678392, 1), (0.3879992929471478, 0.10954773869346734, 1), (0.3877780864119593, 0.11055276381909548, 1), (0.38755485972576453, 0.11155778894472362, 1), (0.38732961288856343, 0.11256281407035176, 1), (0.3871023459003561, 0.11356783919597992, 1), (0.3868730587611424, 0.11457286432160806, 1), (0.38664175147092245, 0.1155778894472362, 1), (0.3864084240296962, 0.11658291457286434, 1), (0.38617307643746374, 0.11758793969849247, 1), (0.3859357086942249, 0.11859296482412061, 1), (0.38569632079997984, 0.11959798994974875, 1), (0.38545491275472843, 0.1206030150753769, 1), (0.38521148455847076, 0.12160804020100503, 1), (0.3849660362112068, 0.12261306532663317, 1), (0.38471856771293655, 0.12361809045226133, 1), (0.38446907906366, 0.12462311557788947, 1), (0.3842175702633772, 0.1256281407035176, 1), (0.3839640413120881, 0.12663316582914574, 1), (0.3837084922097927, 0.12763819095477388, 1), (0.383450922956491, 0.12864321608040202, 1), (0.38319133355218304, 0.12964824120603016, 1), (0.3829297239968688, 0.1306532663316583, 1), (0.3826660942905482, 0.13165829145728644, 1), (0.3824004444332214, 0.13266331658291458, 1), (0.3821327744248883, 0.13366834170854272, 1), (0.3818630842655489, 0.13467336683417086, 1), (0.38159137395520315, 0.135678391959799, 1), (0.38131764349385117, 0.13668341708542714, 1), (0.3810418928814929, 0.13768844221105528, 1), (0.3807641221181284, 0.13869346733668342, 1), (0.3804843312037575, 0.13969849246231159, 1), (0.38020252013838035, 0.14070351758793972, 1), (0.3799186889219969, 0.14170854271356786, 1), (0.3796328375546072, 0.142713567839196, 1), (0.3793449660362112, 0.14371859296482414, 1), (0.37905507436680896, 0.14472361809045228, 1), (0.37876316254640036, 0.14572864321608042, 1), (0.3784692305749855, 0.14673366834170856, 1), (0.3781732784525643, 0.1477386934673367, 1), (0.3778753061791369, 0.14874371859296484, 1), (0.37757531375470316, 0.14974874371859298, 1), (0.37727330117926317, 0.15075376884422112, 1), (0.37696926845281686, 0.15175879396984926, 1), (0.37666321557536425, 0.1527638190954774, 1), (0.3763551425469054, 0.15376884422110554, 1), (0.3760450493674402, 0.15477386934673368, 1), (0.3757329360369688, 0.15577889447236182, 1), (0.37541880255549104, 0.15678391959798996, 1), (0.37510264892300704, 0.1577889447236181, 1), (0.3747844751395167, 0.15879396984924624, 1), (0.3744642812050201, 0.15979899497487438, 1), (0.3741420671195172, 0.16080402010050254, 1), (0.373817832883008, 0.16180904522613068, 1), (0.3734915784954925, 0.16281407035175882, 1), (0.3731633039569708, 0.16381909547738696, 1), (0.3728330092674428, 0.1648241206030151, 1), (0.3725006944269084, 0.16582914572864324, 1), (0.3721663594353678, 0.16683417085427138, 1), (0.3718300042928209, 0.16783919597989952, 1), (0.3714916289992677, 0.16884422110552766, 1), (0.37115123355470825, 0.1698492462311558, 1), (0.37080881795914244, 0.17085427135678394, 1), (0.3704643822125704, 0.17185929648241208, 1), (0.3701179263149921, 0.17286432160804022, 1), (0.36976945026640745, 0.17386934673366836, 1), (0.3694189540668165, 0.1748743718592965, 1), (0.36906643771621933, 0.17587939698492464, 1), (0.36871190121461583, 0.17688442211055277, 1), (0.368355344562006, 0.17788944723618091, 1), (0.36799676775838996, 0.17889447236180905, 1), (0.3676361708037676, 0.1798994974874372, 1), (0.36727355369813897, 0.18090452261306533, 1), (0.36690891644150403, 0.18190954773869347, 1), (0.3665422590338628, 0.18291457286432164, 1), (0.3661735814752153, 0.18391959798994978, 1), (0.36580288376556147, 0.18492462311557792, 1), (0.3654301659049014, 0.18592964824120606, 1), (0.36505542789323503, 0.1869346733668342, 1), (0.3646786697305624, 0.18793969849246234, 1), (0.3642998914168834, 0.18894472361809048, 1), (0.3639190929521982, 0.18994974874371862, 1), (0.36353627433650665, 0.19095477386934676, 1), (0.36315143556980883, 0.1919597989949749, 1), (0.36276457665210476, 0.19296482412060303, 1), (0.3623756975833944, 0.19396984924623117, 1), (0.3619847983636777, 0.1949748743718593, 1), (0.36159187899295475, 0.19597989949748745, 1), (0.3611969394712255, 0.1969849246231156, 1), (0.36079997979848993, 0.19798994974874373, 1), (0.3604009999747481, 0.19899497487437187, 1), (0.36, 0.2, 1)]
trajectory5 = [baxter.convert_to_joint_data(p, None) for p in trajectory5]


circle_trajectory = np.load('circle_trajectory.npy')
circle_trajectory = [baxter.convert_to_joint_data(p, None) for p in circle_trajectory]
np.save('circle_trajectory_joint',circle_trajectory)

one_step_trajectory = np.load('one_step_trajectory.npy')
one_step_trajectory = [baxter.convert_to_joint_data(p, None) for p in one_step_trajectory]
np.save('one_step_trajectory_joint',one_step_trajectory)


linear_trajectory = np.load('linear_trajectory.npy')
linear_trajectory = [baxter.convert_to_joint_data(p, None) for p in linear_trajectory]
np.save('linear_trajectory_joint',linear_trajectory)


sinus_trajectory = np.load('sinus_trajectory.npy')
sinus_trajectory = [baxter.convert_to_joint_data(p, None) for p in sinus_trajectory]
np.save('sinus_trajectory_joint',sinus_trajectory)


    
dictionary = {"linear": linear_trajectory, "circular": circle_trajectory, "step":one_step_trajectory, "sinus": sinus_trajectory, "triangular": trajectory1
                  , "polynomial": trajectory2, "impuls": trajectory3, "multi-step":trajectory4}


flag = input("Do you want to demonstrate an input trajectory? (Y/N): ")
if(flag == "N"):
    sys.exit()


t_name = input("Please enter what trajectory to demonstrate: ", )



print("visualizing the demonstration...")
line_start=None
trajectory = dictionary[t_name]
baxter.set_joint_position(position=trajectory[0], t=0.1)
for position_xyz in trajectory:
    #print("given position: ",position_xyz)
    baxter.set_joint_position(position=position_xyz, t=0.1, sleep=False)
    #baxter.set_joint_position(position=position_xyz, t=0.005, sleep=False, traj=True)
    end_effector_position = baxter.get_end_effector_pose()
    #print("position robot reaches: ",end_effector_position)

    print("diff = " ,(position_xyz[0] + position_xyz[0] + position_xyz[0] - end_effector_position[0] - end_effector_position[1] - end_effector_position[2]))
    if line_start is not None:
        line_id = baxter.add_line(line_start, end_effector_position, color=[0., 0., 0.], lineWidth=3)
        debug_line_ids.append(line_id)
    line_start=end_effector_position
       