import pybullet as p
import pybullet_data
from manipulator import Manipulator
import csv

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)
p.setRealTimeSimulation(0)
planeId=p.loadURDF("plane.urdf")

baxter = Manipulator(p, "./baxter_common/baxter_description/urdf/toms_baxter.urdf", position=(0,0,0.9), ik_idx=20)
baxter.add_debug_text()

joint_names = [joint_name.decode() for joint_name in baxter.names]


'''
file for setting up the pybullet scene

BELOW ARE SOME SPECIAL KEYS FOR PYBULLET SIMULATOR SCREEN:
ctrl + G --> for closing the extension windows in simulator screen
ctrl + left click
ctrl + right click
'''



# labels for ease! !!NOTE!!!: demos from baxter contains time values, predictons do not!!!
labels = ["time","left_s0","left_s1","left_e0","left_e1","left_w0","left_w1",\
            "left_w2","left_gripper","right_s0","right_s1","right_e0","right_e1",\
            "right_w0","right_w1", "right_w2","right_gripper"]


test_dataset = []
with open('./code_inventory/liquid_pour_train_data/salimdemet.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        test_dataset.append(row)


test_dataset = test_dataset[1:]
debug_line_ids = [] # to remove them when needed


    
while debug_line_ids:   p.removeUserDebugItem(debug_line_ids.pop())

print("visualizing the demonstration...")
line_start=None
for joint_states in test_dataset:
    position=[float(joint_states[labels.index(joint_name)]) if(joint_name in labels) else 0.0 for joint_name in joint_names]
    baxter.set_joint_position(position=position, t=0.01, sleep=False, traj=True)
    end_effector_position = baxter.get_end_effector_pose()
    #print(end_effector_position)
    if line_start is not None:
        line_id = baxter.add_line(line_start, end_effector_position, color=[0., 0., 0.], lineWidth=3)
        debug_line_ids.append(line_id)
    line_start=end_effector_position
    


    """
    print("visualizing the predictions...")
    line_start=None
    pred_key = os.path.splitext(key)[0]
    for joint_states in pred[pred_key]["mean"]:
        position=[float(joint_states[labels.index(joint_name)-1]) if(joint_name in labels) else 0.0 for joint_name in joint_names]
        baxter.set_joint_position(position=position)
        end_effector_position = baxter.get_end_effector_pose()
        if line_start is not None:
            line_id = baxter.add_line(line_start, end_effector_position, lineWidth=3)
            debug_line_ids.append(line_id)
        line_start=end_effector_position
        p.stepSimulation()
"""