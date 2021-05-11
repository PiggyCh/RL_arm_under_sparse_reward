import os
from gym import utils
from  pybullet_learning.bmiribot_env.bmirobot_env_v4_slide import bmirobotGymEnv

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push.xml')

class bmirobotGympushEnv(bmirobotGymEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        bmirobotGymEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)