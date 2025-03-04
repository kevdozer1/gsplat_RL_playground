import os
import time
import random
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces


class SimpleRobotEnv(gym.Env):
    """
    A simple environment with a robotic arm and a cube for basic manipulation tasks.
    This is the starting point for the PLAYGROUND project.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, render_mode=None, max_episode_steps=1000):
        super().__init__()
        
        # Rendering setup
        self.render_mode = render_mode
        
        # PyBullet setup
        if render_mode == "human":
            self.client = p.connect(p.GUI)  # Graphical interface
        else:
            self.client = p.connect(p.DIRECT)  # Headless
        
        # Environment configuration
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Path for robot URDFs and object models
        self.urdf_path = pybullet_data.getDataPath()
        
        # Object types configuration
        self.object_types = [
            {'type': 'box', 'halfExtents': [0.05, 0.05, 0.05], 'color': [1, 0, 0, 1]},  # Red box
            {'type': 'sphere', 'radius': 0.05, 'color': [0, 1, 0, 1]},                  # Green sphere
            {'type': 'cylinder', 'radius': 0.05, 'height': 0.1, 'color': [0, 0, 1, 1]}  # Blue cylinder
        ]
        
        # Robot and object IDs
        self.robot_id = None
        self.object_id = None
        self.target_id = None
        
        # Control parameters
        self.robot_joints = []
        self.num_joints = 0
        self.end_effector_index = None
        
        # Action and observation spaces
        # Simple position control for robot arm joints
        # For simplicity in this first iteration, we'll control 4 joints
        self.num_control_joints = 4
        
        # Action space: joint position targets for controlled joints
        # Each joint can be moved within its limits
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.num_control_joints,),
            dtype=np.float32
        )
        
        # Observation space: joint positions, velocities, end-effector position, 
        # object position, object orientation, object-EE distance
        observation_dim = (
            self.num_control_joints +  # Joint positions
            self.num_control_joints +  # Joint velocities
            3 +                        # End-effector position
            3 +                        # Object position
            4 +                        # Object orientation (quaternion)
            1                          # Distance to object
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(observation_dim,),
            dtype=np.float32
        )
        
        # Reset the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset counters and state
        self.current_step = 0
        
        # Reset simulation
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(self.urdf_path)
        
        # Load plane
        p.loadURDF("plane.urdf")
        
        # Load robot (using UR5 as an example, but this can be changed)
        # For the first iteration, we'll use a simple robot like the KUKA arm from PyBullet examples
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
        
        # Get joint information
        self.num_joints = p.getNumJoints(self.robot_id)
        self.robot_joints = list(range(self.num_joints))
        
        # Identify end-effector
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[12].decode('utf-8') == "lbr_iiwa_link_7":
                self.end_effector_index = i
                break
        
        # Reset all joints to home position
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, 0)
        
        # Remove any existing object
        if hasattr(self, 'object_id') and self.object_id is not None:
            p.removeBody(self.object_id)
        
        # Create a new object with randomized position
        x_pos = 0.5 + np.random.uniform(-0.1, 0.1)
        y_pos = 0.0 + np.random.uniform(-0.1, 0.1)
        object_position = [x_pos, y_pos, 0.05]
        
        # Randomly select an object type
        self.selected_object = random.choice(self.object_types)
        
        # Create collision and visual shapes based on the selected object type
        if self.selected_object['type'] == 'box':
            collision_shape_id = p.createCollisionShape(
                p.GEOM_BOX, 
                halfExtents=self.selected_object['halfExtents']
            )
            visual_shape_id = p.createVisualShape(
                p.GEOM_BOX, 
                halfExtents=self.selected_object['halfExtents'], 
                rgbaColor=self.selected_object['color']
            )
        elif self.selected_object['type'] == 'sphere':
            collision_shape_id = p.createCollisionShape(
                p.GEOM_SPHERE, 
                radius=self.selected_object['radius']
            )
            visual_shape_id = p.createVisualShape(
                p.GEOM_SPHERE, 
                radius=self.selected_object['radius'], 
                rgbaColor=self.selected_object['color']
            )
        elif self.selected_object['type'] == 'cylinder':
            collision_shape_id = p.createCollisionShape(
                p.GEOM_CYLINDER, 
                radius=self.selected_object['radius'], 
                height=self.selected_object['height']
            )
            visual_shape_id = p.createVisualShape(
                p.GEOM_CYLINDER, 
                radius=self.selected_object['radius'], 
                length=self.selected_object['height'], 
                rgbaColor=self.selected_object['color']
            )
        
        # Create the object in the simulation
        self.object_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=object_position,
            baseOrientation=[0, 0, 0, 1]
        )
        
        # Define target position (visual marker)
        visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 0.7])
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_id,
            basePosition=[1.5, 0, 0.5]
        )
        
        # Let physics settle
        for _ in range(20):
            p.stepSimulation()
        
        # Get observation
        observation = self._get_observation()
        info = {}
        
        return observation, info

    def step(self, action):
        """Execute action and return new state, reward, done, and info."""
        self.current_step += 1
        
        # Convert normalized actions (-1 to 1) to actual joint positions
        # We'll control the first num_control_joints joints of the robot
        for i in range(min(self.num_control_joints, self.num_joints)):
            # Simple position control
            # Map from [-1,1] to reasonable joint limits
            # For simplicity, using a fixed range; in a real implementation, use actual joint limits
            target_position = action[i] * np.pi/2  # Map to [-pi/2, pi/2]
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=target_position,
                force=500,
                maxVelocity=1
            )
        
        # Simulate one step
        p.stepSimulation()
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._compute_reward(observation)
        
        # Check if episode is done - IMPORTANT: convert to Python booleans
        terminated = bool(self._is_terminated(observation))
        truncated = bool(self.current_step >= self.max_episode_steps)
        
        # Additional info
        info = {}
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Get the current state observation."""
        # Get joint states for the controlled joints
        joint_states = []
        joint_velocities = []
        
        for i in range(min(self.num_control_joints, self.num_joints)):
            state = p.getJointState(self.robot_id, i)
            joint_states.append(state[0])
            joint_velocities.append(state[1])
        
        # Get end-effector position
        if self.end_effector_index is not None:
            ee_state = p.getLinkState(self.robot_id, self.end_effector_index)
            ee_position = ee_state[0]
        else:
            ee_position = (0, 0, 0)
        
        # Get object position and orientation
        obj_position, obj_orientation = p.getBasePositionAndOrientation(self.object_id)
        
        # Calculate distance between end-effector and object
        distance = np.sqrt(sum((np.array(ee_position) - np.array(obj_position))**2))
        
        # Combine all observations
        observation = np.concatenate([
            np.array(joint_states),
            np.array(joint_velocities),
            np.array(ee_position),
            np.array(obj_position),
            np.array(obj_orientation),
            np.array([distance])
        ]).astype(np.float32)
        
        return observation

    def _compute_reward(self, observation):
        """
        Compute the reward based on the current state.
        The reward is a combination of:
        1. Closeness of end-effector to object (for reaching)
        2. Closeness of object to target (for pushing)
        """
        # Extract positions from observation
        ee_position = observation[2*self.num_control_joints:2*self.num_control_joints+3]
        obj_position = observation[2*self.num_control_joints+3:2*self.num_control_joints+6]
        # Object orientation is at observation[2*self.num_control_joints+6:2*self.num_control_joints+10]
        distance_ee_to_obj = observation[-1]
        
        # Define target position (could be moved to config)
        target_position = np.array([1.5, 0, 0.5])
        
        # Calculate distance from object to target
        distance_obj_to_target = np.sqrt(sum((np.array(obj_position) - target_position)**2))
        
        # Rewards
        # 1. Negative distance from end-effector to object (closer is better)
        reaching_reward = -distance_ee_to_obj
        
        # 2. Negative distance from object to target (closer is better)
        pushing_reward = -distance_obj_to_target
        
        # 3. Bonus for getting close to object
        close_bonus = 0.0
        if distance_ee_to_obj < 0.05:
            close_bonus = 1.0
            
        # 4. Success bonus if object is very close to target
        success_bonus = 0.0
        if distance_obj_to_target < 0.1:
            success_bonus = 10.0
            
        # Combine rewards - initially emphasize reaching, then pushing
        # This creates a natural curriculum: first learn to reach, then to push
        reward = reaching_reward + 2.0 * pushing_reward + close_bonus + success_bonus
        
        return float(reward)  # Convert to Python float to avoid NumPy type issues

    def _is_terminated(self, observation):
        """Check if the episode should terminate."""
        # Extract object position from observation
        obj_position = observation[2*self.num_control_joints+3:2*self.num_control_joints+6]
        # Object orientation is at observation[2*self.num_control_joints+6:2*self.num_control_joints+10]
        
        # Define a target position for the cube (could be moved to config)
        target_position = np.array([1.5, 0, 0.5])
        
        # Calculate distance to target
        distance_to_target = np.sqrt(sum((np.array(obj_position) - target_position)**2))
        
        # Episode terminates successfully if cube is close to target
        # Convert to Python boolean to avoid NumPy type issues
        return bool(distance_to_target < 0.1)  # Done if cube is near target

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            # Already rendering in GUI mode
            # Optionally add a slight delay to make it more viewable
            time.sleep(0.01)
            return None
        elif self.render_mode == "rgb_array":
            # Render to RGB array
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[1.0, -1.0, 1.0],
                cameraTargetPosition=[0, 0, 0],
                cameraUpVector=[0, 0, 1]
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
            )
            width, height = 320, 240
            img_arr = p.getCameraImage(
                width, height, view_matrix, proj_matrix, 
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            rgb = img_arr[2]
            return rgb
        else:
            return None

    def close(self):
        """Close the environment."""
        if self.client >= 0:
            p.disconnect(self.client)
            self.client = -1