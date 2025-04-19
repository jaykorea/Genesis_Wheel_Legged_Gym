from wheeled_legged_gym.envs.base.wheeled_legged_robot_config import WheeledLeggedRobotCfg, WheeledLeggedRobotCfgPPO

class FlamingoCfg( WheeledLeggedRobotCfg ):
    class sim:
        precision = "32" # Currently "64" precision not working for the rsl_rl lib
        gravity = [0.0, 0.0 , -9.81]  # [m/s^2]

    class env( WheeledLeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 28  # 6 + 8 + 3 + 3 + 8 = 28
        num_privileged_obs = 31 # 6 + 8 + 3 + 3 + 8 + 3 = 31
        num_stacks = 3
        num_actions = 8
        debug = False # if debugging, visualize contacts, 
        debug_viz = False # draw debug visualizations
    
    class terrain( WheeledLeggedRobotCfg.terrain ):
        mesh_type = "plane" # none, plane, heightfield
        friction = 1.0
        restitution = 0.0
        
    class init_state( WheeledLeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.38288] # x,y,z [m]
        rot = [1.0, 0.0, 0.0, 0.0] # w, x, y, z [quat]
        class ranges( WheeledLeggedRobotCfg.init_state.ranges):
            roll = [-0.0, 0.0]
            pitch = [-0.0, 0.0]
            yaw = [-3.14, 3.14]
            
            lin_vel_x = [-0.0, 0.0] # x,y,z [m/s]
            lin_vel_y = [-0.0, 0.0]   # x,y,z [m/s]
            lin_vel_z = [-0.0, 0.0]   # x,y,z [m/s]

            ang_vel_x = [-0.0, 0.0]   # x,y,z [rad/s]
            ang_vel_y = [-0.0, 0.0]   # x,y,z [rad/s]
            ang_vel_z = [-0.0, 0.0]   # x,y,z [rad/s]

        randomize_joint = {
            "left_hip_joint": False,
            "right_hip_joint": False,
            "left_shoulder_joint": False,
            "right_shoulder_joint": False,
            "left_leg_joint": False,
            "right_leg_joint": False,
            "left_wheel_joint": False,
            "right_wheel_joint": False,
            "range": [-0.1, 0.1],
        }
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_joint': 0.0,   # [rad]
            'right_hip_joint': 0.0,   # [rad]
            'left_shoulder_joint': 0.0,   # [rad]
            'right_shoulder_joint': 0.0,   # [rad]
            'left_leg_joint': 0.0,   # [rad]
            'right_leg_joint': 0.0,   # [rad]
            'left_wheel_joint': 0.0 ,  # [rad]
            'right_wheel_joint': 0.0,   # [rad]
        }
        default_joint_velocities = { # = target velocities [rad/s] when action = 0.0
            'left_hip_joint': 0.0,   # [rad/s]
            'right_hip_joint': 0.0,   # [rad/s]
            'left_shoulder_joint': 0.0,   # [rad/s]
            'right_shoulder_joint': 0.0,   # [rad/s]
            'left_leg_joint': 0.0,   # [rad/s]
            'right_leg_joint': 0.0,   # [rad/s]
            'left_wheel_joint': 0.0 ,  # [rad/s]
            'right_wheel_joint': 0.0,   # [rad/s]
        }

    class control( WheeledLeggedRobotCfg.control ):
        use_implicit_controller = True
        dt = 0.02 # sim frequency 50Hz
        decimation = 4
        class control_p:
            stiffness = {
                'left_hip_joint':100.0      , 'right_hip_joint': 100.0,
                'left_shoulder_joint': 100.0 , 'right_shoulder_joint': 100.0,
                'left_leg_joint': 150.0      , 'right_leg_joint': 150.0,
                } # [N*m/rad]
            damping = {
                'left_hip_joint': 1.5       , 'right_hip_joint': 1.5,
                'left_shoulder_joint': 1.5  , 'right_shoulder_joint': 1.5,
                'left_leg_joint': 2.0       , 'right_leg_joint': 2.0,
                } # [N*m*s/rad]
            torque_limit =  {
                'left_hip_joint': 60.0      , 'right_hip_joint': 60.0,
                'left_shoulder_joint': 60.0 , 'right_shoulder_joint': 60.0,
                'left_leg_joint': 60.0      , 'right_leg_joint': 60.0,
                } # [N*m]
            vel_limit = {
                'left_hip_joint': 20.0      , 'right_hip_joint': 20.0,
                'left_shoulder_joint': 20.0 , 'right_shoulder_joint': 20.0,
                'left_leg_joint': 20.0      , 'right_leg_joint': 20.0,
                } # [rad/s]
            action_scale = 2.0

        class control_v:
            stiffness =     {'left_wheel_joint': 0.0,  'right_wheel_joint': 0.0}  # [N*m/rad]
            damping =       {'left_wheel_joint': 0.5, 'right_wheel_joint': 0.5}     # [N*m*s/rad]
            torque_limit =  {'left_wheel_joint': 60.0, 'right_wheel_joint': 60.0} # [N*m]
            vel_limit =     {'left_wheel_joint': 20.0, 'right_wheel_joint': 20.0}
            action_scale = 20.0

    class asset( WheeledLeggedRobotCfg.asset ):
        asset_type = "mjcf"
        convexify = True
        if asset_type == "urdf":
            file = '{WHEELED_LEGGED_GYM_ROOT_DIR}/resources/robots/flamingo/urdf/flamingo_v1_5_1.urdf'
        elif asset_type == "mjcf":
            file = '{WHEELED_LEGGED_GYM_ROOT_DIR}/resources/robots/flamingo/xml/flamingo_v1_5_1.xml'

        base_dof_names = ["base_free_joint"]
        dof_names = [ # specify the sequence of actions
            'left_hip_joint'      , 'right_hip_joint',
            'left_shoulder_joint' , 'right_shoulder_joint',
            'left_leg_joint'      , 'right_leg_joint',
            'left_wheel_joint'    , 'right_wheel_joint'
        ]

        foot_name = ["wheel"]
        penalize_contacts_on = ["base", "shoulder", "leg"]
        terminate_after_contacts_on = ["base", "shoulder", "leg"]
        links_to_keep = []
        self_collisions = False
  
    class rewards( WheeledLeggedRobotCfg.rewards ):
        only_positive_rewards = False
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.8
        base_height_target = 0.36288 # 0.36288
        class scales( WheeledLeggedRobotCfg.rewards.scales ):
            # command tracking
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.5
            # limitation
            torque_limits = -0.01
            dof_pos_limits = -10.0
            dof_vel_limits = -0.0
            collision = -1.0
            shoulder_align = -1.0
            hip_deviations = -5.0
            shoulder_deviations = -1.0
            
            # smooth
            dof_vel = 0.0
            dof_acc = -0.0 # -2.5e-7
            action_rate = -0.01
            torques = -5.0e-5

            # balance
            ang_vel_xy = -0.05
            lin_vel_z = -0.5
            base_height = -5.0
            orientation = -2.0
            dof_close_to_default = -0.0
            # gait
            feet_air_time = 0.0

            # Termination
            termination = -200.0
        
    class commands( WheeledLeggedRobotCfg.commands ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 3 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges( WheeledLeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-0.0, 0.0] # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0.0, 0.0]    # min max [rad/s]
            heading = [0.0, 0.0]
    

    class domain_rand:   
        randomize_friction = True
        friction_range = [0.3, 1.5]

        randomize_base_mass = True
        added_mass_range = [-1.0, 2.0]

        randomize_com_displacement = True
        com_displacement_range = [-0.002, 0.002]

        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0

        simulate_action_latency = True # 1 step delay

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        #* It should be matched with the single observation buffer size
        class noise_scales:
            dof_pos = 0.04
            dof_vel = 1.5
            ang_vel = 0.25
            gravity = 0.05
            actions = 0.0
            commands = 0.0

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]       # [m]
        lookat = [11., 5, 3.]  # [m]
        num_rendered_envs = 12  # number of environments to be rendered
        add_camera = False

class FlamingoCfgPPO( WheeledLeggedRobotCfgPPO ):
    run_name = 'stand'
    experiment_name = 'flamingo_v1_5_1'
    save_interval = 100
    load_run = -1
    checkpoint = -1
    max_iterations = 1000