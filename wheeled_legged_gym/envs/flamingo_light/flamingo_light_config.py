from wheeled_legged_gym.envs.base.wheeled_legged_robot_config import WheeledLeggedRobotCfg, WheeledLeggedRobotCfgPPO

class FlamingoLightCfg( WheeledLeggedRobotCfg ):
    class sim:
        precision = "32"
        gravity = [0.0, 0.0 , -9.81]  # [m/s^2]

    class env( WheeledLeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 16
        num_privileged_obs = 19
        num_stacks = 3
        num_actions = 4
        debug = False # if debugging, visualize contacts, 
        debug_viz = False # draw debug visualizations
    
    class terrain( WheeledLeggedRobotCfg.terrain ):
        mesh_type = "plane" # none, plane, heightfield
        friction = 1.0
        restitution = 0.0
        
    class init_state( WheeledLeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.2707] # x,y,z [m]
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
            "left_shoulder_joint": True,
            "right_shoulder_joint": True,
            "left_wheel_joint": False,
            "right_wheel_joint": False,
            "range": [-0.1, 0.1],
        }
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_shoulder_joint': 0.0,   # [rad]
            'right_shoulder_joint': 0.0,   # [rad]
            'left_wheel_joint': 0.0 ,  # [rad]
            'right_wheel_joint': 0.0,   # [rad]
        }
        default_joint_velocities = { # = target velocities [rad/s] when action = 0.0
            'left_shoulder_joint': 0.0,   # [rad/s]
            'right_shoulder_joint': 0.0,   # [rad/s]
            'left_wheel_joint': 0.0 ,  # [rad/s]
            'right_wheel_joint': 0.0,   # [rad/s]
        }

    class control( WheeledLeggedRobotCfg.control ):
        use_implicit_controller = True
        dt = 0.02 # sim frequency 50Hz
        decimation = 4
        class control_p:
            stiffness = {'left_shoulder_joint': 30.0, 'right_shoulder_joint': 30.0}  # [N*m/rad]
            damping = {'left_shoulder_joint': 1.5, 'right_shoulder_joint': 1.5}     # [N*m*s/rad]
            torque_limit =  {'left_shoulder_joint': 12.0, 'right_shoulder_joint': 12.0} # [N*m]
            vel_limit = {'left_shoulder_joint': 30.0, 'right_shoulder_joint': 30.0} # [rad/s]
            action_scale = 1.25

        class control_v:
            stiffness = {'left_wheel_joint': 0., 'right_wheel_joint': 0.}  # [N*m/rad]
            damping = {'left_wheel_joint': 0.25, 'right_wheel_joint': 0.25}     # [N*m*s/rad]
            torque_limit =  {'left_wheel_joint': 12.0, 'right_wheel_joint': 12.0} # [N*m]
            vel_limit = {'left_wheel_joint': 30.0, 'right_wheel_joint': 30.0}
            action_scale = 30.0

    class asset( WheeledLeggedRobotCfg.asset ):
        asset_type = "mjcf"
        convexify = True
        file = '{WHEELED_LEGGED_GYM_ROOT_DIR}/resources/robots/flamingo_light/xml/flamingo_light_v1.xml'
        base_dof_names = ["base_free_joint"]
        all_dof_names = [
            'left_shoulder_joint',
            'right_shoulder_joint',
            'left_leg_joint',
            'right_leg_joint',
            'left_upper_joint',
            'right_upper_joint',
            'left_wheel_joint',
            'right_wheel_joint',
        ]
        dof_names = [# specify yhe sequence of actions
            'left_shoulder_joint',
            'right_shoulder_joint',
            'left_wheel_joint',
            'right_wheel_joint',]

        foot_name = ["wheel"]
        penalize_contacts_on = ["base", "leg"]
        terminate_after_contacts_on = ["base", "leg"]
        links_to_keep = []
        self_collisions = False
  
    class rewards( WheeledLeggedRobotCfg.rewards ):
        only_positive_rewards = False
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        base_height_target = 0.41652 # 0.41652
        class scales( WheeledLeggedRobotCfg.rewards.scales ):
            # command tracking
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            # limitation
            dof_pos_limits = -10.0
            dof_vel_limits = -1.0
            collision = -1.0
            # smooth
            ang_vel_xy = -0.075
            lin_vel_z = -1.75
            base_height = -20.0
            orientation = -2.0
            dof_vel = 0.0
            dof_acc = -2.0e-7
            action_rate = -0.01
            torques = -5.0e-4
            # gait
            feet_air_time = 0.0
            dof_close_to_default = -0.0
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
        friction_range = [0.3, 1.2]

        randomize_base_mass = True
        added_mass_range = [-0.5, 1.0]

        randomize_com_displacement = True
        com_displacement_range = [-0.002, 0.002]

        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 0.5

        simulate_action_latency = False # 1 step delay


    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        #* It should be matched with the observation buffer size
        class noise_scales:
            dof_pos = 0.04
            dof_vel = 1.5
            ang_vel = 0.2
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

class FlamingoLightCfgPPO( WheeledLeggedRobotCfgPPO ):
    run_name = 'stand'
    experiment_name = 'flamingo_light_v1'
    save_interval = 100
    load_run = -1
    checkpoint = -1
    max_iterations = 1000

    # class algorithm( LeggedRobotCfgPPO.algorithm ):
    #     entropy_coef = 0.01
