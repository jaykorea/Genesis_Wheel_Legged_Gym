from wheeled_legged_gym.envs.base.wheeled_legged_robot_config import WheeledLeggedRobotCfg, WheeledLeggedRobotCfgPPO

class GO2Cfg( WheeledLeggedRobotCfg ):
    
    class env( WheeledLeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 42
        num_privileged_obs = 45
        num_stacks = 3
        num_actions = 12

    class terrain( WheeledLeggedRobotCfg.terrain ):
        mesh_type = "plane" # none, plane, heightfield
        friction = 1.0
        restitution = 0.
        
    class init_state( WheeledLeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
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
            'FL_hip_joint': False,   
            'RL_hip_joint': False,   
            'FR_hip_joint': False ,  
            'RR_hip_joint': False,   

            'FL_thigh_joint': False,
            'RL_thigh_joint': False, 
            'FR_thigh_joint': False,  
            'RR_thigh_joint': False, 

            'FL_calf_joint': False,  
            'RL_calf_joint': False,   
            'FR_calf_joint': False, 
            'RR_calf_joint': False,    
            "range": [-0.1, 0.1],
        }
        default_joint_angles = {
            'FL_hip_joint': 0.1,     # [rad]
            'RL_hip_joint': 0.1,     # [rad]
            'FR_hip_joint': -0.1,    # [rad]
            'RR_hip_joint': -0.1,    # [rad]

            'FL_thigh_joint': 0.8,   # [rad]
            'RL_thigh_joint': 1.0,   # [rad]
            'FR_thigh_joint': 0.8,   # [rad]
            'RR_thigh_joint': 1.0,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,   # [rad]
            'FR_calf_joint': -1.5,   # [rad]
            'RR_calf_joint': -1.5,   # [rad]
        }

        default_joint_velocities = { # = target angles [rad]/s when action = 0.0
            'FL_hip_joint': 0.0,     # [rad]/s
            'RL_hip_joint': 0.0,     # [rad]/s
            'FR_hip_joint': 0.0,    # [rad]/s
            'RR_hip_joint': 0.0,     # [rad]/s

            'FL_thigh_joint': 0.0,   # [rad]/s
            'RL_thigh_joint': 0.0,   # [rad]/s
            'FR_thigh_joint': 0.0,   # [rad]/s
            'RR_thigh_joint': 0.0,   # [rad]/s

            'FL_calf_joint': 0.0,    # [rad]/s
            'RL_calf_joint': 0.0,    # [rad]/s
            'FR_calf_joint': 0.0,    # [rad]/s
            'RR_calf_joint': 0.0,    # [rad]/s
        }

    class control( WheeledLeggedRobotCfg.control ):
        dt =  0.02  # control frequency 50Hz
        decimation = 4 # decimation: Number of control action updates @ sim DT per policy DT
        class control_p:
            stiffness = { # = target angles [rad] when action = 0.0
                'FL_hip_joint': 20.0,     # [rad]
                'RL_hip_joint': 20.0,     # [rad]
                'FR_hip_joint': 20.0,   # [rad]
                'RR_hip_joint': 20.0,    # [rad]

                'FL_thigh_joint': 20.0,   # [rad]
                'RL_thigh_joint': 20.0,    # [rad]
                'FR_thigh_joint': 20.0,   # [rad]
                'RR_thigh_joint': 20.0,    # [rad]

                'FL_calf_joint': 20.0,   # [rad]
                'RL_calf_joint': 20.0,   # [rad]
                'FR_calf_joint': 20.0,   # [rad]
                'RR_calf_joint': 20.0,   # [rad]
            }
            damping = { # = target angles [rad] when action = 0.0
                'FL_hip_joint': 0.5,  
                'RL_hip_joint': 0.5,  
                'FR_hip_joint': 0.5,  
                'RR_hip_joint': 0.5,   
                'FL_thigh_joint': 0.5,
                'RL_thigh_joint': 0.5,
                'FR_thigh_joint': 0.5,
                'RR_thigh_joint': 0.5,
                'FL_calf_joint': 0.5,
                'RL_calf_joint': 0.5,
                'FR_calf_joint': 0.5,
                'RR_calf_joint': 0.5,   
            }
            torque_limit =  { # = target angles [rad] when action = 0.0
                'FL_hip_joint': 23.0,  
                'RL_hip_joint': 23.0,  
                'FR_hip_joint': 23.0,  
                'RR_hip_joint': 23.0,   
                'FL_thigh_joint': 23.0,
                'RL_thigh_joint': 23.0,
                'FR_thigh_joint': 23.0,
                'RR_thigh_joint': 23.0,
                'FL_calf_joint': 23.0,
                'RL_calf_joint': 23.0,
                'FR_calf_joint': 23.0,
                'RR_calf_joint': 23.0,   
            }
            vel_limit = { # = target angles [rad] when action = 0.0
                'FL_hip_joint': 30.0,  
                'RL_hip_joint': 30.0,  
                'FR_hip_joint': 30.0,  
                'RR_hip_joint': 30.0,   
                'FL_thigh_joint': 30.0,
                'RL_thigh_joint': 30.0,
                'FR_thigh_joint': 30.0,
                'RR_thigh_joint': 30.0,
                'FL_calf_joint': 30.0,
                'RL_calf_joint': 30.0,
                'FR_calf_joint': 30.0,
                'RR_calf_joint': 30.0,   
            }
            action_scale = 0.25

    class asset( WheeledLeggedRobotCfg.asset ):
        file = '{WHEELED_LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        dof_names = [
            'FR_hip_joint',
            'FR_thigh_joint',
            'FR_calf_joint',
            'FL_hip_joint',
            'FL_thigh_joint',
            'FL_calf_joint',
            'RR_hip_joint',
            'RR_thigh_joint',
            'RR_calf_joint',
            'RL_hip_joint',
            'RL_thigh_joint',
            'RL_calf_joint',]
        foot_name = ["foot"]
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        links_to_keep = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        self_collisions = False
  
    class rewards( WheeledLeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.36
        class scales( WheeledLeggedRobotCfg.rewards.scales ):
            # limitation
            dof_pos_limits = -10.0
            collision = -1.0
            # command tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # smooth
            lin_vel_z = -2.0
            base_height = -1.0
            ang_vel_xy = -0.05
            orientation = -3.0
            dof_vel = -5.e-4
            dof_acc = -2.e-7
            action_rate = -0.01
            torques = -2.e-4
            # gait
            feet_air_time = 1.0
            # dof_close_to_default = -0.1
    
    class commands( WheeledLeggedRobotCfg.commands ):
        curriculum = True
        max_curriculum = 1.
        num_commands = 3 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges( WheeledLeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [0, 0]
    
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

        simulate_action_latency = True # 1 step delay
    
    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]       # [m]
        lookat = [11., 5, 3.]  # [m]
        num_rendered_envs = 10  # number of environments to be rendered
        add_camera = False

class GO2CfgPPO( WheeledLeggedRobotCfgPPO ):
    run_name = 'walk'
    experiment_name = 'go2'
    save_freq = 100
    load_run = -1
    checkpoint = -1
    max_iterations = 600

    # class algorithm( WheeledLeggedRobotCfgPPO.algorithm ):
    #     entropy_coef = 0.01
