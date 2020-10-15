from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
import cv2
import open3d as o3d
import numpy as np
from matplotlib import cm
import time
import carla
import pygame

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LABEL_COLORS = np.array([
    (255, 255, 255), # None
    (70, 70, 70),    # Building
    (100, 40, 40),   # Fences
    (55, 90, 80),    # Other
    (220, 20, 60),   # Pedestrian
    (153, 153, 153), # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),     # Vehicle
    (102, 102, 156), # Wall
    (220, 220, 0),   # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),   # Water
    (145, 170, 100), # Terrain
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses

def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)

def get_entry_point():
    return 'TestAgent'


class KeyboardControl(object):

    """
    Keyboard control for the human agent
    """

    def __init__(self, path_to_conf_file):
        """
        Init
        """
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        self._clock = pygame.time.Clock()

        # Get the mode
        if path_to_conf_file:

            with (open(path_to_conf_file, "r")) as f:
                lines = f.read().split("\n")
                self._mode = lines[0].split(" ")[1]
                self._endpoint = lines[1].split(" ")[1]

            # Get the needed vars
            if self._mode == "log":
                self._log_data = {'records': []}

            elif self._mode == "playback":
                self._index = 0
                self._control_list = []

                with open(self._endpoint) as fd:
                    try:
                        self._records = json.load(fd)
                        self._json_to_control()
                    except json.JSONDecodeError:
                        pass
        else:
            self._mode = "normal"
            self._endpoint = None

    def _json_to_control(self):

        # transform strs into VehicleControl commands
        for entry in self._records['records']:
            control = carla.VehicleControl(throttle=entry['control']['throttle'],
                                           steer=entry['control']['steer'],
                                           brake=entry['control']['brake'],
                                           hand_brake=entry['control']['hand_brake'],
                                           reverse=entry['control']['reverse'],
                                           manual_gear_shift=entry['control']['manual_gear_shift'],
                                           gear=entry['control']['gear'])
            self._control_list.append(control)

    def parse_events(self, timestamp):
        """
        Parse the keyboard events and set the vehicle controls accordingly
        """
        # Move the vehicle
        if self._mode == "playback":
            self._parse_json_control()
        else:
            self._parse_vehicle_keys(pygame.key.get_pressed(), timestamp*1000)

        # Record the control
        if self._mode == "log":
            self._record_control()

        return self._control

    def _parse_vehicle_keys(self, keys, milliseconds):
        """
        Calculate new vehicle controls based on input keys
        """

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 
            elif event.type == pygame.KEYUP:
                if event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                    self._control.reverse = self._control.gear < 0

        if keys[K_UP] or keys[K_w]:
            self._control.throttle = 0.6
        else:
            self._control.throttle = 0.0

        steer_increment = 3e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        steer_cache = min(0.95, max(-0.95, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_json_control(self):

        if self._index < len(self._control_list):
            self._control = self._control_list[self._index]
            self._index += 1
        else:
            print("JSON file has no more entries")

    def _record_control(self):
        new_record = {
            'control': {
                'throttle': self._control.throttle,
                'steer': self._control.steer,
                'brake': self._control.brake,
                'hand_brake': self._control.hand_brake,
                'reverse': self._control.reverse,
                'manual_gear_shift': self._control.manual_gear_shift,
                'gear': self._control.gear
            }
        }

        self._log_data['records'].append(new_record)

    def __del__(self):
        # Get ready to log user commands
        if self._mode == "log" and self._log_data:
            with open(self._endpoint, 'w') as fd:
                json.dump(self._log_data, fd, indent=4, sort_keys=True)

class CameraInterface(object):
    """
    Class to control a vehicle manually for debugging purposes
    """
    def __init__(self):
        self._width = 1600
        self._height = 1200
        self._surface = None

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode((self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Human Agent")

    def run_interface(self, input_data):
        """
        Run the GUI
        """

        # process sensor data
        image_front = input_data['Center'][1][:, :, 0:3]
        image_back = input_data["Back"][1][:, :, 0:3]
        image_right = input_data["Right"][1][:, :, 0:3]
        image_left = input_data["Left"][1][:, :, 0:3]
        
        image_all = np.concatenate(
            [
                np.concatenate([image_front, image_back], axis=1),
                np.concatenate([image_right, image_left], axis=1)
            ],
            axis=0
        )
        # cv2.imshow("all image", image_all)
        # cv2.waitKey(1)

        # display image
        image_all = image_all[:, :, -1::-1]
        self._surface = pygame.surfarray.make_surface(image_all.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

    def _quit(self):
        pygame.quit()

class LidarInterface(object):
    def __init__(self):
        self.point_list = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name='Carla Lidar',
            width=800,
            height=600,
            left=480,
            top=270)
        self.vis.capture_depth_float_buffer(do_render=True)
        self.is_add_pointcloud = False
        self.pre_lidar_points = None
    
    def run_interface(self, input_data):
        lidar_data = input_data["LIDAR"][1]
        
        if self.pre_lidar_points is not None:
            lidar_data = np.concatenate([self.pre_lidar_points, lidar_data], axis=0)
        self.pre_lidar_points = input_data["LIDAR"][1]
        
        # Isolate the intensity and compute a color for it
        intensity = lidar_data[:, -1]
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        int_color = np.c_[
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])
        ]
        points = lidar_data[:, :-1]
        self.point_list.points = o3d.utility.Vector3dVector(points)
        # self.point_list.colors = o3d.utility.Vector3dVector(int_color)

        self.vis.get_render_option().background_color = [1, 1, 1]
        self.vis.get_render_option().point_size = 2.0
        self.vis.get_render_option().show_coordinate_frame = False
        add_open3d_axis(self.vis)
        if not self.is_add_pointcloud:
            self.vis.add_geometry(self.point_list)
            self.is_add_pointcloud = True
        self.vis.update_geometry(self.point_list)
        self.vis.poll_events()
        self.vis.update_renderer()
        time.sleep(0.001)


class TestAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS # At a minimum, this method sets the Leaderboard modality. In this case, SENSORS
        self._clock = pygame.time.Clock()

        self._camera_interface = CameraInterface()
        self._lidar_interface = LidarInterface()
        self._controller = KeyboardControl(path_to_conf_file)
        self._prev_timestamp = 0

    def sensors(self):
        sensors = [
                    {'type': 'sensor.camera.rgb', 'x': 1.2, 'y': 0.0, 'z': 2.60, 'roll': 0.0, 'pitch': -30.0, 'yaw': 0.0,
                    'width': 800, 'height': 600, 'fov': 100, 'id': 'Center'},
                    {'type': 'sensor.camera.rgb', 'x': -1.2, 'y': 0.0, 'z': 2.60, 'roll': 0.0, 'pitch': -30.0, 'yaw': 180.0,
                    'width': 800, 'height': 600, 'fov': 100, 'id': 'Back'},
                    {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 1.0, 'z': 2.60, 'roll': 0.0, 'pitch': -30.0, 'yaw': 90.0,
                    'width': 800, 'height': 600, 'fov': 100, 'id': 'Right'},
                    {'type': 'sensor.camera.rgb', 'x': 0.0, 'y': -1.0, 'z': 2.60, 'roll': 0.0, 'pitch': -30.0, 'yaw': -90.0,
                    'width': 800, 'height': 600, 'fov': 100, 'id': 'Left'},
                    {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0,
                    'yaw': -45.0, 'id': 'LIDAR'},
                    # {'type': 'sensor.other.radar', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0,
                    # 'yaw': -45.0, 'fov': 30, 'id': 'RADAR'},
                    # {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'},
                    # {'type': 'sensor.other.imu', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0,
                    # 'yaw': -45.0, 'id': 'IMU'},
                    # # {'type': 'sensor.opendrive_map', 'reading_frequency': 1, 'id': 'OpenDRIVE'},
                    # {'type': 'sensor.speedometer',  'reading_frequency': 20, 'id': 'SPEED'},
                ]
        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """

        self._camera_interface.run_interface(input_data)
        self._lidar_interface.run_interface(input_data)
        control = self._controller.parse_events(timestamp - self._prev_timestamp)
        
        return control
        # time.sleep(0.04)
        # self.agent_engaged = True
        # self._hic.run_interface(input_data)

        # control = self._controller.parse_events(timestamp - self._prev_timestamp)
        # self._prev_timestamp = timestamp

        # return control