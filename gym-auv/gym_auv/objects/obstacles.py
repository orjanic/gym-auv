from xmlrpc.client import Boolean
import numpy as np
import shapely.geometry
import shapely.affinity

import gym_auv.utils.constants as const
import gym_auv.utils.geomutils as geom
# from gym_auv.objects.vessel import _odesolver45
from abc import ABC, abstractmethod
import copy

def _odesolver45(f, y, h): #(copied from vessel.py, import gave circular dependencies)
    """Calculate the next step of an IVP of a time-invariant ODE with a RHS
    described by f, with an order 4 approx. and an order 5 approx.
    Parameters:
        f: function. RHS of ODE.
        y: float. Current position.
        h: float. Step length.
    Returns:
        q: float. Order 2 approx.
        w: float. Order 3 approx.
    """
    s1 = f(y)
    s2 = f(y+h*s1/4.0)
    s3 = f(y+3.0*h*s1/32.0+9.0*h*s2/32.0)
    s4 = f(y+1932.0*h*s1/2197.0-7200.0*h*s2/2197.0+7296.0*h*s3/2197.0)
    s5 = f(y+439.0*h*s1/216.0-8.0*h*s2+3680.0*h*s3/513.0-845.0*h*s4/4104.0)
    s6 = f(y-8.0*h*s1/27.0+2*h*s2-3544.0*h*s3/2565+1859.0*h*s4/4104.0-11.0*h*s5/40.0)
    w = y + h*(25.0*s1/216.0+1408.0*s3/2565.0+2197.0*s4/4104.0-s5/5.0)
    q = y + h*(16.0*s1/135.0+6656.0*s3/12825.0+28561.0*s4/56430.0-9.0*s5/50.0+2.0*s6/55.0)
    return w, q

class BaseObstacle(ABC):
    def __init__(self, *args, **kwargs) -> None:
        """Initializes obstacle instance by calling private setup method implemented by
         subclasses of BaseObstacle and calculating obstacle boundary."""
        self._prev_position = []
        self._prev_heading = []
        self._setup(*args, **kwargs)
        self._boundary = self._calculate_boundary()
        if not self._boundary.is_valid:
            self._boundary = self._boundary.buffer(0)
        self._init_boundary = copy.deepcopy(self._boundary)


    @property
    def boundary(self) -> shapely.geometry.Polygon:
        """shapely.geometry.Polygon object used for simulating the 
        sensors' detection of the obstacle instance."""
        return self._boundary

    @property
    def init_boundary(self) -> shapely.geometry.Polygon:
        """shapely.geometry.Polygon object used for simulating the 
        sensors' detection of the obstacle instance."""
        return self._init_boundary

    def update(self, dt:float) -> None:
        """Updates the obstacle according to its dynamic behavior, e.g. 
        a ship model and recalculates the boundary."""
        has_changed = self._update(dt)
        if has_changed:
            self._boundary = self._calculate_boundary()
            if not self._boundary.is_valid:
                self._boundary = self._boundary.buffer(0)

    @abstractmethod
    def _calculate_boundary(self) -> shapely.geometry.Polygon:
        """Returns a shapely.geometry.Polygon instance representing the obstacle
        given its current state."""

    @abstractmethod
    def _setup(self, *args, **kwargs) -> None:
        """Initializes the obstacle given the constructor parameters provided to
        the specific BaseObstacle extension."""

    def _update(self, _dt:float) -> bool:
        """Performs the specific update routine associated with the obstacle.
        Returns a boolean flag representing whether something changed or not.

        Returns
        -------
        has_changed : bool
        """
        return False

    @property
    def path_taken(self) -> list:
        """Returns an array holding the path of the obstacle in cartesian
        coordinates."""
        return self._prev_position

    @property
    def heading_taken(self) -> list:
        """Returns an array holding the heading of the obstacle at previous timesteps."""
        return self._prev_heading

class CircularObstacle(BaseObstacle):
    def _setup(self, position, radius, color=(0.6, 0, 0)):
        self.color = color
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        if radius < 0:
            raise ValueError
        self.static = True
        self.radius = radius
        self.position = position.flatten()

    def _calculate_boundary(self):
        return shapely.geometry.Point(*self.position).buffer(self.radius).boundary.simplify(0.3, preserve_topology=False)

class PolygonObstacle(BaseObstacle):
    def _setup(self, points, color=(0.6, 0, 0)):
        self.static = True
        self.color = color
        self.points = points

    def _calculate_boundary(self):
        return shapely.geometry.Polygon(self.points)

class LineObstacle(BaseObstacle):
    def _setup(self, points):
        self.static = True
        self.points = points

    def _calculate_boundary(self):
        return shapely.geometry.LineString(self.points)

class VesselObstacle(BaseObstacle):
    def _setup(self, width, trajectory, init_position=None, init_heading=None, init_update=True, name=''):
        self.static = False
        self.width = width
        self.trajectory = trajectory
        self.trajectory_velocities = []
        self.name = name
        i = 0
        while i < len(trajectory)-1:
            cur_t = trajectory[i][0]
            next_t = trajectory[i+1][0]
            cur_waypoint = trajectory[i][1]
            next_waypoint = trajectory[i+1][1]

            dx = (next_waypoint[0] - cur_waypoint[0])/(next_t - cur_t)
            dy = (next_waypoint[1] - cur_waypoint[1])/(next_t - cur_t)

            for _ in range(cur_t, next_t):
                self.trajectory_velocities.append((dx, dy))
            
            i += 1

        self.waypoint_counter = 0
        self.points = [
            (-self.width/2, -self.width/2),
            (-self.width/2, self.width/2),
            (self.width/2, self.width/2),
            (3/2*self.width, 0),
            (self.width/2, -self.width/2),
        ]
        if init_position is not None:
            self.position = init_position
        else:
            self.position = np.array(self.trajectory[0][1])
        self.init_position = self.position.copy() 
        if init_heading is not None:
            self.dx0 = self.trajectory_velocities[0][0]  # THOMAS
            self.dy0 = self.trajectory_velocities[0][1]  # THOMAS
            self.heading = init_heading
            self.init_heading = init_heading
        else:
            self.dx0 = self.trajectory_velocities[0][0]
            self.dy0 = self.trajectory_velocities[0][1]
            self.heading = geom.princip(np.arctan2(self.dy0, self.dx0))
            self.init_heading = geom.princip(np.arctan2(self.dy0, self.dx0))
            #self.heading = np.pi/2  # THOMAS 06.08.21 -- FIX VESSEL HEADINGS ON TRAJECORY PLOTS

        if init_update:
            self.update(dt=0.1)

    def _update(self, dt):
        self.waypoint_counter += dt

        index = int(np.floor(self.waypoint_counter))

        if index >= len(self.trajectory_velocities) - 1:
            self.waypoint_counter = 0
            index = 0
            self.position = np.array(self.trajectory[0][1])

        dx = self.trajectory_velocities[index][0]
        dy = self.trajectory_velocities[index][1]

        self.dx = dt*dx
        self.dy = dt*dy
        self.heading = geom.princip(np.arctan2(self.dy, self.dx))
        self.position = self.position + np.array([self.dx, self.dy])
        self._prev_position.append(self.position)
        self._prev_heading.append(self.heading)
        self._step_counter = 0

        return True

    def _calculate_boundary(self):
        ship_angle = self.heading# float(geom.princip(self.heading))

        boundary_temp = shapely.geometry.Polygon(self.points)
        boundary_temp = shapely.affinity.rotate(boundary_temp, ship_angle, use_radians=True, origin='centroid')
        boundary_temp = shapely.affinity.translate(boundary_temp, xoff=self.position[0], yoff=self.position[1])

        return boundary_temp

class AdversarialVesselObstacle(BaseObstacle):
    def _setup(self, width, trajectory, init_position=None, init_heading=None, init_update=True, name=''):
        self.static = False
        self.width = width
        self.trajectory = trajectory
        self.trajectory_velocities = []
        self.name = name

        for i in range(len(trajectory) - 1):
            cur_t = trajectory[i][0]
            next_t = trajectory[i+1][0]
            cur_waypoint = trajectory[i][1]
            next_waypoint = trajectory[i+1][1]

            dx = (next_waypoint[0] - cur_waypoint[0])/(next_t - cur_t)
            dy = (next_waypoint[1] - cur_waypoint[1])/(next_t - cur_t)

            for _ in range(cur_t, next_t):
                self.trajectory_velocities.append((dx, dy))

        self.waypoint_counter = 0
        self.points = [
            (-self.width/2, -self.width/2),
            (-self.width/2, self.width/2),
            (self.width/2, self.width/2),
            (3/2*self.width, 0),
            (self.width/2, -self.width/2),
        ]
        if init_position is not None:
            self.position = init_position
        else:
            self.position = np.array(self.trajectory[0][1])
        self.init_position = self.position.copy() 
        if init_heading is not None:
            self.dx0 = self.trajectory_velocities[0][0]  # THOMAS
            self.dy0 = self.trajectory_velocities[0][1]  # THOMAS
            self.heading = init_heading
            self.init_heading = init_heading
        else:
            self.dx0 = self.trajectory_velocities[0][0]
            self.dy0 = self.trajectory_velocities[0][1]
            self.heading = geom.princip(np.arctan2(self.dy0, self.dx0))
            self.init_heading = geom.princip(np.arctan2(self.dy0, self.dx0))
            #self.heading = np.pi/2  # THOMAS 06.08.21 -- FIX VESSEL HEADINGS ON TRAJECORY PLOTS

        init_x = trajectory[0][1][0]
        init_y = trajectory[0][1][1]

        init_speed = [0, 0, 0]
        init_state = np.array([init_x, init_y, init_heading], dtype=np.float64)
        init_speed = np.array(init_speed, dtype=np.float64)
        self._state = np.hstack([init_state, init_speed])
        self._prev_states = np.vstack([self._state])
        self._input = [0, 0]
        self._prev_inputs =np.vstack([self._input])

        self._step_counter = 0

        if init_update:
            self.update(action=[0, 0])
    
    def update(self, action:list) -> None:
        """Updates the obstacle according to its dynamic behavior, e.g. 
        a ship model and recalculates the boundary."""
        has_changed = self._update(action)
        if has_changed:
            self._boundary = self._calculate_boundary()
            if not self._boundary.is_valid:
                self._boundary = self._boundary.buffer(0)

    def _update(self, action:list):
        """
        Simulates the adversarial vessel one step forward after applying the given action.

        Parameters
        ----------
        action : np.ndarray[thrust_input, torque_input]
        """
        # self.waypoint_counter += dt

        # index = int(np.floor(self.waypoint_counter))

        # if index >= len(self.trajectory_velocities) - 1:
        #     self.waypoint_counter = 0
        #     index = 0
        #     self.position = np.array(self.trajectory[0][1])

        # dx = self.trajectory_velocities[index][0]
        # dy = self.trajectory_velocities[index][1]

        # self.dx = dt*dx
        # self.dy = dt*dy
        # self.heading = geom.princip(np.arctan2(self.dy, self.dx))
        # self.position = self.position + np.array([self.dx, self.dy])
        # self._prev_position.append(self.position)
        # self._prev_heading.append(self.heading)

        # print(f'\n-------------------\naction: {action}\naction type: {type(action)}\n--------------------\n')
        self._input = np.array([self._thrust_surge(action[0]), self._moment_steer(action[1])])
        w, q = _odesolver45(self._state_dot, self._state, 1.0) # self.config["t_step_size"] = 1.0 in DEFAULT_CONFIG in gym_auv/gym_auv/__init__.py
        
        self.dx = self._state[0] - q[0]
        self.dy = self._state[1] - q[1]

        self._state = q
        self._state[2] = geom.princip(self._state[2])

        self._prev_states = np.vstack([self._prev_states,self._state])
        self._prev_inputs = np.vstack([self._prev_inputs,self._input])

        self._step_counter += 1

        return True

    def _calculate_boundary(self):
        ship_angle = self.heading# float(geom.princip(self.heading))

        boundary_temp = shapely.geometry.Polygon(self.points)
        boundary_temp = shapely.affinity.rotate(boundary_temp, ship_angle, use_radians=True, origin='centroid')
        boundary_temp = shapely.affinity.translate(boundary_temp, xoff=self.position[0], yoff=self.position[1])

        return boundary_temp
    
    def _state_dot(self, state):
        psi = state[2]
        nu = state[3:]

        tau = np.array([self._input[0], 0, self._input[1]])

        eta_dot = geom.Rzyx(0, 0, geom.princip(psi)).dot(nu)
        nu_dot = const.M_inv.dot(
            tau
            #- const.D.dot(nu)
            - const.N(nu).dot(nu)
        )
        state_dot = np.concatenate([eta_dot, nu_dot])
        return state_dot
    
    def _thrust_surge(self, surge):
        surge = np.clip(surge, 0, 1)
        return surge*2.0 # self.config['thrust_max_auv'] in gym_auv/gym_auv/__init__.py
    def _moment_steer(self, steer):
        steer = np.clip(steer, -1, 1)
        return steer*0.15 # self.config['moment_max_auv'] in gym_auv/gym_auv/__init__.py
