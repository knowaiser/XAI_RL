import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import queue

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20) # in self.clock.tick_busy_loop(30) it is suggested that fps=30
        self._queues = []
        self._settings = None
        self.collisions = []

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        try:
            self.frame = self.world.tick()
            data = [self._retrieve_data(q, timeout) for q in self._queues[:-1]]
            # collision sensor is the last element in the queue
            collision = self._detect_collision(self._queues[-1])
            
            assert all(x.frame == self.frame for x in data)

            #return data + [collision]
            return data + [collision]
        
        except queue.Empty:
            print("empty queue")
            #return None, None, None
            return None, None

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data
    
    def _detect_collision(self, sensor):
        # This collision is not fully aligned with other sensors, fix later
        try:
            data = sensor.get(block=False)
            return data
        except queue.Empty:
            return None
        
    ############################################################################
    # A new method to detect collision with curbs
    ############################################################################
        
    def on_collision(self, event):
        # Access the ID of the other actor involved in the collision
        other_actor = self.world.get_actor(event.other_actor_id)
        if other_actor is not None:
            # Check if the other actor's type is a curb or sidewalk
            # Note: You'll need to replace 'curb' and 'sidewalk' with the actual type names used in CARLA
            if 'curb' in other_actor.type_id or 'sidewalk' in other_actor.type_kd:
                print("Collision with curb detected!")
                # Here, apply a penalty or handle the collision accordingly

        # Attach the listener function to the collision sensor
        self.collision_sensor.listen(on_collision)
