#!/usr/bin/python3

from __future__ import annotations
import math
import numpy as np
from scipy.signal import argrelextrema
from typing import Iterable

import rospy
from geometry_msgs.msg import Twist, Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from tf import transformations


class VFH:
    def __init__(self):
        rospy.init_node("wall_follower", anonymous=False)  # type: ignore

        self.heading = 0
        self.path = Path()

        self.goal = Point()
        self.goal.x = -7
        self.goal.y = -13

        self.err_coeff = 0.05

        self.dt = 0.005
        rate = 1 / self.dt

        self.a = 1
        self.b = 0.25
        self.alpha = 5  # number of degrees that each sector occupies
        self.threshold = 1
        self.smoothing_proximity = 2
        self.smax = 2

        self.sectors: list[float] = [0] * (360 // self.alpha)

        self.r = rospy.Rate(rate)

        self.cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
        self.odom_sub = rospy.Subscriber(
            "/odom", Odometry, callback=self.odometry_callback
        )
        self.laser_sub = rospy.Subscriber(
            "/scan", LaserScan, callback=self.laser_callback
        )
        self.path_pub = rospy.Publisher("/path", Path, queue_size=10)
        self.errs = []

        rospy.wait_for_message("/odom", Odometry)
        rospy.wait_for_message("/scan", LaserScan)

    def laser_callback(self, msg: LaserScan):
        self.laser_data = msg

    def odometry_callback(self, msg: Odometry | None = None):  # type: ignore
        if msg is None:  # type: ignore
            msg: Odometry = rospy.wait_for_message("/odom", Odometry)  # type: ignore
        self.position = msg.pose.pose.position

        orientation = msg.pose.pose.orientation
        _, _, yaw = transformations.euler_from_quaternion(  # type: ignore
            (orientation.x, orientation.y, orientation.z, orientation.w)
        )
        self.heading = yaw

        pose = PoseStamped()
        self.path.header = pose.header = msg.header
        pose.pose = msg.pose.pose
        self.path.poses.append(pose)  # type: ignore
        rospy.logdebug("path %s", self.path)  # type: ignore
        self.path_pub.publish(self.path)  # type: ignore

    def get_rotation(self, target: Point, heading: float):
        rad = math.atan2(target.y - self.position.y, target.x - self.position.x)
        diff = rad - heading
        rospy.logdebug("heading: %s, rot: %s", math.degrees(heading), math.degrees(diff))  # type: ignore
        return diff

    def get_cell_value(self, prob: float, distance: float) -> float:
        return (prob**2) * (self.a - self.b * distance)

    @staticmethod
    def smooth_array(arr: list[float], l: int):
        proximate_indices = range(-l + 1, l)
        smooth_index = lambda idx: np.sum(
            [(l - abs(pi)) * arr[(idx + pi) % len(arr)] for pi in proximate_indices]
        ) / (2 * l + 1)
        return [smooth_index(idx) for idx, _ in enumerate(arr)]

    def update_sectors(self, laser_message: LaserScan) -> Iterable[float]:
        chunk_size = len(laser_message.ranges) // self.alpha
        ranges = map(
            lambda r: self.get_cell_value(prob=1, distance=r), laser_message.ranges
        )
        sectors: map[float] = map(
            lambda range_group: np.sum(range_group),
            np.array_split(list(ranges), chunk_size),
        )
        self.sectors = self.smooth_array(list(sectors), self.smoothing_proximity)
        return self.sectors

    def get_next_direction(self):
        sectors_extrema = argrelextrema(self.sectors, lambda x, _: x < self.threshold)
        valleys = []
        current_valley_buffer = set()
        for idx in range(len(sectors_extrema)):
            if sectors_extrema[idx] - sectors_extrema[idx - 1] == 1:
                current_valley_buffer.add(sectors_extrema[idx - 1])
                current_valley_buffer.add(sectors_extrema[idx])
            else:
                if len(current_valley_buffer) > self.smax:
                    valleys.append(list(current_valley_buffer))
                current_valley_buffer = set()
        rotation_degrees = math.degrees(self.get_rotation(self.goal, self.heading))
        rotation_sector = rotation_degrees // self.alpha
        candidate_sectors = []
        for valley_idx, valley in enumerate(valleys):
            valley_sectors_distance_to_target = np.array(valley) - rotation_sector
            min_sector_distance_idx = np.argmin(
                np.abs(valley_sectors_distance_to_target)
            )
            candidate_sectors.append(
                (
                    valley_idx,
                    min_sector_distance_idx,
                    np.abs(valley_sectors_distance_to_target[min_sector_distance_idx]),
                )
            )
        target_valley_idx, target_sector_idx_in_valley, _ = min(
            candidate_sectors, key=lambda c: c[2]
        )
        target_valley = valleys[target_valley_idx]
        theta_sector_range = (
            range(
                target_sector_idx_in_valley - self.smax, target_sector_idx_in_valley + 1
            )
            if target_sector_idx_in_valley + self.smax > len(target_valley)
            else range(
                target_sector_idx_in_valley, target_sector_idx_in_valley + self.smax + 1
            )
        )
        return ((theta_sector_range[0] + theta_sector_range[-1]) / 2) * self.alpha
    
    def run(self):
        rospy.loginfo("starting robot...")
        while not rospy.is_shutdown():
            rospy.wait_for_message("/scan", LaserScan)
            self.update_sectors(self.laser_data)
            next_dir_degrees = self.get_next_direction()
            error = math.radians(next_dir_degrees)
            twist = Twist()
            twist.angular.z = self.err_coeff * error
            self.cmd_vel.publish()
            self.r.sleep()


if __name__ == "__main__":
    try:
        vfh = VFH()
        vfh.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Terminated.")  # type: ignore
