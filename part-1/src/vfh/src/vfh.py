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
        rospy.init_node("vfh", anonymous=False)  # type: ignore

        self.heading = 0
        self.path = Path()

        self.goal = Point()
        self.goal.x = 7
        self.goal.y = 13

        self.dt = 0.05
        rate = 1 / self.dt

        self.angular_speed_pid = PID(1, 0, 0, self.dt)
        self.linear_velocity = 0.15

        self.a = 2
        self.b = .55
        self.alpha = 5  # number of degrees that each sector occupies
        self.threshold = 3
        self.smoothing_proximity = 2
        self.smax = 6

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
        ranges = map(
            lambda r: self.get_cell_value(prob=1, distance=r), laser_message.ranges
        )
        ranges = list(map(lambda r: r if r >= 0 else 0, ranges))
        chunk_count = len(laser_message.ranges) // self.alpha
        sectors: map[float] = map(
            lambda range_group: np.sum(range_group),
            np.array_split(ranges, chunk_count),
        )
        self.sectors = self.smooth_array(list(sectors), self.smoothing_proximity)
        return self.sectors

    def get_next_direction(self):
        sectors_extrema = argrelextrema(
            np.array(self.sectors), lambda x, _: x < self.threshold
        )[0]
        valleys: list[set[int]] = []
        current_valley_buffer = set()
        for idx in range(-len(sectors_extrema) + 1, len(sectors_extrema)):
            if sectors_extrema[idx] - sectors_extrema[idx - 1] == 1:
                current_valley_buffer.add(sectors_extrema[idx - 1])
                current_valley_buffer.add(sectors_extrema[idx])
            else:
                valleys.append(current_valley_buffer)
                current_valley_buffer = set()

        def revolving_sort(arr, mod):
            diff = [
                (i, abs(x - mod) if x > mod / 2 else x)
                for i, x in enumerate(np.mod(arr, mod))
            ]
            sorted_diff = sorted(diff, key=lambda x: x[1])
            return sorted_diff

        def mergable(seta: set, setb: set):
            l = revolving_sort(list(seta.union(setb)), len(self.sectors))
            for idx in range(1, len(l)):
                prev, curr = l[idx - 1][1], l[idx][1]
                if abs(curr - prev) > 1:
                    return False
            return True

        candidate_valleys: list[set[int]] = []
        for valleyi in valleys:
            for valleyj in valleys:
                if len(valleyi.intersection(valleyj)) > 0 or mergable(valleyi, valleyj):
                    candidate_valleys.append(valleyi.union(valleyj))
        for valleyi in valleys:
            for valleyj in valleys:
                if len(valleyi.intersection(valleyj)) > 0 or mergable(valleyi, valleyj):
                    candidate_valleys.append(valleyi.union(valleyj))

        candidates_to_remove = []
        for i, valleyi in enumerate(candidate_valleys):
            if len(valleyi) < self.smax:
                candidates_to_remove.append(valleyi)
                continue
            for j, valleyj in enumerate(candidate_valleys[:i]):
                if valleyi.issuperset(valleyj) and valleyi.difference(valleyj):
                    candidates_to_remove.append(valleyj)
                    continue
        for to_remove in candidates_to_remove:
            try: candidate_valleys.remove(to_remove)
            except: pass

        unique = []
        for valley in candidate_valleys:
            if valley not in unique:
                unique.append(valley)
        candidate_valleys = unique

        goal_degrees = math.degrees(self.get_rotation(self.goal, self.heading))
        if goal_degrees < 0:
            goal_degrees += 360
        goal_sector = goal_degrees // self.alpha
        candidate_sectors: list[tuple[int, np.intp, float]] = []
        for valley_idx, valley in enumerate(candidate_valleys):
            valley_sectors_distance_to_target = abs(
                np.array(list(valley)) - goal_sector
            )
            min_sector_distance_idx = np.argmin(
                np.mod(valley_sectors_distance_to_target, len(self.sectors))
            )
            candidate_sectors.append(
                (
                    valley_idx,
                    min_sector_distance_idx,
                    np.mod(
                        valley_sectors_distance_to_target[min_sector_distance_idx],
                        len(self.sectors),
                    ),
                )
            )
        if not candidate_sectors:
            return 0
        target_valley_idx, target_sector_idx_in_valley, _ = min(
            candidate_sectors, key=lambda c: c[2]
        )
        selected_valley = list(candidate_valleys[target_valley_idx])
        theta_sector_idx_range = (
            range(
                target_sector_idx_in_valley - self.smax + 1,
                target_sector_idx_in_valley + 1,
            )
            if target_sector_idx_in_valley + self.smax
            >= len(selected_valley)  # gt or gteq?
            else range(
                target_sector_idx_in_valley,
                target_sector_idx_in_valley + self.smax,  # + 1?
            )
        )
        theta_sector_range = list(map(lambda idx: selected_valley[idx], theta_sector_idx_range))  # type: ignore
        diff = [
            (i, abs(x - len(self.sectors)) if x > len(self.sectors) / 2 else x)
            for i, x in enumerate(np.mod(theta_sector_range, len(self.sectors)))
        ]
        sorted_diff = sorted(diff, key=lambda x: x[1])
        theta_median = theta_sector_range[sorted_diff[len(sorted_diff) // 2][0]]
        rospy.loginfo(
            f"\nself.heading {math.degrees(self.heading)}\ngoal_degrees {goal_degrees}\ngoal_sector {goal_sector}\nvalleys{valleys}\ncandidate_valleys {candidate_valleys}\ncandidate_sectors {candidate_sectors}\nselected_valley {selected_valley}\ntarget_sector {selected_valley[target_sector_idx_in_valley]}\ntheta_median {theta_median}\n"  # \nlaser_data {self.laser_data.ranges}"
        )
        return theta_median

    def run(self):
        rospy.loginfo("starting robot...")
        while not rospy.is_shutdown():
            rospy.wait_for_message("/scan", LaserScan)
            self.update_sectors(self.laser_data)
            next_dir_degrees = self.get_next_direction()
            error = math.radians(next_dir_degrees)
            if error > math.pi:
                error -= 2 * math.pi
            elif error < -math.pi:
                error += 2 * math.pi

            twist = Twist()
            twist.linear.x = self.linear_velocity * (math.pi - abs(error)) / math.pi
            twist.angular.z = self.angular_speed_pid.apply(error)
            rospy.loginfo(f"error {error} angular.z {twist.angular.z}")
            self.cmd_vel.publish(twist)
            self.r.sleep()


class PID:
    def __init__(self, k_p: float, k_i: float, k_d: float, dt: float):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        self.dt = dt

        self.prev_theta_error = 0
        self.sum_i_theta = 0

    def apply(self, err: float):
        self.sum_i_theta += err * self.dt

        P = self.k_p * err
        I = self.k_i * self.sum_i_theta
        D = self.k_d * (err - self.prev_theta_error)

        self.prev_theta_error = err
        return P + I + D


if __name__ == "__main__":
    try:
        vfh = VFH()
        vfh.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Terminated.")  # type: ignore
