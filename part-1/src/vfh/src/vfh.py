#!/usr/bin/python3

from __future__ import annotations
import math
import numpy as np
from scipy.signal import argrelextrema
from typing import Iterable
from ordered_set import OrderedSet

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

        self.goal = Point(4.5, 0, 0)
        self.goals = [
            Point(3, 4.5, 0),
            Point(2.5, 1.2, 0),
            Point(.5, 1.5, 0),
            Point(3.5, 6, 0.0),
            Point(5.5, 5, 0),
            Point(6.7, 2.6, 0),
            Point(8, 6, 0),
            Point(13, 7, 0),
        ]

        self.dt = 0.005
        rate = 1 / self.dt

        self.angular_speed_pid = PID(0.4, 0, 0, self.dt)
        self.min_linear_velocity = 0.05
        self.max_velocity = .75
        self.h_m = 2.5
        self.get_linear_velocity_factor = (
            lambda: self.max_velocity * (1 - min(self.sectors[0], self.h_m) / self.h_m)
        )
        self.epsilon = .5
        self.max_rotation = 2 * math.pi

        self.a = 1
        self.b = 0.25
        self.alpha = 5  # number of degrees that each sector occupies
        self.threshold = 2.95
        self.smoothing_proximity = 2
        self.smax = 12

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

    @staticmethod
    def revolving_sort(arr, mod):
        diff = [
            (i, abs(x - mod) if x > mod / 2 else x)
            for i, x in enumerate(np.mod(arr, mod))
        ]
        sorted_diff = sorted(diff, key=lambda x: x[1])
        return sorted_diff

    @staticmethod
    def mergable(seta: OrderedSet, setb: OrderedSet, mod: int) -> tuple[bool, list[OrderedSet]]:
        if not seta or not setb:
            return False, []

        if np.mod(seta[0] - 1, mod) == setb[-1]:
            return True, [setb, seta]

        if np.mod(seta[-1] + 1, mod) == setb[0]:
            return True, [seta, setb]

        intersection = seta.intersection(setb)
        if len(intersection) <= 0:
            return False, []
        
        if intersection[0] == seta[0]:
            return True, [setb, seta]
        else:
            return True, [seta, setb]

    @staticmethod
    def merge_mod_extremas(extremas: np.ndarray, mod: int):
        valleys: list[OrderedSet[int]] = []
        current_valley_buffer = OrderedSet([])
        for idx in range(-len(extremas) + 1, len(extremas)):
            if extremas[idx] - extremas[idx - 1] == 1:
                current_valley_buffer.add(extremas[idx - 1])
                current_valley_buffer.add(extremas[idx])
            else:
                valleys.append(current_valley_buffer)
                current_valley_buffer = OrderedSet([])
        candidate_valleys: list[OrderedSet[int]] = []
        for valleyi in valleys:
            for valleyj in valleys:
                mergable, valleys_in_order = VFH.mergable(valleyi, valleyj, mod)
                if mergable and valleys_in_order:
                    candidate_valleys.append(valleys_in_order[0].union(valleys_in_order[1]))
        for valleyi in valleys:
            for valleyj in valleys:
                mergable, valleys_in_order = VFH.mergable(valleyi, valleyj, mod)
                if mergable and valleys_in_order:
                    candidate_valleys.append(valleys_in_order[0].union(valleys_in_order[1]))

        candidates_to_remove = []
        for i, valleyi in enumerate(candidate_valleys):
            for _, valleyj in enumerate(candidate_valleys[:i]):
                if valleyi.issuperset(valleyj) and valleyi.difference(valleyj):
                    candidates_to_remove.append(valleyj)
                    continue
        for to_remove in candidates_to_remove:
            try:
                candidate_valleys.remove(to_remove)
            except:
                pass

        unique = []
        for valley in candidate_valleys:
            if valley not in unique:
                unique.append(valley)
        candidate_valleys = unique
        return valleys, candidate_valleys

    @staticmethod
    def get_best_sectors_in_valley_wrt_goal(valleys: list, goal_sector: int, mod: int):
        candidate_sectors: list[tuple[int, np.intp, float]] = []
        for valley_idx, valley in enumerate(valleys):
            valley = np.array(list(valley))
            a, b = valley - goal_sector, goal_sector - valley
            moda, modb = np.mod(a, mod), np.mod(b, mod)
            argmina, argminb = np.argmin(moda), np.argmin(modb)
            distance_a, distance_b = moda[argmina], modb[argminb]
            if distance_a < distance_b:
                min_sector_distance_idx = argmina
                distance = distance_a
            else:
                min_sector_distance_idx = argminb
                distance = distance_b
            candidate_sectors.append((valley_idx, min_sector_distance_idx, distance))
        return candidate_sectors

    def get_next_direction(self):
        # print(f'sectors {self.sectors}')
        sectors_extrema = argrelextrema(
            np.array(self.sectors), lambda x, _: x < self.threshold
        )[0]
        unmerged_valleys, candidate_valleys = VFH.merge_mod_extremas(
            sectors_extrema, len(self.sectors)
        )

        goal_degrees = math.degrees(self.get_rotation(self.goal, self.heading))
        if goal_degrees < 0:
            goal_degrees += 360
        goal_sector = int(goal_degrees // self.alpha)

        candidate_sectors = VFH.get_best_sectors_in_valley_wrt_goal(
            candidate_valleys, goal_sector, mod=len(self.sectors)
        )
        if not candidate_sectors:
            return 0

        selected_valley_idx, near_border, _ = min(candidate_sectors, key=lambda c: c[2])
        selected_valley = list(candidate_valleys[selected_valley_idx])
        if len(selected_valley) > self.smax:
            near_border_far_border_idx_range = (
                range(
                    near_border - self.smax + 1,
                    near_border + 1,
                )
                if near_border + self.smax >= len(selected_valley)  # gt or gteq?
                else range(
                    near_border,
                    near_border + self.smax,  # + 1?
                )
            )
        else:
            near_border_far_border_idx_range = range(0, len(selected_valley))
        near_far_border_range = list(map(lambda idx: selected_valley[idx], near_border_far_border_idx_range))  # type: ignore
        revolving_sorted_near_far_border_range = VFH.revolving_sort(
            near_far_border_range, len(self.sectors)
        )
        theta = (
            near_far_border_range[
                revolving_sorted_near_far_border_range[
                    len(revolving_sorted_near_far_border_range) // 2
                ][0]
            ]
            * self.alpha
        )
        rospy.loginfo(
            f"\
            \ncurrently at ({self.position.x:.2}, {self.position.y:.2}) \
            \ngoal at ({self.goal.x:.2F}, {self.goal.y:.2F}) \
            \nself.heading {math.degrees(self.heading):.2F}\
            \ngoal_sector {goal_sector}\
            \ncandidate_valleys {candidate_valleys}\
            \ncandidate_sectors {candidate_sectors}\
            \nselected_valley {selected_valley}\
            \ntarget_sector {selected_valley[near_border]}\
            \ntheta {theta}\n"
            # \nunmerged_valleys {unmerged_valleys}"
        )
        return theta

    @staticmethod
    def get_distance(s: Point, f: Point) -> float:
        return np.sqrt((s.x - f.x) ** 2 + (s.y - f.y) ** 2)

    def run(self):
        rospy.loginfo("starting robot...")
        while not rospy.is_shutdown():
            if VFH.get_distance(self.position, self.goal) < self.epsilon:
                if not self.goals:
                    rospy.loginfo("reached final goal")
                    return
                self.goal = self.goals[0]
                self.goals = self.goals[1:]
                rospy.loginfo("got next goal at (%s, %s)", self.goal.x, self.goal.y)

            rospy.wait_for_message("/scan", LaserScan)
            self.update_sectors(self.laser_data)
            next_dir_degrees = self.get_next_direction()
            error = math.radians(next_dir_degrees)
            if error > math.pi:
                error -= 2 * math.pi
            elif error < -math.pi:
                error += 2 * math.pi

            twist = Twist()
            velocity_factor = self.get_linear_velocity_factor() 
            rotation_factor = (1 - abs(error) / self.max_rotation)
            v = velocity_factor * rotation_factor
            twist.linear.x = v + self.min_linear_velocity
            twist.angular.z = self.angular_speed_pid.apply(error)
            rospy.loginfo(f"error {error} angular.z {twist.angular.z} vfactor {velocity_factor:.2F} rotfactor {rotation_factor:.2F}")
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
