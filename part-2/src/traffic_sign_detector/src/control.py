#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

# Author: Leon Jung, Gilbert, Ashe Kim
 
import rospy
import numpy as np
from enum import Enum
from std_msgs.msg import Float64, UInt8
from geometry_msgs.msg import Twist
from time import time

class ControlLane():
    def __init__(self):
        self.sub_lane = rospy.Subscriber('/control/lane', Float64, self.cbFollowLane, queue_size = 1)
        self.sub_max_vel = rospy.Subscriber('/control/max_vel', Float64, self.cbGetMaxVel, queue_size = 1)
        self.pub_cmd_vel = rospy.Publisher('/control/cmd_vel', Twist, queue_size = 1)

        self.sub_construction = rospy.Subscriber('/detect/construction_sign', UInt8, self.construction_cb, queue_size = 1)
        self.is_construction = 0
        self.sub_intersection = rospy.Subscriber('/detect/intersection_sign', UInt8, self.intersection_cb, queue_size = 1)
        self.is_intersection = 0
        self.sub_parking = rospy.Subscriber('/detect/parking_sign', UInt8, self.parking_cb, queue_size = 1)
        self.is_parking = 0

        self.sleep_until = time()
        self.ignore_sleep_until = time()
        self.time = time()

        self.lastError = 0
        self.MAX_VEL = 0.1

        rospy.on_shutdown(self.fnShutDown)

    def construction_cb(self, msg: UInt8):
        self.is_construction = msg.data

    def intersection_cb(self, msg: UInt8):
        self.is_intersection = msg.data # 1 intersection, 2 left, 3 right, 0 nothing

    def parking_cb(self, msg: UInt8):
        self.is_parking = msg.data # 1 intersection, 2 left, 3 right, 0 nothing

    def cbGetMaxVel(self, max_vel_msg):
        self.MAX_VEL = max_vel_msg.data

    def cbFollowLane(self, desired_center):
        center = desired_center.data

        error = center - 500

        Kp = 0.0025
        Kd = 0.007

        angular_z = Kp * error + Kd * (error - self.lastError)
        self.lastError = error
        
        twist = Twist()
        twist.linear.x = min(self.MAX_VEL * ((1 - abs(error) / 500) ** 2.2), 0.05)
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = -max(angular_z, -2.0) if angular_z < 0 else -min(angular_z, 2.0)
        if self.is_intersection == 2:
            rospy.loginfo('turning left')
            twist.angular.z = 10
        elif self.is_intersection == 3:
            rospy.loginfo('turning right')
            twist.angular.z = -10
        elif self.is_construction == 1 or self.is_intersection == 1:
            twist.linear.x /= 2
            rospy.loginfo('Slowing down for construction/intersection')
        elif self.is_parking:
            now = time()
            if now < self.sleep_until:
                return
            elif now - self.time < self.ignore_sleep_dur:
                pass
        
        self.time = time()
        self.pub_cmd_vel.publish(twist)

    def fnShutDown(self):
        rospy.loginfo("Shutting down. cmd_vel will be 0")

        twist = Twist()
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
        self.pub_cmd_vel.publish(twist) 

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('control')
    node = ControlLane()
    node.main()
