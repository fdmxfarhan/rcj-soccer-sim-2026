from rcj_soccer_robot import TIME_STEP
import numpy as np
import math

o2_goal_attack_mode = 0
o2_ambush_attack_mode = 1
o2_stationing_attack_mode = 2
o2_double_pushing_mode = 3
o2_midfield_defense_mode = 4
o2_wall_defense_mode = 5
o2_deflection_defense_mode = 6
o2_goal_defense_mode = 7
o2_trap_ball_main_mode = 8
o2_trap_ball_support_mode = 9


# Small control tolerances: deliberately conservative to avoid changing the main strategy.
O2_MODE_HYSTERESIS = 0.015
O2_STEERING_DEADBAND = 1.0
O2_ATTACK_EPSILON = 1e-4
O2_DUPLICATE_CHASE_MARGIN = 0.08
O2_SUPPORT_X_LIMIT = 0.34
O2_SUPPORT_Y_MIN = -0.35
O2_SUPPORT_Y_MAX = 0.25
O2_TEAMMATE_SEPARATION_DISTANCE = 0.18
O2_TEAMMATE_SEPARATION_OFFSET = 0.10

class OffenseYellow2:
    o2_mode = 0
    o2_side = 1
    o2_goal_x, o2_goal_y = 0, 0.75
    o2_off_angle = 25
    o2_last_ball_x, o2_last_ball_y = 0, 0
    o2_last_mode = 0
    o2_limit_x = -0.12
    o2_timer = 0
    o2_push_ball = False
    o2_trap_stage = -1
    o2_double_push_x, o2_double_push_y = 0, 0
    o2_wall_dest_x, o2_wall_dest_y = 0.44, -0.7
    o2_team = "Y"

    # Run Blue
    def execute_strategy_two(self, o2_robot, o2_data, o2_lop1, o2_lop2, o2_roles, o2_trap_stage):
        o2_left_speed, o2_right_speed = 0, 0

        o2_robot_angle = o2_data.getRobotHeading()
        o2_robot_x, o2_robot_y = o2_data.getRobotPos()
        o2_ball_x, o2_ball_y = o2_data.getBallPos()
        o2_ball_distance = o2_data.getBallDistance()
        o2_ball_pred_x, o2_ball_pred_y = o2_data.getBallPredPos()
        o2_ball_pred_angle = o2_data.getBallPredAngle()
        o2_offense1_x, o2_offense1_y = o2_data.getRobotPosSearch(o2_roles.getRoleToRobot(1))
        o2_offense2_x, o2_offense2_y = o2_data.getRobotPosSearch(o2_roles.getRoleToRobot(2))
        o2_defense_x, o2_defense_y = o2_data.getRobotPosSearch(o2_roles.getRoleToRobot(3))
        o2_offense1_angle = o2_data.getHeadingSearch(o2_roles.getRoleToRobot(1))
        o2_offense1_ball_distance = o2_data.getBallDistanceSearch(o2_roles.getRoleToRobot(1))
        o2_offense2_ball_distance = o2_data.getBallDistanceSearch(o2_roles.getRoleToRobot(2))
        o2_detect = o2_data.getTeamDetect()

        # C5/C6: use team ball awareness to avoid two offensive robots
        # independently committing to the same ball.
        o2_primary_observer = o2_data.getPrimaryBallObserver()
        o2_team_ball_state = o2_data.getTeamBallState()
        o2_other_offense_id = o2_roles.getRoleToRobot(1)
        o2_yield_to_teammate = (
            o2_team_ball_state == "KNOWN" and
            o2_primary_observer == o2_other_offense_id and
            o2_offense1_ball_distance + O2_DUPLICATE_CHASE_MARGIN < o2_offense2_ball_distance
        )

        o2_ball_lack_progress_short = o2_lop1["ball"]

        o2_spin_speed = 6
        o2_angle_range = 15

        o2_run_pass = False

        self.o2_goal_x, self.o2_goal_y = 0, 0.75

        if o2_ball_y > 0.65:
            self.o2_limit_x = -0.15
        else:
            self.o2_limit_x = -0.12

        if o2_detect and not o2_data.is_flipped:
            if o2_robot_y > -0.6 and o2_ball_y < -0.6 and o2_ball_x > 0:
                o2_offset = 0.15
                if o2_defense_y < 0.62:
                    self.o2_wall_dest_x, self.o2_wall_dest_y = 0.44, -0.7
                elif o2_defense_x < 0.25:
                    self.o2_wall_dest_x, self.o2_wall_dest_y = 0.25 + o2_offset, -0.7
                elif o2_defense_x > 0.4:
                    self.o2_wall_dest_x, self.o2_wall_dest_y = 0.4 + o2_offset, -0.7
                else:
                    self.o2_wall_dest_x, self.o2_wall_dest_y = o2_defense_x + o2_offset, -0.7
            if self.o2_trap_stage != -1 or (
                    o2_offense1_y < -0.67 and o2_offense2_y < -0.67 and o2_ball_y < -0.7 and o2_offense2_x < o2_ball_x < o2_offense1_x and o2_ball_x > 0.2 and o2_offense1_y > o2_ball_y and o2_offense2_y > o2_ball_y and 0.01 < OffenseYellow2.calc_distance_two(
                    self, o2_offense1_x, o2_offense1_y, o2_offense2_x, o2_offense2_y) < 0.16):
                self.o2_mode = o2_trap_ball_main_mode
                if o2_offense1_y < -0.67 and o2_offense2_y < -0.67 and o2_ball_y < -0.7 and o2_offense2_x < o2_ball_x < o2_offense1_x and o2_ball_x > 0.2 and o2_offense1_y > o2_ball_y and o2_offense2_y > o2_ball_y and 0.01 < OffenseYellow2.calc_distance_two(
                        self, o2_offense1_x, o2_offense1_y, o2_offense2_x, o2_offense2_y) < 0.16 and self.o2_trap_stage == -1:
                    self.o2_trap_stage = 1
            elif o2_trap_stage != -1:
                self.o2_mode = o2_trap_ball_support_mode
            elif o2_ball_y < -0.66 and o2_ball_x > 0:
                self.o2_mode = o2_wall_defense_mode
            elif o2_ball_y < -0.64 and -0.38 <= o2_ball_x and (o2_ball_x <= -0.25 or o2_robot_y < o2_ball_y):
                self.o2_mode = o2_goal_defense_mode
            elif o2_ball_y < -0.64 and not 0.25 > o2_ball_x > -0.25:
                self.o2_mode = o2_midfield_defense_mode
            elif o2_ball_y < -0.53 and 0.25 > o2_ball_x > -0.25 and o2_defense_y < -0.59:
                self.o2_mode = o2_deflection_defense_mode
            elif o2_ball_lack_progress_short and o2_ball_x < self.o2_limit_x:
                if OffenseYellow2.inside_angle_range_two(self, o2_offense1_angle, 55, o2_angle_range, 35, 20) and -0.42 < o2_ball_x < -0.12 and o2_ball_y > 0.65 and \
                        -0.47 < o2_offense1_x < -0.12 and o2_offense1_y > 0.64 and o2_robot_y > 0.3:
                    self.o2_mode = o2_double_pushing_mode
                else:
                    self.o2_mode = o2_stationing_attack_mode
            else:
                o2_goal_condition = (o2_ball_x >= self.o2_limit_x and
                                     (o2_ball_y <= -0.22 or
                                      o2_offense2_ball_distance < o2_offense1_ball_distance or
                                      o2_ball_x > -self.o2_limit_x))
                o2_goal_hold = (self.o2_mode == o2_goal_attack_mode and
                                o2_ball_x >= self.o2_limit_x - O2_MODE_HYSTERESIS)
                if ((self.o2_push_ball is True and o2_ball_x >= -0.2) or
                        o2_goal_condition or o2_goal_hold):
                    self.o2_mode = o2_goal_attack_mode
                    if self.o2_last_mode == o2_ambush_attack_mode:
                        self.o2_push_ball = True
                else:
                    self.o2_push_ball = False
                    self.o2_mode = o2_ambush_attack_mode
            self.o2_timer = 0
        else:
            self.o2_mode = o2_stationing_attack_mode
            self.o2_timer += (TIME_STEP / 1000)

        # C6: the observing/closer offensive teammate owns the active chase.
        if o2_yield_to_teammate and self.o2_mode == o2_goal_attack_mode:
            self.o2_mode = o2_stationing_attack_mode

        # Ball Chasing and Shooting
        if self.o2_mode == o2_goal_attack_mode:
            if OffenseYellow2.inside_position_range_two(self, o2_ball_x, o2_ball_y, 0, 0, 0.1, 0.1) and o2_ball_distance < 0.15 and o2_robot_y < o2_ball_y and OffenseYellow2.calc_distance_two(self, o2_ball_x, o2_ball_y, self.o2_last_ball_x, self.o2_last_ball_y) < math.pow(10, -5) * 5:
                self.o2_goal_x, self.o2_goal_y = 0.05, 0.75
            elif o2_ball_x < -0.25 and o2_ball_y < 0:
                self.o2_goal_x, self.o2_goal_y = -0.05, 0.75
            elif o2_ball_x > 0.25 and o2_ball_y < 0:
                self.o2_goal_x, self.o2_goal_y = 0.05, 0.75
            else:
                self.o2_goal_x, self.o2_goal_y = 0.03, 0.75

            # RECHECK
            o2_goal_angle = (math.degrees(math.atan2(self.o2_goal_x - o2_ball_pred_x, self.o2_goal_y - o2_ball_pred_y)) + 360) % 360

            if o2_ball_y > 0.6 and (o2_ball_x > 0.35 or o2_ball_x < -0.35):
                o2_goal_angle = 295

            if (o2_ball_x > 0.52 or o2_ball_x < -0.52) and -0.54 < o2_ball_y < 0.54:
                o2_goal_angle = 0
            elif o2_ball_x > 0.55 or o2_ball_x < -0.55:
                o2_goal_angle = 0

            # RECHECK
            if 0 < o2_ball_y < 0.5 and -0.4 < o2_ball_x < 0.4:
                o2_goal_temp_angle = math.atan2(abs(self.o2_goal_x - o2_ball_pred_x), abs(self.o2_goal_y - o2_ball_pred_y))
                o2_goal_temp_distance = 0.015
                o2_ball_temp_x = o2_ball_pred_x - math.sin(o2_goal_temp_angle) * o2_goal_temp_distance
                o2_ball_temp_y = o2_ball_pred_y + math.cos(o2_goal_temp_angle) * o2_goal_temp_distance
            else:
                o2_goal_temp_angle = math.atan2(self.o2_goal_x - o2_ball_pred_x, self.o2_goal_y - o2_ball_pred_y)
                o2_goal_temp_distance = 0.015
                o2_ball_temp_x = o2_ball_pred_x + math.sin(o2_goal_temp_angle) * o2_goal_temp_distance
                o2_ball_temp_y = o2_ball_pred_y + math.cos(o2_goal_temp_angle) * o2_goal_temp_distance

            o2_ball_pred_angle = (math.degrees(
                math.atan2(o2_ball_temp_x - o2_robot_x, o2_ball_temp_y - o2_robot_y)) - o2_robot_angle + 360) % 360

            o2_ball_direction = o2_ball_pred_angle
            if o2_ball_direction > 180:
                o2_ball_direction = o2_ball_direction - 360
            o2_abs_ba = (o2_ball_direction + o2_robot_angle - o2_goal_angle + 360) % 360

            if o2_abs_ba < self.o2_off_angle or o2_abs_ba > 360 - self.o2_off_angle:
                o2_offset = 0
            elif o2_ball_distance < 0.1:
                o2_offset = 62
            elif o2_ball_distance > 1:
                o2_offset = 1
            else:
                o2_offset = 61 * pow(o2_ball_distance - 1, 2) + 1

            if 180 < o2_abs_ba < 360 - self.o2_off_angle:
                o2_offset *= -1

            if o2_ball_pred_y > 0.66 and (0.55 < o2_ball_x < 0.62 or -0.55 > o2_ball_x > -0.62):
                o2_offset *= 0.5
            elif (o2_robot_x >= 0.51 or o2_robot_x <= -0.51) and 0.63 > o2_ball_pred_y - 0.0375 > o2_robot_y:
                o2_offset *= (0.63 - abs(o2_robot_y)) / 0.9
            elif (o2_ball_x <= -0.42 or (o2_ball_x <= -0.35 and o2_ball_y < 0.1)) and (o2_robot_x <= -0.35 or o2_robot_x >= 0.35) and self.o2_off_angle < o2_abs_ba < 180:
                o2_offset *= -1
            elif (o2_ball_x >= 0.42 or (o2_ball_x >= 0.35 and o2_ball_y < 0.1)) and (o2_robot_x <= -0.35 or o2_robot_x >= 0.35) and 180 < o2_abs_ba < 360 - self.o2_off_angle:
                o2_offset *= -1

            o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_ball_pred_angle, 0, 180)

            if o2_closest_angle == 0:
                self.o2_side = 1
            elif o2_closest_angle == 180:
                self.o2_side = -1

            if self.o2_side == 1:
                o2_move_angle = o2_ball_direction + o2_offset
            else:
                if o2_ball_direction + o2_offset >= 0:
                    o2_move_angle = o2_ball_direction + o2_offset - 180
                else:
                    o2_move_angle = 180 - abs((o2_ball_direction + o2_offset))

            if OffenseYellow2.inside_position_range_two(self, o2_ball_x, o2_ball_y, 0.575, 0.73, 0.01, 0.01) and o2_ball_distance < 0.07 and o2_ball_y > o2_robot_y:
                o2_shoot_angle = 300
                o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle, (o2_shoot_angle + 360) % 360,
                                                                       (o2_shoot_angle + 360 + 180) % 360)
                if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                    o2_spin_speed = 6
                    o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle, o2_closest_angle,
                                                            o2_spin_speed)
            elif -0.58 < o2_ball_x < 0.58 and 0.675 > o2_ball_y > -0.2:
                o2_left_speed, o2_right_speed = OffenseYellow2.curve_drive_two(self, o2_move_angle, self.o2_side, 45)
            elif (o2_ball_x < -0.58 or o2_ball_x > 0.58) and o2_ball_y < 0:
                o2_left_speed, o2_right_speed = OffenseYellow2.curve_drive_two(self, o2_move_angle, self.o2_side, 90)
            else:
                o2_left_speed, o2_right_speed = OffenseYellow2.curve_drive_two(self, o2_move_angle, self.o2_side, 60)

        # Robot Ambushing
        elif self.o2_mode == o2_ambush_attack_mode:
            o2_angle_range = 4
            o2_shoot_angle = 0
            o2_point_x = self.o2_limit_x + 0.04
            if OffenseYellow2.calc_distance_two(self, o2_ball_x, o2_ball_y, self.o2_last_ball_x, self.o2_last_ball_y) < math.pow(10, -5):
                o2_dest_x, o2_dest_y = o2_point_x, o2_ball_y
                o2_offset = -0.0725
            else:
                o2_dest_x, o2_dest_y = OffenseYellow2.solve_attack_point_two(self, o2_point_x, o2_ball_x, o2_ball_y, self.o2_last_ball_x, self.o2_last_ball_y)
                if o2_dest_x is None or o2_dest_y is None:
                    o2_dest_x, o2_dest_y = o2_point_x, o2_ball_y - 0.075
                o2_offset = 0

            o2_dest_y = o2_dest_y + o2_offset

            if o2_dest_y < 0 and o2_dest_y < o2_ball_y - 0.1:
                o2_dest_y = o2_ball_y - 0.075
            elif (o2_dest_y < -0.1 and o2_ball_y > 0) or (o2_dest_y > 0.1 and o2_ball_y < 0):
                o2_dest_y = -0.0725
            elif o2_dest_y > 0 > o2_ball_y and abs(o2_ball_y - o2_dest_y) > 0.15 and o2_ball_y < self.o2_last_ball_y:
                o2_dest_y = -0.0725
            elif o2_dest_y < 0 < o2_ball_y and abs(o2_ball_y - o2_dest_y) > 0.15 and o2_ball_y > self.o2_last_ball_y:
                o2_dest_y = -0.0725

            if o2_ball_y < 0.57 and o2_dest_y > 0.54:
                o2_dest_y = 0.54
            elif o2_dest_y < -0.4 and -0.25 < o2_ball_y < 0.25:
                o2_dest_y = -0.4
            elif o2_dest_y > 0.675:
                o2_dest_y = 0.675

            if not OffenseYellow2.inside_position_range_two(self, o2_robot_x, o2_robot_y, o2_dest_x, o2_dest_y, 0.005, 0.005):
                o2_robot_radius = 0.0475
                o2_ball_radius = 0.021
                if o2_ball_distance < 0.1 and abs(o2_robot_x) + o2_robot_radius > abs(o2_ball_x) - o2_ball_radius > abs(o2_dest_x) and \
                        o2_robot_y + o2_robot_radius > o2_ball_y - o2_ball_radius > o2_dest_y:
                    o2_temp_dest_x, o2_temp_dest_y = OffenseYellow2.solve_target_point_two(self, o2_point_x, o2_robot_x, o2_robot_y, o2_ball_x, o2_ball_y + 0.1)
                    o2_left_speed, o2_right_speed = OffenseYellow2.drive_to_point_two(self, o2_left_speed, o2_right_speed, o2_temp_dest_x, o2_temp_dest_y, o2_robot_x,
                                                               o2_robot_y, o2_robot_angle, o2_spin_speed, o2_angle_range)
                else:
                    o2_left_speed, o2_right_speed = OffenseYellow2.drive_to_point_two(self, o2_left_speed, o2_right_speed, o2_dest_x, o2_dest_y, o2_robot_x,
                                                           o2_robot_y, o2_robot_angle, o2_spin_speed, o2_angle_range)
                o2_left_speed, o2_right_speed = OffenseYellow2.drive_to_point_two(self, o2_left_speed, o2_right_speed, o2_dest_x, o2_dest_y, o2_robot_x,
                                                           o2_robot_y, o2_robot_angle, o2_spin_speed, o2_angle_range)
            else:
                o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle, (o2_shoot_angle + 360) % 360, (o2_shoot_angle + 360 + 180) % 360)
                if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                    o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle, o2_closest_angle, o2_spin_speed)
                else:
                    o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 0, 2)

        # Robot Stationing and Tracking
        elif self.o2_mode == o2_stationing_attack_mode:
            if not o2_detect:
                if o2_ball_x > 0:
                    if o2_ball_y < -0.6:
                        o2_dest_x, o2_dest_y = 0.3, -0.35
                    else:
                        o2_dest_x, o2_dest_y = 0.3, -0.2
                else:
                    if o2_ball_y < -0.6:
                        if self.o2_timer < 5:
                            o2_dest_x, o2_dest_y = 0, -0.35
                        else:
                            o2_dest_x, o2_dest_y = 0.2, -0.35
                    else:
                        if self.o2_timer < 5:
                            o2_dest_x, o2_dest_y = 0, -0.2
                        else:
                            o2_dest_x, o2_dest_y = 0.2, -0.2
            else:
                if o2_yield_to_teammate:
                    # C7: dynamic support point, mirrored from Offense 1.
                    o2_primary_x, o2_primary_y = o2_data.getRobotPosSearch(o2_primary_observer)
                    o2_dest_x = max(-O2_SUPPORT_X_LIMIT, min(O2_SUPPORT_X_LIMIT,
                                                              0.35 * o2_ball_x - 0.20 * o2_primary_x))
                    o2_dest_y = max(O2_SUPPORT_Y_MIN, min(O2_SUPPORT_Y_MAX,
                                                          0.30 * o2_ball_y - 0.10 * o2_primary_y - 0.12))
                elif o2_ball_y > 0:
                    if o2_ball_x < 0:
                        o2_attack_point_distance = OffenseYellow2.calc_distance_two(self, o2_ball_x, o2_ball_y, -0.3, 0.3)
                        o2_defense_point_distance = OffenseYellow2.calc_distance_two(self, o2_ball_x, o2_ball_y, -0.3, -0.3)
                    else:
                        o2_attack_point_distance = OffenseYellow2.calc_distance_two(self, o2_ball_x, o2_ball_y, 0.3, 0.3)
                        o2_defense_point_distance = OffenseYellow2.calc_distance_two(self, o2_ball_x, o2_ball_y, 0.3, -0.3)
                    if o2_attack_point_distance + 0.03 < o2_defense_point_distance:
                        o2_dest_x, o2_dest_y = 0, 0
                    else:
                        o2_dest_x, o2_dest_y = 0, -0.1
                else:
                    o2_dest_x, o2_dest_y = 0, -0.2
            # Anti-clustering: when the ball is temporarily lost, both
            # offensive robots can otherwise receive the same stationing
            # target (especially around x=0).  Give them a small role-based
            # separation only while they are physically too close.
            if not o2_detect:
                o2_teammate_dx = o2_robot_x - o2_offense1_x
                o2_teammate_dy = o2_robot_y - o2_offense1_y
                o2_teammate_distance = math.hypot(o2_teammate_dx, o2_teammate_dy)
                if o2_teammate_distance < O2_TEAMMATE_SEPARATION_DISTANCE:
                    if o2_teammate_distance > 1e-6:
                        o2_dest_x += O2_TEAMMATE_SEPARATION_OFFSET * o2_teammate_dx / o2_teammate_distance
                        o2_dest_y += O2_TEAMMATE_SEPARATION_OFFSET * o2_teammate_dy / o2_teammate_distance
                    else:
                        o2_dest_x += O2_TEAMMATE_SEPARATION_OFFSET

            o2_angle_range = 8
            if not OffenseYellow2.inside_position_range_two(self, o2_robot_x, o2_robot_y, o2_dest_x, o2_dest_y, 0.02, 0.02):
                o2_left_speed, o2_right_speed = OffenseYellow2.drive_to_point_two(self, o2_left_speed, o2_right_speed, o2_dest_x, o2_dest_y, o2_robot_x,
                                                           o2_robot_y, o2_robot_angle, o2_spin_speed, o2_angle_range)
            else:
                o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle, 0, 180)
                if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                    o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle, o2_closest_angle, o2_spin_speed)
                else:
                    o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 0, 2)

        # Double Pushing
        elif self.o2_mode == o2_double_pushing_mode:
            o2_angle_range = 8
            o2_offset_x, o2_offset_y = -0.03, -0.035
            self.o2_double_push_x, self.o2_double_push_y = o2_offense1_x + o2_offset_x, o2_offense1_y + o2_offset_y
            o2_dest_x, o2_dest_y = self.o2_double_push_x, self.o2_double_push_y
            if OffenseYellow2.inside_position_range_two(self, o2_robot_x, o2_robot_y, o2_dest_x, o2_dest_y, 0.05, 0.05) or \
                    OffenseYellow2.inside_position_range_two(self, o2_robot_x, o2_robot_y, o2_offense1_x, o2_offense1_y, 0.1,
                                           0.1) and o2_robot_x < o2_offense1_x:
                o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle, 55, 235)
                if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                    o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle, o2_closest_angle,
                                                            o2_spin_speed)
                else:
                    o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 55, -10)
            else:
                o2_point_angle = (math.degrees(math.atan2(o2_dest_x - o2_robot_x, o2_dest_y - o2_robot_y)) - o2_robot_angle + 360) % 360

                o2_push_angle = 60
                o2_point_direction = o2_point_angle
                if o2_point_direction > 180:
                    o2_point_direction = o2_point_direction - 360
                o2_abs_ba = (o2_point_direction + o2_robot_angle - o2_push_angle + 360) % 360

                o2_point_distance = OffenseYellow2.calc_distance_two(self, o2_robot_x, o2_robot_y, o2_dest_x, o2_dest_y)

                if o2_abs_ba < self.o2_off_angle or o2_abs_ba > 360 - self.o2_off_angle:
                    o2_offset = 0
                elif o2_point_distance < 0.1:
                    o2_offset = 62
                elif o2_point_distance > 1:
                    o2_offset = 1
                else:
                    o2_offset = 61 * pow(o2_point_distance - 1, 2) + 1

                if 180 < o2_abs_ba < 360 - self.o2_off_angle:
                    o2_offset *= -1

                o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, (o2_robot_angle + 180) % 360, 0, 180)

                if self.o2_last_mode != o2_double_pushing_mode:
                    if o2_closest_angle == 0:
                        self.o2_side = 1
                    elif o2_closest_angle == 180:
                        self.o2_side = -1

                if self.o2_side == 1:
                    o2_move_angle = o2_point_direction + o2_offset
                else:
                    if o2_point_direction + o2_offset >= 0:
                        o2_move_angle = o2_point_direction + o2_offset - 180
                    else:
                        o2_move_angle = 180 - abs(o2_point_direction + o2_offset)

                o2_left_speed, o2_right_speed = OffenseYellow2.curve_drive_two(self, o2_move_angle, self.o2_side, 45)

        # Wall Defense
        elif self.o2_mode == o2_wall_defense_mode:
            o2_dest_x, o2_dest_y = self.o2_wall_dest_x, self.o2_wall_dest_y
            o2_angle_range = 15
            if not OffenseYellow2.inside_position_range_two(self, o2_robot_x, o2_robot_y, o2_dest_x, o2_dest_y, 1, 1, 0.02, 0.4, 0.05, 1):
                o2_point_angle = (math.degrees(math.atan2(o2_dest_x - o2_robot_x, o2_dest_y - o2_robot_y)) - o2_robot_angle + 360) % 360

                o2_point_direction = o2_point_angle
                if o2_point_direction > 180:
                    o2_point_direction = o2_point_direction - 360

                o2_offset = 0

                o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_point_angle, 0, 180)

                if o2_closest_angle == 0:
                    self.o2_side = 1
                elif o2_closest_angle == 180:
                    self.o2_side = -1

                if self.o2_side == 1:
                    o2_move_angle = o2_point_direction + o2_offset
                else:
                    if o2_point_direction + o2_offset >= 0:
                        o2_move_angle = o2_point_direction + o2_offset - 180
                    else:
                        o2_move_angle = 180 - abs(o2_point_direction + o2_offset)

                o2_left_speed, o2_right_speed = OffenseYellow2.curve_drive_two(self, o2_move_angle, self.o2_side, 60)
            else:
                o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle, 0, 180)
                if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                    o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle, o2_closest_angle,
                                                            o2_spin_speed)
                else:
                    o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 0, 10)

        # Midfield Defense
        elif self.o2_mode == o2_midfield_defense_mode:
            o2_dest_x, o2_dest_y = 0, -0.54
            o2_angle_range = 8
            if not OffenseYellow2.inside_position_range_two(self, o2_robot_x, o2_robot_y, o2_dest_x, o2_dest_y, 0.02, 0.02):
                o2_left_speed, o2_right_speed = OffenseYellow2.drive_to_point_two(self, o2_left_speed, o2_right_speed, o2_dest_x, o2_dest_y, o2_robot_x,
                                                           o2_robot_y, o2_robot_angle, o2_spin_speed, o2_angle_range)
            else:
                o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle, 90, 270)
                if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                    o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle,
                                                            o2_closest_angle, o2_spin_speed)
                else:
                    o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 90, 2)

        # Deflection Defense
        elif self.o2_mode == o2_deflection_defense_mode:
            o2_run_pass = True
            o2_dest_x, o2_dest_y = 0, -0.35
            o2_angle_range = 8
            if not OffenseYellow2.inside_position_range_two(self, o2_robot_x, o2_robot_y, o2_dest_x, o2_dest_y, 0.02, 0.02):
                o2_left_speed, o2_right_speed = OffenseYellow2.drive_to_point_two(self, o2_left_speed, o2_right_speed, o2_dest_x, o2_dest_y, o2_robot_x,
                                                           o2_robot_y, o2_robot_angle, o2_spin_speed, o2_angle_range)
            else:
                o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle, 0, 180)
                if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                    o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle,
                                                            o2_closest_angle, o2_spin_speed)
                else:
                    o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 0, 2)

        # Goal Defense
        elif self.o2_mode == o2_goal_defense_mode:
            o2_dest_x, o2_dest_y = -0.16, -0.8
            o2_angle_range = 8
            if not OffenseYellow2.inside_position_range_two(self, o2_robot_x, o2_robot_y, o2_dest_x, o2_dest_y, 0.02, 0.02):
                o2_left_speed, o2_right_speed = OffenseYellow2.drive_to_point_two(self, o2_left_speed, o2_right_speed, o2_dest_x, o2_dest_y, o2_robot_x,
                                                           o2_robot_y, o2_robot_angle, o2_spin_speed, o2_angle_range)
            else:
                o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle, 90, 270)
                if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                    o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle, o2_closest_angle,
                                                            o2_spin_speed)
                else:
                    o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 90, 10)

        # Main Trap Ball
        elif self.o2_mode == o2_trap_ball_main_mode:
            o2_run_pass = True

            o2_angle_range = 2
            o2_spin_speed = 4

            if self.o2_trap_stage == 1:
                if o2_ball_y > -0.7 or not o2_offense2_x < o2_ball_x < o2_offense1_x or o2_ball_distance > 0.1 or o2_offense1_ball_distance > 0.1:
                    self.o2_trap_stage = -1
                else:
                    o2_shoot_angle = 140
                    o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle,
                                                                               (o2_shoot_angle + 360) % 360,
                                                                               (o2_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                        o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle,
                                                                    o2_closest_angle,
                                                                    o2_spin_speed)
                    else:
                        o2_move_speed = 0
                        o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 0, o2_move_speed)

                        if OffenseYellow2.inside_position_range_two(self, o2_offense1_x, o2_offense1_y, o2_offense1_x, -0.67, 0.01, 0.01) and \
                                OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                            self.o2_trap_stage = 2

            elif self.o2_trap_stage == 2:
                if o2_ball_y > -0.7 or o2_ball_distance > 0.1 or o2_offense1_ball_distance > 0.1:
                    self.o2_trap_stage = -1
                else:
                    o2_shoot_angle = 140
                    o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle,
                                                                               (o2_shoot_angle + 360) % 360,
                                                                               (o2_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                        o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle,
                                                                    o2_closest_angle,
                                                                    o2_spin_speed)
                    else:
                        o2_move_speed = 10
                        o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 0, o2_move_speed)

                if OffenseYellow2.inside_position_range_two(self, o2_ball_x, o2_ball_y, 0.56, -0.73, 0.01, 0.01):
                    self.o2_trap_stage = 3

            elif self.o2_trap_stage == 3:
                if o2_ball_y > -0.51 or o2_ball_x < 0.43:
                    self.o2_trap_stage = -1
                else:
                    o2_shoot_angle = 140
                    o2_dest_x, o2_dest_y = 0.47, -0.67
                    if not OffenseYellow2.inside_position_range_two(self, o2_robot_x, o2_robot_y, o2_dest_x, o2_dest_y, 0.005, 0.005):
                        o2_left_speed, o2_right_speed = OffenseYellow2.drive_to_point_two(self, o2_left_speed, o2_right_speed, o2_dest_x, o2_dest_y,
                                                                       o2_robot_x,
                                                                       o2_robot_y, o2_robot_angle, o2_spin_speed, o2_angle_range)
                    else:
                        o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle,
                                                                                   (o2_shoot_angle + 360) % 360,
                                                                                   (o2_shoot_angle + 360 + 180) % 360)
                        if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                            o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle,
                                                                        o2_closest_angle,
                                                                        o2_spin_speed)
                        else:
                            o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 0, 0)

                            o2_shoot_angle = 0
                            o2_dest_x, o2_dest_y = 0.5715, -0.595
                            o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_offense1_angle,
                                                                                       (o2_shoot_angle + 360) % 360,
                                                                                       (o2_shoot_angle + 360 + 180) % 360)
                            if OffenseYellow2.inside_position_range_two(self, o2_offense1_x, o2_offense1_y, o2_dest_x, o2_dest_y, 0.01, 0.01) and \
                                    OffenseYellow2.inside_angle_range_two(self, o2_offense1_angle, o2_closest_angle, o2_angle_range):
                                self.o2_trap_stage = 4

            elif self.o2_trap_stage == 4:
                if o2_ball_y > -0.51 or o2_ball_x < 0.43:
                    self.o2_trap_stage = -1
                else:
                    if not OffenseYellow2.inside_position_range_two(self, o2_ball_x, o2_ball_y, 0.63, -0.64, 0.01, 0.01):
                        o2_dest_x, o2_dest_y = o2_ball_x, o2_ball_y
                        o2_point_angle = (math.degrees(
                            math.atan2(o2_dest_x - o2_robot_x, o2_dest_y - o2_robot_y)) - o2_robot_angle + 360) % 360

                        o2_point_direction = o2_point_angle
                        if o2_point_direction > 180:
                            o2_point_direction = o2_point_direction - 360

                        o2_offset = 0

                        o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_point_angle, 0, 180)

                        if o2_closest_angle == 0:
                            self.o2_side = 1
                        elif o2_closest_angle == 180:
                            self.o2_side = -1

                        if self.o2_side == 1:
                            o2_move_angle = o2_point_direction + o2_offset
                        else:
                            if o2_point_direction + o2_offset >= 0:
                                o2_move_angle = o2_point_direction + o2_offset - 180
                            else:
                                o2_move_angle = 180 - abs(o2_point_direction + o2_offset)

                        o2_left_speed, o2_right_speed = OffenseYellow2.curve_drive_two(self, o2_move_angle, self.o2_side, 45)

                        if OffenseYellow2.inside_position_range_two(self, o2_ball_x, o2_ball_y, 0.58, -0.73, 0.01, 0.01) and \
                                OffenseYellow2.calc_distance_two(
                                    self, o2_ball_x, o2_ball_y, self.o2_last_ball_x, self.o2_last_ball_y) < math.pow(10, -5):
                            self.o2_trap_stage = 3
                        elif OffenseYellow2.inside_position_range_two(self, o2_ball_x, o2_ball_y, 0.63, -0.68, 0.01, 0.01) and \
                                OffenseYellow2.calc_distance_two(
                                    self, o2_ball_x, o2_ball_y, self.o2_last_ball_x, self.o2_last_ball_y) < math.pow(10, -5):
                            self.o2_trap_stage = 5

                    elif OffenseYellow2.inside_position_range_two(self, o2_robot_x, o2_robot_y, 0.57, -0.69, 0.01, 0.01):
                        self.o2_trap_stage = 5

            elif self.o2_trap_stage == 5:
                if o2_ball_x < 0.56 or not o2_offense1_y > o2_ball_y > o2_offense2_y:
                    self.o2_trap_stage = -1
                else:
                    o2_shoot_angle = 40
                    o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle,
                                                                               (o2_shoot_angle + 360) % 360,
                                                                               (o2_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                        o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle,
                                                                    o2_closest_angle,
                                                                    o2_spin_speed)
                    else:
                        o2_move_speed = 0
                        o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 0, o2_move_speed)

                        self.o2_trap_stage = 6

            elif self.o2_trap_stage == 6:
                if o2_ball_x < 0.56 or o2_ball_distance > 0.1 or o2_offense1_ball_distance > 0.1:
                    self.o2_trap_stage = -1
                else:
                    o2_shoot_angle = 40
                    o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle,
                                                                               (o2_shoot_angle + 360) % 360,
                                                                               (o2_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                        o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle,
                                                                    o2_closest_angle,
                                                                    o2_spin_speed)
                    else:
                        o2_move_speed = -10
                        o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 0, o2_move_speed)

                    if OffenseYellow2.inside_position_range_two(self, o2_ball_x, o2_ball_y, 0.625, 0.67, 0.01, 0.01):
                        self.o2_trap_stage = 7

            elif self.o2_trap_stage == 7:
                if o2_ball_x < 0.4 or o2_ball_y < 0.55:
                    self.o2_trap_stage = -1
                else:
                    o2_shoot_angle = 40
                    o2_dest_x, o2_dest_y = 0.57, 0.57
                    if not OffenseYellow2.inside_position_range_two(self, o2_robot_x, o2_robot_y, o2_dest_x, o2_dest_y, 0.005, 0.005):
                        o2_left_speed, o2_right_speed = OffenseYellow2.drive_to_point_two(self, o2_left_speed, o2_right_speed, o2_dest_x, o2_dest_y,
                                                                       o2_robot_x,
                                                                       o2_robot_y, o2_robot_angle, o2_spin_speed, o2_angle_range)
                    else:
                        o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle,
                                                                                   (o2_shoot_angle + 360) % 360,
                                                                                   (o2_shoot_angle + 360 + 180) % 360)
                        if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                            o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle,
                                                                        o2_closest_angle,
                                                                        o2_spin_speed)
                        else:
                            o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 0, 0)

                            o2_shoot_angle = 90
                            o2_dest_x, o2_dest_y = 0.495, 0.672
                            o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_offense1_angle,
                                                                                       (o2_shoot_angle + 360) % 360,
                                                                                       (o2_shoot_angle + 360 + 180) % 360)
                            if OffenseYellow2.inside_position_range_two(self, o2_offense1_x, o2_offense1_y, o2_dest_x, o2_dest_y, 0.01, 0.01) and \
                                    OffenseYellow2.inside_angle_range_two(self, o2_offense1_angle, o2_closest_angle, o2_angle_range):
                                self.o2_trap_stage = 8

            elif self.o2_trap_stage == 8:
                if o2_ball_x < 0.4 or o2_ball_y < 0.55:
                    self.o2_trap_stage = -1
                else:
                    if not OffenseYellow2.inside_position_range_two(self, o2_ball_x, o2_ball_y, 0.54, 0.73, 0.01, 0.01):
                        o2_dest_x, o2_dest_y = o2_ball_x, o2_ball_y
                        o2_point_angle = (math.degrees(
                            math.atan2(o2_dest_x - o2_robot_x, o2_dest_y - o2_robot_y)) - o2_robot_angle + 360) % 360

                        o2_point_direction = o2_point_angle
                        if o2_point_direction > 180:
                            o2_point_direction = o2_point_direction - 360

                        o2_offset = 0

                        o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_point_angle, 0, 180)

                        if o2_closest_angle == 0:
                            self.o2_side = 1
                        elif o2_closest_angle == 180:
                            self.o2_side = -1

                        if self.o2_side == 1:
                            o2_move_angle = o2_point_direction + o2_offset
                        else:
                            if o2_point_direction + o2_offset >= 0:
                                o2_move_angle = o2_point_direction + o2_offset - 180
                            else:
                                o2_move_angle = 180 - abs(o2_point_direction + o2_offset)

                        o2_left_speed, o2_right_speed = OffenseYellow2.curve_drive_two(self, o2_move_angle, self.o2_side, 45)

                        if OffenseYellow2.inside_position_range_two(self, o2_ball_x, o2_ball_y, 0.63, 0.68, 0.01,
                                                      0.01) and OffenseYellow2.calc_distance_two(
                            self, o2_ball_x, o2_ball_y, self.o2_last_ball_x, self.o2_last_ball_y) < math.pow(10, -5):
                            self.o2_trap_stage = 7
                        elif OffenseYellow2.inside_position_range_two(self, o2_ball_x, o2_ball_y, 0.58, 0.73, 0.01, 0.01) and \
                                OffenseYellow2.calc_distance_two(
                                    self, o2_ball_x, o2_ball_y, self.o2_last_ball_x, self.o2_last_ball_y) < math.pow(10,
                                                                                                         -5) and o2_ball_distance < 0.05:
                            self.o2_trap_stage = 9

                    elif OffenseYellow2.inside_position_range_two(self, o2_robot_x, o2_robot_y, 0.59, 0.67, 0.01, 0.01):
                        self.o2_trap_stage = 9

            elif self.o2_trap_stage == 9:
                if o2_ball_y < 0.69 or not o2_offense2_x > o2_ball_x > o2_offense1_x:
                    self.o2_trap_stage = -1
                else:
                    o2_shoot_angle = 140
                    o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle,
                                                                               (o2_shoot_angle + 360) % 360,
                                                                               (o2_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                        o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle,
                                                                    o2_closest_angle,
                                                                    o2_spin_speed)
                    else:
                        o2_move_speed = 0
                        o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 0, o2_move_speed)

                        self.o2_trap_stage = 10

            elif self.o2_trap_stage == 10:
                if o2_ball_x < 0.1 or not 0.77 > o2_ball_y > 0.68 or o2_ball_distance > 0.1 or o2_offense1_ball_distance > 0.1:
                    self.o2_trap_stage = -1
                else:
                    o2_shoot_angle = 140
                    o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle,
                                                                               (o2_shoot_angle + 360) % 360,
                                                                               (o2_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                        o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle,
                                                                    o2_closest_angle,
                                                                    o2_spin_speed)
                    else:
                        o2_move_speed = -10
                        o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 0, o2_move_speed)

        # Support Trap Ball
        elif self.o2_mode == o2_trap_ball_support_mode:
            o2_run_pass = True

            o2_angle_range = 2
            o2_spin_speed = 4

            o2_robot_radius = 0.0375
            o2_ball_radius = 0.021

            if o2_trap_stage == 1:
                o2_shoot_angle = 90
                o2_dest_x, o2_dest_y = o2_robot_x, -0.672
                if not OffenseYellow2.inside_position_range_two(self, o2_robot_x, o2_robot_y, o2_dest_x, o2_dest_y, 0.01, 0.01):
                    o2_left_speed, o2_right_speed = OffenseYellow2.drive_to_point_two(self, o2_left_speed, o2_right_speed, o2_dest_x, o2_dest_y,
                                                                   o2_robot_x,
                                                                   o2_robot_y, o2_robot_angle, o2_spin_speed, o2_angle_range)
                else:
                    o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle,
                                                                               (o2_shoot_angle + 360) % 360,
                                                                               (o2_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                        o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle,
                                                                    o2_closest_angle,
                                                                    o2_spin_speed)
                    else:
                        o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 0, 0)

            elif o2_trap_stage == 2:
                o2_shoot_angle = 90
                o2_dest_x, o2_dest_y = o2_ball_x - o2_robot_radius - o2_ball_radius + 0.008, -0.672
                if not OffenseYellow2.inside_position_range_two(self, o2_robot_x, o2_robot_y, o2_dest_x, o2_dest_y, 0.005, 0.005):
                    o2_left_speed, o2_right_speed = OffenseYellow2.drive_to_point_two(self, o2_left_speed, o2_right_speed, o2_dest_x, o2_dest_y,
                                                                   o2_robot_x,
                                                                   o2_robot_y, o2_robot_angle, o2_spin_speed, o2_angle_range)
                else:
                    o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle,
                                                                               (o2_shoot_angle + 360) % 360,
                                                                               (o2_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                        o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle,
                                                                    o2_closest_angle, o2_spin_speed)
                    else:
                        o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 0, 0)

            elif o2_trap_stage == 3:
                if OffenseYellow2.inside_position_range_two(self, o2_offense1_x, o2_offense1_y, -0.47, -0.67, 0.005, 0.005):
                    o2_shoot_angle = 0
                    o2_dest_x, o2_dest_y = -0.5715, -0.595
                    if not OffenseYellow2.inside_position_range_two(self, o2_robot_x, o2_robot_y, o2_dest_x, o2_dest_y, 0.005, 0.005):
                        o2_left_speed, o2_right_speed = OffenseYellow2.drive_to_point_two(self, o2_left_speed, o2_right_speed, o2_dest_x, o2_dest_y,
                                                                       o2_robot_x,
                                                                       o2_robot_y, o2_robot_angle, o2_spin_speed, o2_angle_range)
                    else:
                        o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle,
                                                                                   (o2_shoot_angle + 360) % 360,
                                                                                   (o2_shoot_angle + 360 + 180) % 360)
                        if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                            o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle,
                                                                        o2_closest_angle,
                                                                        o2_spin_speed)
                        else:
                            o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 0, 0)

            elif o2_trap_stage == 4:
                o2_shoot_angle = 0
                o2_dest_x, o2_dest_y = -0.5715, -0.595
                if not OffenseYellow2.inside_position_range_two(self, o2_robot_x, o2_robot_y, o2_dest_x, o2_dest_y, 0.005, 0.005):
                    o2_left_speed, o2_right_speed = OffenseYellow2.drive_to_point_two(self, o2_left_speed, o2_right_speed, o2_dest_x, o2_dest_y,
                                                                   o2_robot_x,
                                                                   o2_robot_y, o2_robot_angle, o2_spin_speed, o2_angle_range)
                else:
                    o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle,
                                                                               (o2_shoot_angle + 360) % 360,
                                                                               (o2_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                        o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle,
                                                                    o2_closest_angle,
                                                                    o2_spin_speed)
                    else:
                        o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 0, 2)

            elif o2_trap_stage == 5:
                o2_shoot_angle = 0
                o2_dest_x, o2_dest_y = -0.5715, -0.57
                if not OffenseYellow2.inside_position_range_two(self, o2_robot_x, o2_robot_y, o2_dest_x, o2_dest_y, 0.005, 0.005):
                    o2_left_speed, o2_right_speed = OffenseYellow2.drive_to_point_two(self, o2_left_speed, o2_right_speed, o2_dest_x, o2_dest_y,
                                                                   o2_robot_x,
                                                                   o2_robot_y, o2_robot_angle, o2_spin_speed, o2_angle_range)
                else:
                    o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle,
                                                                               (o2_shoot_angle + 360) % 360,
                                                                               (o2_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                        o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle,
                                                                    o2_closest_angle,
                                                                    o2_spin_speed)
                    else:
                        o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 0, 0)

            elif o2_trap_stage == 6:
                o2_shoot_angle = 0
                o2_dest_x, o2_dest_y = -0.5715, o2_ball_y + o2_robot_radius + o2_ball_radius - 0.008
                if not OffenseYellow2.inside_position_range_two(self, o2_robot_x, o2_robot_y, o2_dest_x, o2_dest_y, 0.005, 0.005):
                    o2_left_speed, o2_right_speed = OffenseYellow2.drive_to_point_two(self, o2_left_speed, o2_right_speed, o2_dest_x, o2_dest_y,
                                                                   o2_robot_x,
                                                                   o2_robot_y, o2_robot_angle, o2_spin_speed, o2_angle_range)
                else:
                    o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle,
                                                                               (o2_shoot_angle + 360) % 360,
                                                                               (o2_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                        o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle,
                                                                    o2_closest_angle, o2_spin_speed)
                    else:
                        o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 0, 0)

            elif o2_trap_stage == 7:
                if OffenseYellow2.inside_position_range_two(self, o2_offense1_x, o2_offense1_y, -0.57, 0.57, 0.005, 0.005):
                    o2_shoot_angle = 90
                    o2_dest_x, o2_dest_y = -0.495, 0.672
                    if not OffenseYellow2.inside_position_range_two(self, o2_robot_x, o2_robot_y, o2_dest_x, o2_dest_y, 0.005, 0.005):
                        o2_left_speed, o2_right_speed = OffenseYellow2.drive_to_point_two(self, o2_left_speed, o2_right_speed, o2_dest_x, o2_dest_y,
                                                                       o2_robot_x,
                                                                       o2_robot_y, o2_robot_angle, o2_spin_speed, o2_angle_range)
                    else:
                        o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle,
                                                                                   (o2_shoot_angle + 360) % 360,
                                                                                   (o2_shoot_angle + 360 + 180) % 360)
                        if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                            o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle,
                                                                        o2_closest_angle,
                                                                        o2_spin_speed)
                        else:
                            o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 0, 0)

            elif o2_trap_stage == 8:
                o2_shoot_angle = 90
                o2_dest_x, o2_dest_y = -0.495, 0.672
                if not OffenseYellow2.inside_position_range_two(self, o2_robot_x, o2_robot_y, o2_dest_x, o2_dest_y, 0.005, 0.005):
                    o2_left_speed, o2_right_speed = OffenseYellow2.drive_to_point_two(self, o2_left_speed, o2_right_speed, o2_dest_x, o2_dest_y,
                                                                   o2_robot_x,
                                                                   o2_robot_y, o2_robot_angle, o2_spin_speed, o2_angle_range)
                else:
                    o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle,
                                                                               (o2_shoot_angle + 360) % 360,
                                                                               (o2_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                        o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle,
                                                                    o2_closest_angle,
                                                                    o2_spin_speed)
                    else:
                        o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 90, 2)

            elif o2_trap_stage == 9:
                o2_shoot_angle = 90
                o2_dest_x, o2_dest_y = -0.495, 0.672
                if not OffenseYellow2.inside_position_range_two(self, o2_robot_x, o2_robot_y, o2_dest_x, o2_dest_y, 0.005, 0.005):
                    o2_left_speed, o2_right_speed = OffenseYellow2.drive_to_point_two(self, o2_left_speed, o2_right_speed, o2_dest_x, o2_dest_y,
                                                                   o2_robot_x,
                                                                   o2_robot_y, o2_robot_angle, o2_spin_speed, o2_angle_range)
                else:
                    o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle,
                                                                               (o2_shoot_angle + 360) % 360,
                                                                               (o2_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                        o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle,
                                                                    o2_closest_angle,
                                                                    o2_spin_speed)
                    else:
                        o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 0, 0)

            elif o2_trap_stage == 10:
                o2_shoot_angle = 90
                o2_dest_x, o2_dest_y = o2_ball_x + o2_robot_radius + o2_ball_radius - 0.008, 0.672
                if not OffenseYellow2.inside_position_range_two(self, o2_robot_x, o2_robot_y, o2_dest_x, o2_dest_y, 0.005, 0.005):
                    o2_left_speed, o2_right_speed = OffenseYellow2.drive_to_point_two(self, o2_left_speed, o2_right_speed, o2_dest_x, o2_dest_y,
                                                                   o2_robot_x,
                                                                   o2_robot_y, o2_robot_angle, o2_spin_speed, o2_angle_range)
                else:
                    o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle,
                                                                               (o2_shoot_angle + 360) % 360,
                                                                               (o2_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle, o2_angle_range):
                        o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle,
                                                                    o2_closest_angle, o2_spin_speed)
                    else:
                        o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, 0, 0)

        o2_distance_range = 0.1
        o2_temp_pred_x, o2_temp_pred_y = 2 * o2_ball_x - self.o2_last_ball_x, 2 * o2_ball_y - self.o2_last_ball_y
        o2_temp_pred_distance = OffenseYellow2.calc_distance_two(self, o2_robot_x, o2_robot_y, o2_temp_pred_x, o2_temp_pred_y)
        if not o2_run_pass and o2_ball_y < 0 and -0.23 < o2_robot_x < 0.23 and o2_robot_y < 0 and (
                (o2_ball_distance < o2_distance_range and o2_ball_y < o2_robot_y) or (
                o2_temp_pred_distance < o2_distance_range and o2_temp_pred_y < o2_robot_y) or (
                        OffenseYellow2.calc_distance_two(self, o2_robot_x, o2_robot_y, self.o2_last_ball_x,
                                             self.o2_last_ball_y) < o2_distance_range and self.o2_last_ball_y < o2_robot_y)):
            o2_left_speed, o2_right_speed = 0, 0

        self.o2_last_ball_x, self.o2_last_ball_y = o2_ball_x, o2_ball_y

        self.o2_last_mode = self.o2_mode

        o2_robot.left_motor.setVelocity(o2_left_speed)
        o2_robot.right_motor.setVelocity(o2_right_speed)

    def reset_strategy_two(self):
        self.o2_mode = 0
        self.o2_side = 1
        self.o2_goal_x, self.o2_goal_y = 0, 0.75
        self.o2_off_angle = 25
        self.o2_last_ball_x, self.o2_last_ball_y = 0, 0
        self.o2_last_mode = 0
        self.o2_limit_x = -0.12
        self.o2_timer = 0
        self.o2_push_ball = False
        self.o2_trap_stage = -1
        self.o2_double_push_x, self.o2_double_push_y = 0, 0
        self.o2_wall_dest_x, self.o2_wall_dest_y = 0.44, -0.7
        self.o2_team = "Y"

    # Add any additional functions here
    def curve_drive_two(self, o2_dir, o2_side, o2_ang=65):
        # Side (+: front, -: back)
        o2_left_speed = o2_side * -10 * (o2_ang + o2_dir * o2_side) / o2_ang
        o2_right_speed = o2_side * -10 * (o2_ang - o2_dir * o2_side) / o2_ang

        if o2_left_speed > 10:
            o2_left_speed = 10
        elif o2_left_speed < -10:
            o2_left_speed = -10
        if o2_right_speed > 10:
            o2_right_speed = 10
        elif o2_right_speed < -10:
            o2_right_speed = -10

        return o2_left_speed, o2_right_speed

    def pick_direction_two(self, o2_robot_angle, o2_angle_a, o2_angle_b):
        o2_angle_a_left = (o2_robot_angle - o2_angle_a + 360) % 360
        o2_angle_a_right = (o2_angle_a - o2_robot_angle + 360) % 360
        o2_angle_b_left = (o2_robot_angle - o2_angle_b + 360) % 360
        o2_angle_b_right = (o2_angle_b - o2_robot_angle + 360) % 360

        o2_angle_smallest = min([o2_angle_a_left, o2_angle_a_right, o2_angle_b_left, o2_angle_b_right])

        if o2_angle_smallest == o2_angle_a_left:
            return o2_angle_a, -o2_angle_a_left
        elif o2_angle_smallest == o2_angle_a_right:
            return o2_angle_a, o2_angle_a_right
        elif o2_angle_smallest == o2_angle_b_left:
            return o2_angle_b, -o2_angle_b_left
        else:
            return o2_angle_b, -o2_angle_b_right

    def drive_to_point_two(self, o2_left_speed, o2_right_speed, o2_dest_x, o2_dest_y, o2_robot_x, o2_robot_y, o2_robot_angle, o2_spin_speed,
                o2_angle_range):
        o2_closest_angle, o2_absolute_angle = OffenseYellow2.relative_coords_two(self, o2_robot_angle, o2_robot_x, o2_robot_y, o2_dest_x, o2_dest_y)
        if not OffenseYellow2.inside_angle_range_two(self, o2_robot_angle, o2_closest_angle,
                                                       o2_angle_range + O2_STEERING_DEADBAND):
            o2_left_speed, o2_right_speed = OffenseYellow2.turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle, o2_closest_angle,
                                                    o2_spin_speed)
        else:
            if self.o2_team == "B":
                o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, o2_absolute_angle, -10)
            else:
                o2_left_speed, o2_right_speed = OffenseYellow2.straight_drive_two(self, o2_robot_angle, o2_absolute_angle, 10)
        return o2_left_speed, o2_right_speed

    def relative_coords_two(self, o2_robot_angle, o2_robot_x, o2_robot_y, o2_dest_x, o2_dest_y):
        o2_absolute_angle = (math.degrees(math.atan2(o2_robot_x - o2_dest_x, o2_robot_y - o2_dest_y))) % 360
        o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle, o2_absolute_angle,
                                                               (o2_absolute_angle + 180) % 360)
        return o2_closest_angle, o2_absolute_angle

    def turn_to_heading_two(self, o2_left_speed, o2_right_speed, o2_robot_angle, o2_angle, o2_spin_speed):
        o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_robot_angle, o2_angle, (o2_angle + 180) % 360)
        # The closest difference is relative to the robot
        if o2_closest_difference > 0:
            o2_left_speed = -o2_spin_speed
            o2_right_speed = o2_spin_speed
        elif o2_closest_difference < 0:
            o2_left_speed = o2_spin_speed
            o2_right_speed = -o2_spin_speed
        return o2_left_speed, o2_right_speed

    def straight_drive_two(self, o2_robot_angle, o2_target_angle, o2_move_speed):
        o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_target_angle, o2_robot_angle,
                                                               (o2_robot_angle + 180) % 360)
        if o2_closest_angle == o2_robot_angle:
            o2_left_speed = o2_move_speed
            o2_right_speed = o2_move_speed
        else:
            o2_left_speed = -o2_move_speed
            o2_right_speed = -o2_move_speed
        return o2_left_speed, o2_right_speed

    def inside_position_range_two(self, o2_pos_x, o2_pos_y, o2_dest_x, o2_dest_y, o2_range_x=0.2, o2_range_y=0.2, o2_range_x_up=None, o2_range_x_down=None,
                      o2_range_y_up=None, o2_range_y_down=None):
        if o2_range_x_up is None:
            o2_range_x_up = o2_range_x
        if o2_range_x_down is None:
            o2_range_x_down = o2_range_x
        if o2_range_y_up is None:
            o2_range_y_up = o2_range_y
        if o2_range_y_down is None:
            o2_range_y_down = o2_range_y
        if o2_pos_x > o2_dest_x + o2_range_x_up or o2_pos_x < o2_dest_x - o2_range_x_down or o2_pos_y > o2_dest_y + o2_range_y_up or o2_pos_y < o2_dest_y - o2_range_y_down:
            return False
        return True

    def inside_angle_range_two(self, o2_angle, o2_target_angle, o2_angle_range=15, o2_angle_left_range=None, o2_angle_right_range=None,
                   o2_both_sides=True):
        if o2_angle_left_range is None:
            o2_angle_left_range = o2_angle_range
        if o2_angle_right_range is None:
            o2_angle_right_range = o2_angle_range
        if o2_both_sides:
            o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_angle, o2_target_angle,
                                                                   (o2_target_angle + 180) % 360)
        else:
            o2_closest_angle, o2_closest_difference = OffenseYellow2.pick_direction_two(self, o2_angle, o2_target_angle, o2_target_angle)
        if o2_angle_right_range > o2_closest_difference > -o2_angle_left_range:
            return True
        return False

    def scale_linear_two(self, o2_raw_value, o2_min_old_value, o2_max_old_value, o2_min_new_value, o2_max_new_value):
        if o2_raw_value >= o2_max_old_value:
            return o2_max_new_value
        elif o2_raw_value <= o2_min_old_value:
            return o2_min_new_value
        else:
            o2_slope = (o2_max_new_value - o2_min_new_value) / (o2_max_old_value - o2_min_old_value)
            o2_intercept = o2_slope * o2_min_old_value - o2_min_new_value
            o2_new_value = o2_slope * o2_raw_value + o2_intercept
        return o2_new_value

    def solve_target_point_two(self, o2_input_value, o2_point_a_x, o2_point_a_y, o2_point_b_x, o2_point_b_y, o2_find_y=True):
        o2_point_a_x, o2_point_a_y = -o2_point_a_x, o2_point_a_y
        o2_point_b_x, o2_point_b_y = -o2_point_b_x, o2_point_b_y
        o2_input_value = -o2_input_value

        o2_point_angle = math.atan2(o2_point_b_y - o2_point_a_y, o2_point_b_x - o2_point_a_x)
        o2_point_slope = math.tan(o2_point_angle)
        o2_point_intercept = o2_point_a_y - o2_point_slope * o2_point_a_x

        if not o2_find_y:
            return -(o2_input_value - o2_point_intercept) / o2_point_slope, o2_input_value
        return -o2_input_value, o2_point_slope * o2_input_value + o2_point_intercept

    def solve_attack_point_two(self, o2_point_x, o2_ball_x, o2_ball_y, o2_last_ball_x, o2_last_ball_y, o2_offset_front=0.0,
                        o2_offset_back=0.0):
        o2_ball_radius = 0.021
        o2_robot_short_radius = 0.0375  # 0.0383
        o2_robot_long_radius = 0.0475

        o2_main_x, o2_main_y = OffenseYellow2.solve_target_point_two(self, o2_point_x, o2_ball_x, o2_ball_y, o2_last_ball_x, o2_last_ball_y)
        o2_main_x, o2_main_y = o2_main_y, -o2_main_x

        o2_point_y = -o2_point_x
        o2_ball_x, o2_ball_y = o2_ball_y, -o2_ball_x
        o2_last_ball_x, o2_last_ball_y = o2_last_ball_y, -o2_last_ball_x

        o2_theta = math.atan2(o2_last_ball_y - o2_ball_y, o2_last_ball_x - o2_ball_x)
        o2_abs_theta = abs(o2_theta)

        if abs(o2_theta) < O2_ATTACK_EPSILON:
            return None, None

        # Finding first gap length
        o2_gap_1 = o2_ball_radius / math.sin(o2_abs_theta)
        o2_length_1 = o2_gap_1 * math.tan(o2_abs_theta)

        # Finding second gap length
        o2_slope_original = math.tan(o2_theta)
        o2_intercept_original = (o2_ball_y - o2_slope_original * o2_ball_x)
        if o2_point_y > o2_ball_y:
            o2_intercept_original += o2_length_1
        else:
            o2_intercept_original -= o2_length_1
        o2_slope_perpendicular = -1 / o2_slope_original
        o2_intercept_perpendicular = (o2_ball_y - o2_slope_perpendicular * o2_ball_x)

        o2_left_matrix = np.array([[-o2_slope_original, 1], [-o2_slope_perpendicular, 1]])
        o2_right_matrix = np.array([o2_intercept_original, o2_intercept_perpendicular])
        o2_solution = np.linalg.solve(o2_left_matrix, o2_right_matrix).tolist()
        o2_solution_x, o2_solution_y = o2_solution[0], o2_solution[1]

        o2_temp_length = math.sqrt(math.pow((o2_main_x + o2_gap_1) - o2_solution_x, 2) + math.pow(o2_main_y - o2_solution_y, 2))
        o2_length_2 = o2_temp_length * math.sin(o2_abs_theta)

        if self.o2_team == "B":
            if o2_ball_x > o2_main_x:
                if o2_length_2 >= o2_robot_long_radius:
                    o2_gap_2 = o2_robot_long_radius / math.tan(o2_abs_theta)
                    o2_dest_x, o2_dest_y = o2_main_x + o2_gap_1 + o2_gap_2 + o2_robot_short_radius + o2_offset_back, o2_point_y
                else:
                    o2_dest_x, o2_dest_y = o2_ball_x + o2_ball_radius + o2_robot_short_radius + o2_offset_back, o2_point_y
            else:
                o2_dest_x, o2_dest_y = o2_main_x + o2_ball_radius + o2_robot_short_radius + o2_offset_front, o2_point_y
        else:
            if o2_ball_x < o2_main_x:
                if o2_length_2 >= o2_robot_long_radius:
                    o2_gap_2 = o2_robot_long_radius / math.tan(o2_abs_theta)
                    o2_dest_x, o2_dest_y = o2_main_x - o2_gap_1 - o2_gap_2 - o2_robot_short_radius - o2_offset_back, o2_point_y
                else:
                    o2_dest_x, o2_dest_y = o2_ball_x - o2_ball_radius - o2_robot_short_radius - o2_offset_back, o2_point_y
            else:
                o2_dest_x, o2_dest_y = o2_main_x - o2_ball_radius - o2_robot_short_radius - o2_offset_front, o2_point_y

        return -o2_dest_y, o2_dest_x

    def calc_distance_two(self, o2_point_a_x, o2_point_a_y, o2_point_b_x, o2_point_b_y):
        return math.sqrt(math.pow(o2_point_a_x - o2_point_b_x, 2) + math.pow(o2_point_a_y - o2_point_b_y, 2))

