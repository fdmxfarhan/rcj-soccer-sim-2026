from rcj_soccer_robot import TIME_STEP
import numpy as np
import math

o1_goal_attack_mode = 0
o1_ambush_attack_mode = 1
o1_stationing_attack_mode = 2
o1_double_pushing_mode = 3
o1_midfield_defense_mode = 4
o1_wall_defense_mode = 5
o1_deflection_defense_mode = 6
o1_goal_defense_mode = 7
o1_trap_ball_main_mode = 8
o1_trap_ball_support_mode = 9


# Small control tolerances: deliberately conservative to avoid changing the main strategy.
O1_MODE_HYSTERESIS = 0.015
O1_STEERING_DEADBAND = 1.0
O1_ATTACK_EPSILON = 1e-4
O1_DUPLICATE_CHASE_MARGIN = 0.08
O1_SUPPORT_X_LIMIT = 0.34
O1_SUPPORT_Y_MIN = -0.35
O1_SUPPORT_Y_MAX = 0.25
O1_TEAMMATE_SEPARATION_DISTANCE = 0.18
O1_TEAMMATE_SEPARATION_OFFSET = 0.10

class OffenseYellow1:
    o1_mode = 0
    o1_side = 1
    o1_goal_x, o1_goal_y = 0, 0.75
    o1_off_angle = 25
    o1_last_ball_x, o1_last_ball_y = 0, 0
    o1_last_mode = 0
    o1_limit_x = 0.12
    o1_timer = 0
    o1_push_ball = False
    o1_trap_stage = -1
    o1_double_push_x, o1_double_push_y = 0, 0
    o1_wall_dest_x, o1_wall_dest_y = -0.44, -0.7
    o1_team = "Y"

    # Run Blue
    def execute_strategy(self, o1_robot, o1_data, o1_lop1, o1_lop2, o1_roles, o1_trap_stage):
        o1_left_speed, o1_right_speed = 0, 0

        o1_robot_angle = o1_data.getRobotHeading()
        o1_robot_x, o1_robot_y = o1_data.getRobotPos()
        o1_ball_x, o1_ball_y = o1_data.getBallPos()
        o1_ball_distance = o1_data.getBallDistance()
        o1_ball_pred_x, o1_ball_pred_y = o1_data.getBallPredPos()
        o1_ball_pred_angle = o1_data.getBallPredAngle()
        o1_offense1_x, o1_offense1_y = o1_data.getRobotPosSearch(o1_roles.getRoleToRobot(1))
        o1_offense2_x, o1_offense2_y = o1_data.getRobotPosSearch(o1_roles.getRoleToRobot(2))
        o1_defense_x, o1_defense_y = o1_data.getRobotPosSearch(o1_roles.getRoleToRobot(3))
        o1_offense2_angle = o1_data.getHeadingSearch(o1_roles.getRoleToRobot(2))
        o1_offense1_ball_distance = o1_data.getBallDistanceSearch(o1_roles.getRoleToRobot(1))
        o1_offense2_ball_distance = o1_data.getBallDistanceSearch(o1_roles.getRoleToRobot(2))
        o1_detect = o1_data.getTeamDetect()

        # C5/C6: use team ball awareness to avoid two offensive robots
        # independently committing to the same ball.  Defense remains free
        # to observe the ball without taking the offensive chase slot.
        o1_primary_observer = o1_data.getPrimaryBallObserver()
        o1_team_ball_state = o1_data.getTeamBallState()
        o1_other_offense_id = o1_roles.getRoleToRobot(2)
        o1_yield_to_teammate = (
            o1_team_ball_state == "KNOWN" and
            o1_primary_observer == o1_other_offense_id and
            o1_offense2_ball_distance + O1_DUPLICATE_CHASE_MARGIN < o1_offense1_ball_distance
        )

        o1_ball_lack_progress_short = o1_lop1["ball"]

        o1_spin_speed = 6
        o1_angle_range = 15

        o1_run_pass = False

        self.o1_goal_x, self.o1_goal_y = 0, 0.75

        if o1_ball_y > 0.65:
            self.o1_limit_x = 0.15
        else:
            self.o1_limit_x = 0.12

        # Offense 1 Covers Upper-Half
        if o1_detect and not o1_data.is_flipped:
            if o1_robot_y > -0.6 and o1_ball_y < -0.6 and o1_ball_x < 0:
                o1_offset = 0.15
                if o1_defense_y < 0.62:
                    self.o1_wall_dest_x, self.o1_wall_dest_y = -0.44, -0.7
                elif o1_defense_x > -0.25:
                    self.o1_wall_dest_x, self.o1_wall_dest_y = -0.25 - o1_offset, -0.7
                elif o1_defense_x < -0.4:
                    self.o1_wall_dest_x, self.o1_wall_dest_y = -0.4 - o1_offset, -0.7
                else:
                    self.o1_wall_dest_x, self.o1_wall_dest_y = o1_defense_x - o1_offset, -0.7
            if self.o1_trap_stage != -1 or (
                    o1_offense1_y < -0.67 and o1_offense2_y < -0.67 and o1_ball_y < -0.7 and o1_offense2_x < o1_ball_x < o1_offense1_x and o1_ball_x < -0.2 and o1_offense1_y > o1_ball_y and o1_offense2_y > o1_ball_y and OffenseYellow1.calc_distance(
                self, o1_offense1_x, o1_offense1_y, o1_offense2_x, o1_offense2_y) < 0.16):
                self.o1_mode = o1_trap_ball_main_mode
                if o1_offense1_y < -0.67 and o1_offense2_y < -0.67 and o1_ball_y < -0.7 and o1_offense2_x < o1_ball_x < o1_offense1_x and o1_ball_x < -0.2 and o1_offense1_y > o1_ball_y and o1_offense2_y > o1_ball_y and OffenseYellow1.calc_distance(
                        self, o1_offense1_x, o1_offense1_y, o1_offense2_x, o1_offense2_y) < 0.16 and self.o1_trap_stage == -1:
                    self.o1_trap_stage = 1
            elif o1_trap_stage != -1:
                self.o1_mode = o1_trap_ball_support_mode
            elif o1_ball_y < -0.66 and o1_ball_x < 0:
                self.o1_mode = o1_wall_defense_mode
            elif o1_ball_y < -0.64 and o1_ball_x <= 0.38 and (o1_ball_x >= 0.25 and o1_robot_y < o1_ball_y):
                self.o1_mode = o1_goal_defense_mode
            elif o1_ball_y < -0.64 and not 0.25 > o1_ball_x > -0.25:
                self.o1_mode = o1_midfield_defense_mode
            elif o1_ball_y < -0.53 and 0.25 > o1_ball_x > -0.25 and o1_defense_y < -0.59:
                self.o1_mode = o1_deflection_defense_mode
            elif o1_ball_lack_progress_short and o1_ball_x > self.o1_limit_x:
                if OffenseYellow1.inside_angle_range(self, o1_offense2_angle, 125, o1_angle_range, 20,
                                           35) and 0.42 > o1_ball_x > 0.12 and o1_ball_y > 0.65 and \
                        0.47 > o1_offense2_x > 0.12 and o1_offense2_y > 0.64 and o1_robot_y > 0.3:
                    self.o1_mode = o1_double_pushing_mode
                else:
                    self.o1_mode = o1_stationing_attack_mode
            else:
                o1_goal_condition = (o1_ball_x <= self.o1_limit_x and
                                     (o1_ball_y >= 0.22 or
                                      o1_offense1_ball_distance < o1_offense2_ball_distance or
                                      o1_ball_x < -self.o1_limit_x))
                o1_goal_hold = (self.o1_mode == o1_goal_attack_mode and
                                o1_ball_x <= self.o1_limit_x + O1_MODE_HYSTERESIS)
                if ((self.o1_push_ball is True and o1_ball_x <= 0.2) or
                        o1_goal_condition or o1_goal_hold):
                    self.o1_mode = o1_goal_attack_mode
                    if self.o1_last_mode == o1_ambush_attack_mode:
                        self.o1_push_ball = True
                else:
                    self.o1_push_ball = False
                    self.o1_mode = o1_ambush_attack_mode
            self.o1_timer = 0
        else:
            self.o1_mode = o1_stationing_attack_mode
            self.o1_timer += (TIME_STEP / 1000)

        # C6: the observing/closer offensive teammate owns the active chase.
        # Keep all special defensive/trap modes intact; only replace normal
        # attacking commitment with support positioning.
        if o1_yield_to_teammate and self.o1_mode == o1_goal_attack_mode:
            self.o1_mode = o1_stationing_attack_mode

        # Ball Chasing and Shooting
        if self.o1_mode == o1_goal_attack_mode:
            if OffenseYellow1.inside_position_range(self, o1_ball_x, o1_ball_y, 0, 0, 0.1,
                                          0.1) and o1_ball_distance < 0.15 and o1_robot_y < o1_ball_y and OffenseYellow1.calc_distance(
                    self, o1_ball_x, o1_ball_y, self.o1_last_ball_x, self.o1_last_ball_y) < math.pow(10, -5) * 5:
                self.o1_goal_x, self.o1_goal_y = -0.05, 0.75
            elif o1_ball_x < -0.25 and o1_ball_y < 0:
                self.o1_goal_x, self.o1_goal_y = -0.05, 0.75
            elif o1_ball_x > 0.25 and o1_ball_y < 0:
                self.o1_goal_x, self.o1_goal_y = 0.05, 0.75
            else:
                self.o1_goal_x, self.o1_goal_y = -0.03, 0.75

            # RECHECK
            o1_goal_angle = (math.degrees(math.atan2(self.o1_goal_x - o1_ball_pred_x, self.o1_goal_y - o1_ball_pred_y)) + 360) % 360

            if o1_ball_y > 0.6 and (o1_ball_x > 0.35 or o1_ball_x < -0.35):
                o1_goal_angle = 65

            if (o1_ball_x > 0.52 or o1_ball_x < -0.52) and -0.54 < o1_ball_y < 0.54:
                o1_goal_angle = 0
            elif o1_ball_x > 0.55 or o1_ball_x < -0.55:
                o1_goal_angle = 0

            # RECHECK
            if 0 < o1_ball_y < 0.5 and -0.4 < o1_ball_x < 0.4:
                o1_goal_temp_angle = math.atan2(abs(self.o1_goal_x - o1_ball_pred_x), abs(self.o1_goal_y - o1_ball_pred_y))
                o1_goal_temp_distance = 0.015
                o1_ball_temp_x = o1_ball_pred_x - math.sin(o1_goal_temp_angle) * o1_goal_temp_distance
                o1_ball_temp_y = o1_ball_pred_y + math.cos(o1_goal_temp_angle) * o1_goal_temp_distance
            else:
                o1_goal_temp_angle = math.atan2(self.o1_goal_x - o1_ball_pred_x, self.o1_goal_y - o1_ball_pred_y)
                o1_goal_temp_distance = 0.015
                o1_ball_temp_x = o1_ball_pred_x + math.sin(o1_goal_temp_angle) * o1_goal_temp_distance
                o1_ball_temp_y = o1_ball_pred_y + math.cos(o1_goal_temp_angle) * o1_goal_temp_distance

            o1_ball_pred_angle = (math.degrees(
                math.atan2(o1_ball_temp_x - o1_robot_x, o1_ball_temp_y - o1_robot_y)) - o1_robot_angle + 360) % 360

            o1_ball_direction = o1_ball_pred_angle
            if o1_ball_direction > 180:
                o1_ball_direction = o1_ball_direction - 360
            o1_abs_ba = (o1_ball_direction + o1_robot_angle - o1_goal_angle + 360) % 360

            if o1_abs_ba < self.o1_off_angle or o1_abs_ba > 360 - self.o1_off_angle:
                o1_offset = 0
            elif o1_ball_distance < 0.1:
                o1_offset = 62
            elif o1_ball_distance > 1:
                o1_offset = 1
            else:
                o1_offset = 61 * pow(o1_ball_distance - 1, 2) + 1

            if 180 < o1_abs_ba < 360 - self.o1_off_angle:
                o1_offset *= -1

            if o1_ball_pred_y > 0.66 and (0.55 < o1_ball_x < 0.62 or -0.55 > o1_ball_x > -0.62):
                o1_offset *= 0.5
            elif (o1_robot_x >= 0.51 or o1_robot_x <= -0.51) and 0.63 > o1_ball_pred_y - 0.0375 > o1_robot_y:
                o1_offset *= (0.63 - abs(o1_robot_y)) / 0.9
            elif (o1_ball_x <= -0.42 or (o1_ball_x <= -0.35 and o1_ball_y < 0.1)) and (o1_robot_x <= -0.35 or o1_robot_x >= 0.35) and self.o1_off_angle < o1_abs_ba < 180:
                o1_offset *= -1
            elif (o1_ball_x >= 0.42 or (o1_ball_x >= 0.35 and o1_ball_y < 0.1)) and (o1_robot_x <= -0.35 or o1_robot_x >= 0.35) and 180 < o1_abs_ba < 360 - self.o1_off_angle:
                o1_offset *= -1

            o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_ball_pred_angle, 0, 180)

            if o1_closest_angle == 0:
                self.o1_side = 1
            elif o1_closest_angle == 180:
                self.o1_side = -1

            if self.o1_side == 1:
                o1_move_angle = o1_ball_direction + o1_offset
            else:
                if o1_ball_direction + o1_offset >= 0:
                    o1_move_angle = o1_ball_direction + o1_offset - 180
                else:
                    o1_move_angle = 180 - abs((o1_ball_direction + o1_offset))

            if OffenseYellow1.inside_position_range(self, o1_ball_x, o1_ball_y, -0.575, 0.73, 0.01,
                                          0.01) and o1_ball_distance < 0.07 and o1_ball_y > o1_robot_y:
                o1_shoot_angle = 60
                o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle, (o1_shoot_angle + 360) % 360,
                                                                           (o1_shoot_angle + 360 + 180) % 360)
                if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                    o1_spin_speed = 6
                    o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                o1_closest_angle,
                                                                o1_spin_speed)
            elif -0.58 < o1_ball_x < 0.58 and 0.675 > o1_ball_y > -0.2:
                o1_left_speed, o1_right_speed = OffenseYellow1.curve_drive(self, o1_move_angle, self.o1_side, 45)
            elif (o1_ball_x < -0.58 or o1_ball_x > 0.58) and o1_ball_y < 0:
                o1_left_speed, o1_right_speed = OffenseYellow1.curve_drive(self, o1_move_angle, self.o1_side, 90)
            else:
                o1_left_speed, o1_right_speed = OffenseYellow1.curve_drive(self, o1_move_angle, self.o1_side, 60)

        # Robot Ambushing
        elif self.o1_mode == o1_ambush_attack_mode:
            o1_angle_range = 4
            o1_shoot_angle = 0

            o1_point_x = self.o1_limit_x - 0.04
            if OffenseYellow1.calc_distance(self, o1_ball_x, o1_ball_y, self.o1_last_ball_x, self.o1_last_ball_y) < math.pow(10, -5):
                o1_dest_x, o1_dest_y = o1_point_x, o1_ball_y
                o1_offset = -0.0725
            else:
                o1_dest_x, o1_dest_y = OffenseYellow1.solve_attack_point(self, o1_point_x, o1_ball_x, o1_ball_y, self.o1_last_ball_x,
                                                              self.o1_last_ball_y)
                if o1_dest_x is None or o1_dest_y is None:
                    o1_dest_x, o1_dest_y = o1_point_x, o1_ball_y - 0.075
                o1_offset = 0

            o1_dest_y = o1_dest_y + o1_offset

            if o1_dest_y < 0 and o1_dest_y < o1_ball_y - 0.1:
                o1_dest_y = o1_ball_y - 0.075
            elif (o1_dest_y < -0.1 and o1_ball_y > 0) or (o1_dest_y > 0.1 and o1_ball_y < 0):
                o1_dest_y = -0.0725
            elif o1_dest_y > 0 > o1_ball_y and abs(o1_ball_y - o1_dest_y) > 0.15 and o1_ball_y < self.o1_last_ball_y:
                o1_dest_y = -0.0725
            elif o1_dest_y < 0 < o1_ball_y and abs(o1_ball_y - o1_dest_y) > 0.15 and o1_ball_y > self.o1_last_ball_y:
                o1_dest_y = -0.0725

            if o1_ball_y < 0.57 and o1_dest_y > 0.54:
                o1_dest_y = 0.54
            elif o1_dest_y < -0.4 and -0.25 < o1_ball_y < 0.25:
                o1_dest_y = -0.4
            elif o1_dest_y > 0.675:
                o1_dest_y = 0.675

            if not OffenseYellow1.inside_position_range(self, o1_robot_x, o1_robot_y, o1_point_x, o1_dest_y, 0.005, 0.005):
                o1_robot_radius = 0.0475
                o1_ball_radius = 0.021
                if o1_ball_distance < 0.1 and abs(o1_robot_x) + o1_robot_radius > abs(o1_ball_x) - o1_ball_radius > abs(o1_dest_x) and \
                        o1_robot_y + o1_robot_radius > o1_ball_y - o1_ball_radius > o1_dest_y:
                    o1_temp_dest_x, o1_temp_dest_y = OffenseYellow1.solve_target_point(self, o1_point_x, o1_robot_x, o1_robot_y, o1_ball_x,
                                                                      o1_ball_y + 0.1)
                    o1_left_speed, o1_right_speed = OffenseYellow1.drive_to_point(self, o1_left_speed, o1_right_speed, o1_temp_dest_x,
                                                                   o1_temp_dest_y,
                                                                   o1_robot_x,
                                                                   o1_robot_y, o1_robot_angle, o1_spin_speed, o1_angle_range)
                else:
                    o1_left_speed, o1_right_speed = OffenseYellow1.drive_to_point(self, o1_left_speed, o1_right_speed, o1_dest_x, o1_dest_y,
                                                                   o1_robot_x,
                                                                   o1_robot_y, o1_robot_angle, o1_spin_speed, o1_angle_range)
                o1_left_speed, o1_right_speed = OffenseYellow1.drive_to_point(self, o1_left_speed, o1_right_speed, o1_dest_x, o1_dest_y, o1_robot_x,
                                                               o1_robot_y, o1_robot_angle, o1_spin_speed, o1_angle_range)
            else:
                o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle, (o1_shoot_angle + 360) % 360,
                                                                           (o1_shoot_angle + 360 + 180) % 360)
                if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                    o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                o1_closest_angle, o1_spin_speed)
                else:
                    o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 0, -2)

        # Robot Stationing and Tracking
        elif self.o1_mode == o1_stationing_attack_mode:
            if not o1_detect:
                if o1_ball_x < 0:
                    if o1_ball_y < -0.6:
                        o1_dest_x, o1_dest_y = -0.3, -0.35
                    else:
                        o1_dest_x, o1_dest_y = -0.3, -0.2
                else:
                    if o1_ball_y < -0.6:
                        if self.o1_timer < 5:
                            o1_dest_x, o1_dest_y = 0, -0.35
                        else:
                            o1_dest_x, o1_dest_y = -0.2, -0.35
                    else:
                        if self.o1_timer < 5:
                            o1_dest_x, o1_dest_y = 0, -0.2
                        else:
                            o1_dest_x, o1_dest_y = -0.2, -0.2
            else:
                if o1_yield_to_teammate:
                    # C7: dynamically support the active attacker instead of
                    # returning to a fixed central parking point.  The target
                    # stays in a safe central band and is biased away from
                    # the primary robot to reduce clustering.
                    o1_primary_x, o1_primary_y = o1_data.getRobotPosSearch(o1_primary_observer)
                    o1_dest_x = max(-O1_SUPPORT_X_LIMIT, min(O1_SUPPORT_X_LIMIT,
                                                              0.35 * o1_ball_x - 0.20 * o1_primary_x))
                    o1_dest_y = max(O1_SUPPORT_Y_MIN, min(O1_SUPPORT_Y_MAX,
                                                          0.30 * o1_ball_y - 0.10 * o1_primary_y - 0.12))
                elif o1_ball_y > 0:
                    if o1_ball_x < 0:
                        o1_attack_point_distance = OffenseYellow1.calc_distance(self, o1_ball_x, o1_ball_y, -0.3, 0.3)
                        o1_defense_point_distance = OffenseYellow1.calc_distance(self, o1_ball_x, o1_ball_y, -0.3, -0.3)
                    else:
                        o1_attack_point_distance = OffenseYellow1.calc_distance(self, o1_ball_x, o1_ball_y, 0.3, 0.3)
                        o1_defense_point_distance = OffenseYellow1.calc_distance(self, o1_ball_x, o1_ball_y, 0.3, -0.3)
                    if o1_attack_point_distance + 0.03 < o1_defense_point_distance:
                        o1_dest_x, o1_dest_y = 0, 0
                    else:
                        o1_dest_x, o1_dest_y = 0, -0.1
                else:
                    o1_dest_x, o1_dest_y = 0, -0.2
            # Anti-clustering: when the ball is temporarily lost, both
            # offensive robots can otherwise receive the same stationing
            # target (especially around x=0).  Give them a small role-based
            # separation only while they are physically too close.
            if not o1_detect:
                o1_teammate_dx = o1_robot_x - o1_offense2_x
                o1_teammate_dy = o1_robot_y - o1_offense2_y
                o1_teammate_distance = math.hypot(o1_teammate_dx, o1_teammate_dy)
                if o1_teammate_distance < O1_TEAMMATE_SEPARATION_DISTANCE:
                    if o1_teammate_distance > 1e-6:
                        o1_dest_x += O1_TEAMMATE_SEPARATION_OFFSET * o1_teammate_dx / o1_teammate_distance
                        o1_dest_y += O1_TEAMMATE_SEPARATION_OFFSET * o1_teammate_dy / o1_teammate_distance
                    else:
                        o1_dest_x -= O1_TEAMMATE_SEPARATION_OFFSET

            o1_angle_range = 8
            if not OffenseYellow1.inside_position_range(self, o1_robot_x, o1_robot_y, o1_dest_x, o1_dest_y, 0.02, 0.02):
                o1_left_speed, o1_right_speed = OffenseYellow1.drive_to_point(self, o1_left_speed, o1_right_speed, o1_dest_x, o1_dest_y, o1_robot_x,
                                                               o1_robot_y, o1_robot_angle, o1_spin_speed, o1_angle_range)
            else:
                o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle, 0, 180)
                if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                    o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                o1_closest_angle, o1_spin_speed)
                else:
                    o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 0, 2)

        # Double Pushing
        elif self.o1_mode == o1_double_pushing_mode:
            o1_angle_range = 8
            o1_offset_x, o1_offset_y = 0.03, -0.035
            self.o1_double_push_x, self.o1_double_push_y = o1_offense2_x + o1_offset_x, o1_offense2_y + o1_offset_y
            o1_dest_x, o1_dest_y = self.o1_double_push_x, self.o1_double_push_y
            if OffenseYellow1.inside_position_range(self, o1_robot_x, o1_robot_y, o1_dest_x, o1_dest_y, 0.05, 0.05) or \
                    OffenseYellow1.inside_position_range(self, o1_robot_x, o1_robot_y, o1_offense2_x, o1_offense2_y, 0.1,
                                               0.1) and o1_robot_x > o1_offense2_x:
                o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle, 125, 305)
                if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                    o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                o1_closest_angle,
                                                                o1_spin_speed)
                else:
                    o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 125, 10)
            else:
                o1_point_angle = (math.degrees(math.atan2(o1_dest_x - o1_robot_x, o1_dest_y - o1_robot_y)) - o1_robot_angle + 360) % 360

                o1_push_angle = 300
                o1_point_direction = o1_point_angle
                if o1_point_direction > 180:
                    o1_point_direction = o1_point_direction - 360
                o1_abs_ba = (o1_point_direction + o1_robot_angle - o1_push_angle + 360) % 360

                o1_point_distance = OffenseYellow1.calc_distance(self, o1_robot_x, o1_robot_y, o1_dest_x, o1_dest_y)

                if o1_abs_ba < self.o1_off_angle or o1_abs_ba > 360 - self.o1_off_angle:
                    o1_offset = 0
                elif o1_point_distance < 0.1:
                    o1_offset = 62
                elif o1_point_distance > 1:
                    o1_offset = 1
                else:
                    o1_offset = 61 * pow(o1_point_distance - 1, 2) + 1

                if 180 < o1_abs_ba < 360 - self.o1_off_angle:
                    o1_offset *= -1

                o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, (o1_robot_angle + 180) % 360, 0, 180)

                if self.o1_last_mode != o1_double_pushing_mode:
                    if o1_closest_angle == 0:
                        self.o1_side = 1
                    elif o1_closest_angle == 180:
                        self.o1_side = -1

                if self.o1_side == 1:
                    o1_move_angle = o1_point_direction + o1_offset
                else:
                    if o1_point_direction + o1_offset >= 0:
                        o1_move_angle = o1_point_direction + o1_offset - 180
                    else:
                        o1_move_angle = 180 - abs(o1_point_direction + o1_offset)

                o1_left_speed, o1_right_speed = OffenseYellow1.curve_drive(self, o1_move_angle, self.o1_side, 45)

        # Wall Defense
        elif self.o1_mode == o1_wall_defense_mode:
            o1_dest_x, o1_dest_y = self.o1_wall_dest_x, self.o1_wall_dest_y
            o1_angle_range = 15
            if not OffenseYellow1.inside_position_range(self, o1_robot_x, o1_robot_y, o1_dest_x, o1_dest_y, 1, 1, 0.4, 0.02, 0.05, 1):
                o1_point_angle = (math.degrees(math.atan2(o1_dest_x - o1_robot_x, o1_dest_y - o1_robot_y)) - o1_robot_angle + 360) % 360

                o1_point_direction = o1_point_angle
                if o1_point_direction > 180:
                    o1_point_direction = o1_point_direction - 360

                o1_offset = 0

                o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_point_angle, 0, 180)

                if o1_closest_angle == 0:
                    self.o1_side = 1
                elif o1_closest_angle == 180:
                    self.o1_side = -1

                if self.o1_side == 1:
                    o1_move_angle = o1_point_direction + o1_offset
                else:
                    if o1_point_direction + o1_offset >= 0:
                        o1_move_angle = o1_point_direction + o1_offset - 180
                    else:
                        o1_move_angle = 180 - abs(o1_point_direction + o1_offset)

                o1_left_speed, o1_right_speed = OffenseYellow1.curve_drive(self, o1_move_angle, self.o1_side, 60)
            else:
                o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle, 0, 180)
                if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                    o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                o1_closest_angle,
                                                                o1_spin_speed)
                else:
                    o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 0, 10)

        # Midfield Defense
        elif self.o1_mode == o1_midfield_defense_mode:
            o1_dest_x, o1_dest_y = 0, -0.54
            o1_angle_range = 8
            if not OffenseYellow1.inside_position_range(self, o1_robot_x, o1_robot_y, o1_dest_x, o1_dest_y, 0.02, 0.02):
                o1_left_speed, o1_right_speed = OffenseYellow1.drive_to_point(self, o1_left_speed, o1_right_speed, o1_dest_x, o1_dest_y, o1_robot_x,
                                                               o1_robot_y, o1_robot_angle, o1_spin_speed, o1_angle_range)
            else:
                o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle, 90, 270)
                if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                    o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                o1_closest_angle, o1_spin_speed)
                else:
                    o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 90, 2)

        # Deflection Defense
        elif self.o1_mode == o1_deflection_defense_mode:
            o1_run_pass = True
            o1_dest_x, o1_dest_y = 0, -0.35
            o1_angle_range = 8
            if not OffenseYellow1.inside_position_range(self, o1_robot_x, o1_robot_y, o1_dest_x, o1_dest_y, 0.02, 0.02):
                o1_left_speed, o1_right_speed = OffenseYellow1.drive_to_point(self, o1_left_speed, o1_right_speed, o1_dest_x, o1_dest_y, o1_robot_x,
                                                               o1_robot_y, o1_robot_angle, o1_spin_speed, o1_angle_range)
            else:
                o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle, 0, 180)
                if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                    o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                o1_closest_angle, o1_spin_speed)
                else:
                    o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 0, 2)

        # Goal Defense
        elif self.o1_mode == o1_goal_defense_mode:
            o1_dest_x, o1_dest_y = 0.16, -0.8
            o1_angle_range = 8
            if not OffenseYellow1.inside_position_range(self, o1_robot_x, o1_robot_y, o1_dest_x, o1_dest_y, 0.02, 0.02):
                o1_left_speed, o1_right_speed = OffenseYellow1.drive_to_point(self, o1_left_speed, o1_right_speed, o1_dest_x, o1_dest_y, o1_robot_x,
                                                               o1_robot_y, o1_robot_angle, o1_spin_speed, o1_angle_range)
            else:
                o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle, 90, 270)
                if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                    o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                o1_closest_angle,
                                                                o1_spin_speed)
                else:
                    o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 90, -10)

        # Main Trap Ball
        elif self.o1_mode == o1_trap_ball_main_mode:
            o1_run_pass = True

            o1_angle_range = 2
            o1_spin_speed = 4

            if self.o1_trap_stage == 1:
                if o1_ball_y > -0.7 or not o1_offense2_x < o1_ball_x < o1_offense1_x or o1_ball_distance > 0.1 or o1_offense2_ball_distance > 0.1:
                    self.o1_trap_stage = -1
                else:
                    o1_shoot_angle = 40
                    o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle,
                                                                               (o1_shoot_angle + 360) % 360,
                                                                               (o1_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                        o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                    o1_closest_angle,
                                                                    o1_spin_speed)
                    else:
                        o1_move_speed = 0
                        o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 0, o1_move_speed)

                        if OffenseYellow1.inside_position_range(self, o1_offense2_x, o1_offense2_y, o1_offense2_x, -0.672, 0.01, 0.01) and \
                                OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                            self.o1_trap_stage = 2

            elif self.o1_trap_stage == 2:
                if o1_ball_y > -0.7 or o1_ball_distance > 0.1 or o1_offense2_ball_distance > 0.1:
                    self.o1_trap_stage = -1
                else:
                    o1_shoot_angle = 40
                    o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle,
                                                                               (o1_shoot_angle + 360) % 360,
                                                                               (o1_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                        o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                    o1_closest_angle,
                                                                    o1_spin_speed)
                    else:
                        o1_move_speed = 10
                        o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 0, o1_move_speed)

                if OffenseYellow1.inside_position_range(self, o1_ball_x, o1_ball_y, -0.56, -0.73, 0.01, 0.01):
                    self.o1_trap_stage = 3

            elif self.o1_trap_stage == 3:
                if o1_ball_y > -0.51 or o1_ball_x > -0.43:
                    self.o1_trap_stage = -1
                else:
                    o1_shoot_angle = 40
                    o1_dest_x, o1_dest_y = -0.47, -0.67
                    if not OffenseYellow1.inside_position_range(self, o1_robot_x, o1_robot_y, o1_dest_x, o1_dest_y, 0.005, 0.005):
                        o1_left_speed, o1_right_speed = OffenseYellow1.drive_to_point(self, o1_left_speed, o1_right_speed, o1_dest_x, o1_dest_y,
                                                                       o1_robot_x,
                                                                       o1_robot_y, o1_robot_angle, o1_spin_speed, o1_angle_range)
                    else:
                        o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle,
                                                                                   (o1_shoot_angle + 360) % 360,
                                                                                   (o1_shoot_angle + 360 + 180) % 360)
                        if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                            o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                        o1_closest_angle,
                                                                        o1_spin_speed)
                        else:
                            o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 0, 0)

                            o1_shoot_angle = 0
                            o1_dest_x, o1_dest_y = -0.572, -0.595
                            o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_offense2_angle,
                                                                                       (o1_shoot_angle + 360) % 360,
                                                                                       (o1_shoot_angle + 360 + 180) % 360)
                            if OffenseYellow1.inside_position_range(self, o1_offense2_x, o1_offense2_y, o1_dest_x, o1_dest_y, 0.01, 0.01) and \
                                    OffenseYellow1.inside_angle_range(self, o1_offense2_angle, o1_closest_angle, o1_angle_range):
                                self.o1_trap_stage = 4

            elif self.o1_trap_stage == 4:
                if o1_ball_y > -0.51 or o1_ball_x > -0.43:
                    self.o1_trap_stage = -1
                else:
                    if not OffenseYellow1.inside_position_range(self, o1_ball_x, o1_ball_y, -0.63, -0.64, 0.01, 0.01):
                        o1_dest_x, o1_dest_y = o1_ball_x, o1_ball_y
                        o1_point_angle = (math.degrees(
                            math.atan2(o1_dest_x - o1_robot_x, o1_dest_y - o1_robot_y)) - o1_robot_angle + 360) % 360

                        o1_point_direction = o1_point_angle
                        if o1_point_direction > 180:
                            o1_point_direction = o1_point_direction - 360

                        o1_offset = 0

                        o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_point_angle, 0, 180)

                        if o1_closest_angle == 0:
                            self.o1_side = 1
                        elif o1_closest_angle == 180:
                            self.o1_side = -1

                        if self.o1_side == 1:
                            o1_move_angle = o1_point_direction + o1_offset
                        else:
                            if o1_point_direction + o1_offset >= 0:
                                o1_move_angle = o1_point_direction + o1_offset - 180
                            else:
                                o1_move_angle = 180 - abs(o1_point_direction + o1_offset)

                        o1_left_speed, o1_right_speed = OffenseYellow1.curve_drive(self, o1_move_angle, self.o1_side, 45)

                        if OffenseYellow1.inside_position_range(self, o1_ball_x, o1_ball_y, -0.58, -0.73, 0.01, 0.01) and \
                                OffenseYellow1.calc_distance(
                                    self, o1_ball_x, o1_ball_y, self.o1_last_ball_x, self.o1_last_ball_y) < math.pow(10, -5):
                            self.o1_trap_stage = 3
                        elif OffenseYellow1.inside_position_range(self, o1_ball_x, o1_ball_y, -0.63, -0.68, 0.01, 0.01) and \
                                OffenseYellow1.calc_distance(
                                    self, o1_ball_x, o1_ball_y, self.o1_last_ball_x, self.o1_last_ball_y) < math.pow(10, -5):
                            self.o1_trap_stage = 5

                    elif OffenseYellow1.inside_position_range(self, o1_robot_x, o1_robot_y, -0.57, -0.69, 0.01, 0.01):
                        self.o1_trap_stage = 5

            elif self.o1_trap_stage == 5:
                if o1_ball_x > -0.56 or not o1_offense2_y > o1_ball_y > o1_offense1_y:
                    self.o1_trap_stage = -1
                else:
                    o1_shoot_angle = 140
                    o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle,
                                                                               (o1_shoot_angle + 360) % 360,
                                                                               (o1_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                        o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                    o1_closest_angle,
                                                                    o1_spin_speed)
                    else:
                        o1_move_speed = 0
                        o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 0, o1_move_speed)

                        self.o1_trap_stage = 6

            elif self.o1_trap_stage == 6:
                if o1_ball_x > -0.56 or o1_ball_distance > 0.1 or o1_offense2_ball_distance > 0.1:
                    self.o1_trap_stage = -1
                else:
                    o1_shoot_angle = 140
                    o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle,
                                                                               (o1_shoot_angle + 360) % 360,
                                                                               (o1_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                        o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                    o1_closest_angle,
                                                                    o1_spin_speed)
                    else:
                        o1_move_speed = -10
                        o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 0, o1_move_speed)

                    if OffenseYellow1.inside_position_range(self, o1_ball_x, o1_ball_y, -0.625, 0.67, 0.01, 0.01):
                        self.o1_trap_stage = 7

            elif self.o1_trap_stage == 7:
                if o1_ball_x > -0.4 or o1_ball_y < 0.55:
                    self.o1_trap_stage = -1
                else:
                    o1_shoot_angle = 140
                    o1_dest_x, o1_dest_y = -0.57, 0.57
                    if not OffenseYellow1.inside_position_range(self, o1_robot_x, o1_robot_y, o1_dest_x, o1_dest_y, 0.005, 0.005):
                        o1_left_speed, o1_right_speed = OffenseYellow1.drive_to_point(self, o1_left_speed, o1_right_speed, o1_dest_x, o1_dest_y,
                                                                       o1_robot_x,
                                                                       o1_robot_y, o1_robot_angle, o1_spin_speed, o1_angle_range)
                    else:
                        o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle,
                                                                                   (o1_shoot_angle + 360) % 360,
                                                                                   (o1_shoot_angle + 360 + 180) % 360)
                        if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                            o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                        o1_closest_angle,
                                                                        o1_spin_speed)
                        else:
                            o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 0, 0)

                            o1_shoot_angle = 90
                            o1_dest_x, o1_dest_y = -0.495, 0.672
                            o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_offense2_angle,
                                                                                       (o1_shoot_angle + 360) % 360,
                                                                                       (o1_shoot_angle + 360 + 180) % 360)
                            if OffenseYellow1.inside_position_range(self, o1_offense2_x, o1_offense2_y, o1_dest_x, o1_dest_y, 0.01, 0.01) and \
                                    OffenseYellow1.inside_angle_range(self, o1_offense2_angle, o1_closest_angle, o1_angle_range):
                                self.o1_trap_stage = 8

            elif self.o1_trap_stage == 8:
                if o1_ball_x > -0.4 or o1_ball_y < 0.55:
                    self.o1_trap_stage = -1
                else:
                    if not OffenseYellow1.inside_position_range(self, o1_ball_x, o1_ball_y, -0.54, 0.73, 0.01, 0.01):
                        o1_dest_x, o1_dest_y = o1_ball_x, o1_ball_y
                        o1_point_angle = (math.degrees(
                            math.atan2(o1_dest_x - o1_robot_x, o1_dest_y - o1_robot_y)) - o1_robot_angle + 360) % 360

                        o1_point_direction = o1_point_angle
                        if o1_point_direction > 180:
                            o1_point_direction = o1_point_direction - 360

                        o1_offset = 0

                        o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_point_angle, 0, 180)

                        if o1_closest_angle == 0:
                            self.o1_side = 1
                        elif o1_closest_angle == 180:
                            self.o1_side = -1

                        if self.o1_side == 1:
                            o1_move_angle = o1_point_direction + o1_offset
                        else:
                            if o1_point_direction + o1_offset >= 0:
                                o1_move_angle = o1_point_direction + o1_offset - 180
                            else:
                                o1_move_angle = 180 - abs(o1_point_direction + o1_offset)

                        o1_left_speed, o1_right_speed = OffenseYellow1.curve_drive(self, o1_move_angle, self.o1_side, 45)

                        if OffenseYellow1.inside_position_range(self, o1_ball_x, o1_ball_y, -0.63, 0.68, 0.01,
                                                      0.01) and OffenseYellow1.calc_distance(
                                self, o1_ball_x, o1_ball_y, self.o1_last_ball_x, self.o1_last_ball_y) < math.pow(10, -5):
                            self.o1_trap_stage = 7
                        elif OffenseYellow1.inside_position_range(self, o1_ball_x, o1_ball_y, -0.58, 0.73, 0.01, 0.01) and \
                                OffenseYellow1.calc_distance(
                                    self, o1_ball_x, o1_ball_y, self.o1_last_ball_x, self.o1_last_ball_y) < math.pow(10,
                                                                                                         -5) and o1_ball_distance < 0.05:
                            self.o1_trap_stage = 9

                    elif OffenseYellow1.inside_position_range(self, o1_robot_x, o1_robot_y, -0.59, 0.67, 0.01, 0.01):
                        self.o1_trap_stage = 9

            elif self.o1_trap_stage == 9:
                if o1_ball_y < 0.69 or not o1_offense2_x > o1_ball_x > o1_offense1_x:
                    self.o1_trap_stage = -1
                else:
                    o1_shoot_angle = 40
                    o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle,
                                                                               (o1_shoot_angle + 360) % 360,
                                                                               (o1_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                        o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                    o1_closest_angle,
                                                                    o1_spin_speed)
                    else:
                        o1_move_speed = 0
                        o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 0, o1_move_speed)

                        self.o1_trap_stage = 10

            elif self.o1_trap_stage == 10:
                if o1_ball_x > -0.1 or not 0.77 > o1_ball_y > 0.68 or o1_ball_distance > 0.1 or o1_offense2_ball_distance > 0.1:
                    self.o1_trap_stage = -1
                else:
                    o1_shoot_angle = 40
                    o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle,
                                                                               (o1_shoot_angle + 360) % 360,
                                                                               (o1_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                        o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                    o1_closest_angle,
                                                                    o1_spin_speed)
                    else:
                        o1_move_speed = -10
                        o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 0, o1_move_speed)

        # Support Trap Ball
        elif self.o1_mode == o1_trap_ball_support_mode:
            o1_run_pass = True

            o1_angle_range = 2
            o1_spin_speed = 4

            o1_robot_radius = 0.0375
            o1_ball_radius = 0.021

            if o1_trap_stage == 1:
                o1_shoot_angle = 90
                o1_dest_x, o1_dest_y = o1_robot_x, -0.67
                if not OffenseYellow1.inside_position_range(self, o1_robot_x, o1_robot_y, o1_dest_x, o1_dest_y, 0.01, 0.01):
                    o1_left_speed, o1_right_speed = OffenseYellow1.drive_to_point(self, o1_left_speed, o1_right_speed, o1_dest_x, o1_dest_y,
                                                                   o1_robot_x,
                                                                   o1_robot_y, o1_robot_angle, o1_spin_speed, o1_angle_range)
                else:
                    o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle,
                                                                               (o1_shoot_angle + 360) % 360,
                                                                               (o1_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                        o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                    o1_closest_angle,
                                                                    o1_spin_speed)
                    else:
                        o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 0, 0)

            elif o1_trap_stage == 2:
                o1_shoot_angle = 90
                o1_dest_x, o1_dest_y = o1_ball_x + o1_robot_radius + o1_ball_radius - 0.008, -0.672
                if not OffenseYellow1.inside_position_range(self, o1_robot_x, o1_robot_y, o1_dest_x, o1_dest_y, 0.005, 0.005):
                    o1_left_speed, o1_right_speed = OffenseYellow1.drive_to_point(self, o1_left_speed, o1_right_speed, o1_dest_x, o1_dest_y,
                                                                   o1_robot_x,
                                                                   o1_robot_y, o1_robot_angle, o1_spin_speed, o1_angle_range)
                else:
                    o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle,
                                                                               (o1_shoot_angle + 360) % 360,
                                                                               (o1_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                        o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                    o1_closest_angle, o1_spin_speed)
                    else:
                        o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 0, 0)

            elif o1_trap_stage == 3:
                if OffenseYellow1.inside_position_range(self, o1_offense2_x, o1_offense2_y, 0.47, -0.67, 0.005, 0.005):
                    o1_shoot_angle = 0
                    o1_dest_x, o1_dest_y = 0.5715, -0.595
                    if not OffenseYellow1.inside_position_range(self, o1_robot_x, o1_robot_y, o1_dest_x, o1_dest_y, 0.005, 0.005):
                        o1_left_speed, o1_right_speed = OffenseYellow1.drive_to_point(self, o1_left_speed, o1_right_speed, o1_dest_x, o1_dest_y,
                                                                       o1_robot_x,
                                                                       o1_robot_y, o1_robot_angle, o1_spin_speed, o1_angle_range)
                    else:
                        o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle,
                                                                                   (o1_shoot_angle + 360) % 360,
                                                                                   (o1_shoot_angle + 360 + 180) % 360)
                        if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                            o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                        o1_closest_angle,
                                                                        o1_spin_speed)
                        else:
                            o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 0, 0)

            elif o1_trap_stage == 4:
                o1_shoot_angle = 0
                o1_dest_x, o1_dest_y = 0.5715, -0.595
                if not OffenseYellow1.inside_position_range(self, o1_robot_x, o1_robot_y, o1_dest_x, o1_dest_y, 0.005, 0.005):
                    o1_left_speed, o1_right_speed = OffenseYellow1.drive_to_point(self, o1_left_speed, o1_right_speed, o1_dest_x, o1_dest_y,
                                                                   o1_robot_x,
                                                                   o1_robot_y, o1_robot_angle, o1_spin_speed, o1_angle_range)
                else:
                    o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle,
                                                                               (o1_shoot_angle + 360) % 360,
                                                                               (o1_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                        o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                    o1_closest_angle,
                                                                    o1_spin_speed)
                    else:
                        o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 0, 2)

            elif o1_trap_stage == 5:
                o1_shoot_angle = 0
                o1_dest_x, o1_dest_y = 0.5715, -0.57
                if not OffenseYellow1.inside_position_range(self, o1_robot_x, o1_robot_y, o1_dest_x, o1_dest_y, 0.005, 0.005):
                    o1_left_speed, o1_right_speed = OffenseYellow1.drive_to_point(self, o1_left_speed, o1_right_speed, o1_dest_x, o1_dest_y,
                                                                   o1_robot_x,
                                                                   o1_robot_y, o1_robot_angle, o1_spin_speed, o1_angle_range)
                else:
                    o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle,
                                                                               (o1_shoot_angle + 360) % 360,
                                                                               (o1_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                        o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                    o1_closest_angle,
                                                                    o1_spin_speed)
                    else:
                        o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 0, 0)

            elif o1_trap_stage == 6:
                o1_shoot_angle = 0
                o1_dest_x, o1_dest_y = 0.5715, o1_ball_y + o1_robot_radius + o1_ball_radius - 0.008
                if not OffenseYellow1.inside_position_range(self, o1_robot_x, o1_robot_y, o1_dest_x, o1_dest_y, 0.005, 0.005):
                    o1_left_speed, o1_right_speed = OffenseYellow1.drive_to_point(self, o1_left_speed, o1_right_speed, o1_dest_x, o1_dest_y,
                                                                   o1_robot_x,
                                                                   o1_robot_y, o1_robot_angle, o1_spin_speed, o1_angle_range)
                else:
                    o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle,
                                                                               (o1_shoot_angle + 360) % 360,
                                                                               (o1_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                        o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                    o1_closest_angle, o1_spin_speed)
                    else:
                        o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 0, 0)

            elif o1_trap_stage == 7:
                if OffenseYellow1.inside_position_range(self, o1_offense2_x, o1_offense2_y, 0.57, 0.57, 0.005, 0.005):
                    o1_shoot_angle = 90
                    o1_dest_x, o1_dest_y = 0.495, 0.672
                    if not OffenseYellow1.inside_position_range(self, o1_robot_x, o1_robot_y, o1_dest_x, o1_dest_y, 0.005, 0.005):
                        o1_left_speed, o1_right_speed = OffenseYellow1.drive_to_point(self, o1_left_speed, o1_right_speed, o1_dest_x, o1_dest_y,
                                                                       o1_robot_x,
                                                                       o1_robot_y, o1_robot_angle, o1_spin_speed, o1_angle_range)
                    else:
                        o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle,
                                                                                   (o1_shoot_angle + 360) % 360,
                                                                                   (o1_shoot_angle + 360 + 180) % 360)
                        if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                            o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                        o1_closest_angle,
                                                                        o1_spin_speed)
                        else:
                            o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 0, 0)

            elif o1_trap_stage == 8:
                o1_shoot_angle = 90
                o1_dest_x, o1_dest_y = 0.495, 0.672
                if not OffenseYellow1.inside_position_range(self, o1_robot_x, o1_robot_y, o1_dest_x, o1_dest_y, 0.005, 0.005):
                    o1_left_speed, o1_right_speed = OffenseYellow1.drive_to_point(self, o1_left_speed, o1_right_speed, o1_dest_x, o1_dest_y,
                                                                   o1_robot_x,
                                                                   o1_robot_y, o1_robot_angle, o1_spin_speed, o1_angle_range)
                else:
                    o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle,
                                                                               (o1_shoot_angle + 360) % 360,
                                                                               (o1_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                        o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                    o1_closest_angle,
                                                                    o1_spin_speed)
                    else:
                        o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 90, 2)

            elif o1_trap_stage == 9:
                o1_shoot_angle = 90
                o1_dest_x, o1_dest_y = 0.495, 0.672
                if not OffenseYellow1.inside_position_range(self, o1_robot_x, o1_robot_y, o1_dest_x, o1_dest_y, 0.005, 0.005):
                    o1_left_speed, o1_right_speed = OffenseYellow1.drive_to_point(self, o1_left_speed, o1_right_speed, o1_dest_x, o1_dest_y,
                                                                   o1_robot_x,
                                                                   o1_robot_y, o1_robot_angle, o1_spin_speed, o1_angle_range)
                else:
                    o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle,
                                                                               (o1_shoot_angle + 360) % 360,
                                                                               (o1_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                        o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                    o1_closest_angle,
                                                                    o1_spin_speed)
                    else:
                        o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 0, 0)

            elif o1_trap_stage == 10:
                o1_shoot_angle = 90
                o1_dest_x, o1_dest_y = o1_ball_x - o1_robot_radius - o1_ball_radius + 0.008, 0.672
                if not OffenseYellow1.inside_position_range(self, o1_robot_x, o1_robot_y, o1_dest_x, o1_dest_y, 0.005, 0.005):
                    o1_left_speed, o1_right_speed = OffenseYellow1.drive_to_point(self, o1_left_speed, o1_right_speed, o1_dest_x, o1_dest_y,
                                                                   o1_robot_x,
                                                                   o1_robot_y, o1_robot_angle, o1_spin_speed, o1_angle_range)
                else:
                    o1_closest_angle, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle,
                                                                               (o1_shoot_angle + 360) % 360,
                                                                               (o1_shoot_angle + 360 + 180) % 360)
                    if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle, o1_angle_range):
                        o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle,
                                                                    o1_closest_angle, o1_spin_speed)
                    else:
                        o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, 0, 0)

        o1_distance_range = 0.1
        o1_temp_pred_x, o1_temp_pred_y = 2 * o1_ball_x - self.o1_last_ball_x, 2 * o1_ball_y - self.o1_last_ball_y
        o1_temp_pred_distance = OffenseYellow1.calc_distance(self, o1_robot_x, o1_robot_y, o1_temp_pred_x, o1_temp_pred_y)
        if not o1_run_pass and o1_ball_y < 0 and -0.23 < o1_robot_x < 0.23 and o1_robot_y < 0 and \
                ((o1_ball_distance < o1_distance_range and o1_ball_y < o1_robot_y) or
                 (o1_temp_pred_distance < o1_distance_range and o1_temp_pred_y < o1_robot_y) or
                 (OffenseYellow1.calc_distance(self, o1_robot_x, o1_robot_y, self.o1_last_ball_x,
                                           self.o1_last_ball_y) < o1_distance_range and
                  self.o1_last_ball_y < o1_robot_y)):
            o1_left_speed, o1_right_speed = 0, 0

        self.o1_last_ball_x, self.o1_last_ball_y = o1_ball_x, o1_ball_y

        self.o1_last_mode = self.o1_mode

        o1_robot.left_motor.setVelocity(o1_left_speed)
        o1_robot.right_motor.setVelocity(o1_right_speed)

    def reset_strategy(self):
        self.o1_mode = 0
        self.o1_side = 1
        self.o1_goal_x, self.o1_goal_y = 0, 0.75
        self.o1_off_angle = 25
        self.o1_last_ball_x, self.o1_last_ball_y = 0, 0
        self.o1_last_mode = 0
        self.o1_limit_x = 0.12
        self.o1_timer = 0
        self.o1_push_ball = False
        self.o1_trap_stage = -1
        self.o1_double_push_x, self.o1_double_push_y = 0, 0
        self.o1_wall_dest_x, self.o1_wall_dest_y = -0.44, -0.7
        self.o1_team = "Y"

    # Add any additional functions here
    def curve_drive(self, o1_dir, o1_side, o1_ang=65):
        # Side (+: front, -: back)
        o1_left_speed = o1_side * -10 * (o1_ang + o1_dir * o1_side) / o1_ang
        o1_right_speed = o1_side * -10 * (o1_ang - o1_dir * o1_side) / o1_ang

        if o1_left_speed > 10:
            o1_left_speed = 10
        elif o1_left_speed < -10:
            o1_left_speed = -10
        if o1_right_speed > 10:
            o1_right_speed = 10
        elif o1_right_speed < -10:
            o1_right_speed = -10

        return o1_left_speed, o1_right_speed

    def pick_direction(self, o1_robot_angle, o1_angle_a, o1_angle_b):
        o1_angle_a_left = (o1_robot_angle - o1_angle_a + 360) % 360
        o1_angle_a_right = (o1_angle_a - o1_robot_angle + 360) % 360
        o1_angle_b_left = (o1_robot_angle - o1_angle_b + 360) % 360
        o1_angle_b_right = (o1_angle_b - o1_robot_angle + 360) % 360

        o1_angle_smallest = min([o1_angle_a_left, o1_angle_a_right, o1_angle_b_left, o1_angle_b_right])

        if o1_angle_smallest == o1_angle_a_left:
            return o1_angle_a, -o1_angle_a_left
        elif o1_angle_smallest == o1_angle_a_right:
            return o1_angle_a, o1_angle_a_right
        elif o1_angle_smallest == o1_angle_b_left:
            return o1_angle_b, -o1_angle_b_left
        else:
            return o1_angle_b, -o1_angle_b_right

    def drive_to_point(self, o1_left_speed, o1_right_speed, o1_dest_x, o1_dest_y, o1_robot_x, o1_robot_y, o1_robot_angle, o1_spin_speed,
                o1_angle_range):
        o1_closest_angle, o1_absolute_angle = OffenseYellow1.relative_coords(self, o1_robot_angle, o1_robot_x, o1_robot_y, o1_dest_x, o1_dest_y)
        if not OffenseYellow1.inside_angle_range(self, o1_robot_angle, o1_closest_angle,
                                                   o1_angle_range + O1_STEERING_DEADBAND):
            o1_left_speed, o1_right_speed = OffenseYellow1.turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle, o1_closest_angle,
                                                        o1_spin_speed)
        else:
            if self.o1_team == "B":
                o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, o1_absolute_angle, -10)
            else:
                o1_left_speed, o1_right_speed = OffenseYellow1.straight_drive(self, o1_robot_angle, o1_absolute_angle, 10)
        return o1_left_speed, o1_right_speed

    def relative_coords(self, o1_robot_angle, o1_robot_x, o1_robot_y, o1_dest_x, o1_dest_y):
        o1_absolute_angle = (math.degrees(math.atan2(o1_robot_x - o1_dest_x, o1_robot_y - o1_dest_y))) % 360
        o1_closest_angle, _ = OffenseYellow1.pick_direction(self, o1_robot_angle, o1_absolute_angle,
                                                                   (o1_absolute_angle + 180) % 360)
        return o1_closest_angle, o1_absolute_angle

    def turn_to_heading(self, o1_left_speed, o1_right_speed, o1_robot_angle, o1_angle, o1_spin_speed):
        _, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_robot_angle, o1_angle, (o1_angle + 180) % 360)
        # The closest difference is relative to the robot
        if o1_closest_difference > 0:
            o1_left_speed = -o1_spin_speed
            o1_right_speed = o1_spin_speed
        elif o1_closest_difference < 0:
            o1_left_speed = o1_spin_speed
            o1_right_speed = -o1_spin_speed
        return o1_left_speed, o1_right_speed

    def straight_drive(self, o1_robot_angle, o1_target_angle, o1_move_speed):
        o1_closest_angle, _ = OffenseYellow1.pick_direction(self, o1_target_angle, o1_robot_angle,
                                                                   (o1_robot_angle + 180) % 360)
        if o1_closest_angle == o1_robot_angle:
            o1_left_speed = o1_move_speed
            o1_right_speed = o1_move_speed
        else:
            o1_left_speed = -o1_move_speed
            o1_right_speed = -o1_move_speed
        return o1_left_speed, o1_right_speed

    def inside_position_range(self, o1_pos_x, o1_pos_y, o1_dest_x, o1_dest_y, o1_range_x=0.2, o1_range_y=0.2, o1_range_x_up=None, o1_range_x_down=None,
                      o1_range_y_up=None, o1_range_y_down=None):
        if o1_range_x_up is None:
            o1_range_x_up = o1_range_x
        if o1_range_x_down is None:
            o1_range_x_down = o1_range_x
        if o1_range_y_up is None:
            o1_range_y_up = o1_range_y
        if o1_range_y_down is None:
            o1_range_y_down = o1_range_y
        if o1_pos_x > o1_dest_x + o1_range_x_up or o1_pos_x < o1_dest_x - o1_range_x_down or o1_pos_y > o1_dest_y + o1_range_y_up or o1_pos_y < o1_dest_y - o1_range_y_down:
            return False
        return True

    def inside_angle_range(self, o1_angle, o1_target_angle, o1_angle_range=15, o1_angle_left_range=None, o1_angle_right_range=None,
                   o1_both_sides=True):
        if o1_angle_left_range is None:
            o1_angle_left_range = o1_angle_range
        if o1_angle_right_range is None:
            o1_angle_right_range = o1_angle_range
        if o1_both_sides:
            _, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_angle, o1_target_angle,
                                                                       (o1_target_angle + 180) % 360)
        else:
            _, o1_closest_difference = OffenseYellow1.pick_direction(self, o1_angle, o1_target_angle, o1_target_angle)
        if o1_angle_right_range > o1_closest_difference > -o1_angle_left_range:
            return True
        return False

    def scale_linear(self, o1_raw_value, o1_min_old_value, o1_max_old_value, o1_min_new_value, o1_max_new_value):
        if o1_raw_value >= o1_max_old_value:
            return o1_max_new_value
        elif o1_raw_value <= o1_min_old_value:
            return o1_min_new_value
        else:
            o1_slope = (o1_max_new_value - o1_min_new_value) / (o1_max_old_value - o1_min_old_value)
            o1_intercept = o1_slope * o1_min_old_value - o1_min_new_value
            o1_new_value = o1_slope * o1_raw_value + o1_intercept
        return o1_new_value

    def solve_target_point(self, o1_input_value, o1_point_a_x, o1_point_a_y, o1_point_b_x, o1_point_b_y, o1_find_x=True):
        o1_point_a_x, o1_point_a_y = -o1_point_a_x, o1_point_a_y
        o1_point_b_x, o1_point_b_y = -o1_point_b_x, o1_point_b_y

        if o1_find_x:
            o1_input_value = -o1_input_value

        o1_point_angle = math.atan2(o1_point_b_y - o1_point_a_y, o1_point_b_x - o1_point_a_x)
        o1_point_slope = math.tan(o1_point_angle)
        o1_point_intercept = o1_point_a_y - o1_point_slope * o1_point_a_x

        if not o1_find_x:
            return -(o1_input_value - o1_point_intercept) / o1_point_slope, o1_input_value
        return -o1_input_value, o1_point_slope * o1_input_value + o1_point_intercept

    def solve_attack_point(self, o1_point_x, o1_ball_x, o1_ball_y, o1_last_ball_x, o1_last_ball_y, o1_offset_front=0.0,
                        o1_offset_back=0.0):
        o1_ball_radius = 0.021
        o1_robot_short_radius = 0.0375  # 0.0383
        o1_robot_long_radius = 0.0475

        o1_main_x, o1_main_y = OffenseYellow1.solve_target_point(self, o1_point_x, o1_ball_x, o1_ball_y, o1_last_ball_x, o1_last_ball_y)
        o1_main_x, o1_main_y = o1_main_y, -o1_main_x

        o1_point_y = -o1_point_x
        o1_ball_x, o1_ball_y = o1_ball_y, -o1_ball_x
        o1_last_ball_x, o1_last_ball_y = o1_last_ball_y, -o1_last_ball_x

        o1_theta = math.atan2(o1_last_ball_y - o1_ball_y, o1_last_ball_x - o1_ball_x)
        o1_abs_theta = abs(o1_theta)

        if abs(o1_theta) < O1_ATTACK_EPSILON:
            return None, None

        # Finding first gap length
        o1_gap_1 = o1_ball_radius / math.sin(o1_abs_theta)
        o1_length_1 = o1_gap_1 * math.tan(o1_abs_theta)

        # Finding second gap length
        o1_slope_original = math.tan(o1_theta)
        o1_intercept_original = (o1_ball_y - o1_slope_original * o1_ball_x)
        if o1_point_y > o1_ball_y:
            o1_intercept_original += o1_length_1
        else:
            o1_intercept_original -= o1_length_1
        o1_slope_perpendicular = -1 / o1_slope_original
        o1_intercept_perpendicular = (o1_ball_y - o1_slope_perpendicular * o1_ball_x)

        o1_left_matrix = np.array([[-o1_slope_original, 1], [-o1_slope_perpendicular, 1]])
        o1_right_matrix = np.array([o1_intercept_original, o1_intercept_perpendicular])
        o1_solution = np.linalg.solve(o1_left_matrix, o1_right_matrix).tolist()
        o1_solution_x, o1_solution_y = o1_solution[0], o1_solution[1]

        o1_temp_length = math.sqrt(math.pow((o1_main_x + o1_gap_1) - o1_solution_x, 2) + math.pow(o1_main_y - o1_solution_y, 2))
        o1_length_2 = o1_temp_length * math.sin(o1_abs_theta)

        if self.o1_team == "B":
            if o1_ball_x > o1_main_x:
                if o1_length_2 >= o1_robot_long_radius:
                    o1_gap_2 = o1_robot_long_radius / math.tan(o1_abs_theta)
                    o1_dest_x, o1_dest_y = o1_main_x + o1_gap_1 + o1_gap_2 + o1_robot_short_radius + o1_offset_back, o1_point_y
                else:
                    o1_dest_x, o1_dest_y = o1_ball_x + o1_ball_radius + o1_robot_short_radius + o1_offset_back, o1_point_y
            else:
                o1_dest_x, o1_dest_y = o1_main_x + o1_ball_radius + o1_robot_short_radius + o1_offset_front, o1_point_y
        else:
            if o1_ball_x < o1_main_x:
                if o1_length_2 >= o1_robot_long_radius:
                    o1_gap_2 = o1_robot_long_radius / math.tan(o1_abs_theta)
                    o1_dest_x, o1_dest_y = o1_main_x - o1_gap_1 - o1_gap_2 - o1_robot_short_radius - o1_offset_back, o1_point_y
                else:
                    o1_dest_x, o1_dest_y = o1_ball_x - o1_ball_radius - o1_robot_short_radius - o1_offset_back, o1_point_y
            else:
                o1_dest_x, o1_dest_y = o1_main_x - o1_ball_radius - o1_robot_short_radius - o1_offset_front, o1_point_y

        return -o1_dest_y, o1_dest_x

    def calc_distance(self, o1_point_a_x, o1_point_a_y, o1_point_b_x, o1_point_b_y):
        return math.sqrt(math.pow(o1_point_a_x - o1_point_b_x, 2) + math.pow(o1_point_a_y - o1_point_b_y, 2))











