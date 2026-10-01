import numpy as np
import math
import time

df_front_defense_mode = 0
df_diagonal_left_defense_mode = 1
df_diagonal_right_defense_mode = 2
df_wall_spin_defense_mode = 3
df_wall_side_defense_mode = 4

# Geometry/control tolerances kept intentionally small.
DF_PREDICTION_DX_EPSILON = 1e-5
DF_DIAGONAL_X = -0.44
DF_DIAGONAL_Y = 0.35
DF_DIAGONAL_SLOPE = 1.4
DF_ROBOT_WALL_MIN_X = -0.65
DF_ROBOT_WALL_MAX_X = -0.57
DF_COVERAGE_BIAS_LIMIT = 0.10
DF_COVERAGE_MIN_Y = -0.32
DF_COVERAGE_MAX_Y = 0.32

class DefenseYellow:
    df_mode = 0
    df_switch_mode = False
    df_last_ball_pos = {'x': 0.0, 'y': 0.0}
    df_pred_n = 40
    df_prev_mode = 0
    df_face = 1
    df_goalline_range = 0.07
    df_set_time = time.time_ns()

    # Run Yellow
    def execute_defense(self, df_robot, df_data, df_lop1, df_lop2, df_roles):
        df_robot_angle = df_data.getRobotHeading()
        df_robot_y, df_robot_x = df_data.getRobotPos()
        df_ball_y, df_ball_x = df_data.getBallPos()
        df_offense1_y, df_offense1_x = df_data.getRobotPosSearch(df_roles.getRoleToRobot(1))
        df_offense2_y, df_offense2_x = df_data.getRobotPosSearch(df_roles.getRoleToRobot(2))
        df_defense_y, df_defense_x = df_data.getRobotPosSearch(df_roles.getRoleToRobot(3))
        df_detect = df_data.getTeamDetect()

        # C8: when an offensive robot is the trusted primary observer,
        # bias the defender slightly toward the uncovered side of the field.
        # This is intentionally capped so the defender still follows the
        # existing ball/goal-line logic rather than abandoning the ball.
        df_primary_observer = df_data.getPrimaryBallObserver()
        df_team_ball_state = df_data.getTeamBallState()
        df_coverage_bias = 0.0
        if df_team_ball_state == "KNOWN" and df_primary_observer in (
                df_roles.getRoleToRobot(1), df_roles.getRoleToRobot(2)):
            df_primary_robot = df_primary_observer
            if df_primary_robot == df_roles.getRoleToRobot(1):
                df_primary_y = df_offense1_y
                df_primary_position_age = df_data.getTeammate1PositionAge() if df_primary_robot != getattr(df_data, "unit_id", 0) else 0.0
            else:
                df_primary_y = df_offense2_y
                df_primary_position_age = df_data.getTeammate2PositionAge() if df_primary_robot != getattr(df_data, "unit_id", 0) else 0.0
            if df_primary_position_age <= df_data.TEAM_DATA_FRESHNESS_TIMEOUT and abs(df_primary_y) > 0.18:
                df_coverage_bias = -0.10 if df_primary_y > 0 else 0.10

        df_robot_angle = (-df_robot_angle + 90) % 360

        df_robot_pos = {'x': df_robot_x, "y": df_robot_y}
        df_ball_pos = {'x': df_ball_x, "y": df_ball_y}

        df_ball_pos_diff = {'x': df_ball_pos['x'] - self.df_last_ball_pos['x'],
                         'y': df_ball_pos['y'] - self.df_last_ball_pos['y']}
        df_ball_dist = math.sqrt((df_robot_pos['x'] - df_ball_pos['x']) ** 2 + (df_robot_pos['y'] - df_ball_pos['y']) ** 2)

        self.df_pred_n = DefenseYellow.scale_defense(self, df_ball_dist, 30, 50, 0, 0.8)

        df_ball_pos_pred = {'x': df_ball_pos['x'] + df_ball_pos_diff['x'] * df_ball_dist * self.df_pred_n,
                         'y': df_ball_pos['y'] + df_ball_pos_diff['y'] * df_ball_dist * self.df_pred_n}

        if abs(df_ball_pos_diff['x']) > DF_PREDICTION_DX_EPSILON:
            df_ball_pos_pred = {'x': df_ball_pos['x'] + df_ball_pos_diff['x'] * df_ball_dist * self.df_pred_n,
                             'y': df_ball_pos['y'] + df_ball_pos_diff['y'] / df_ball_pos_diff['x'] * (-0.54 - df_ball_pos['x'])}

            if df_ball_pos_pred['y'] > 0.63:
                # ball_pos_pred['y'] = ball_pos_pred['y'] - (ball_pos_pred['y'] - 0.63)
                df_ball_pos_pred['y'] = 0.63
            if df_ball_pos_pred['y'] < -0.63:
                # ball_pos_pred['y'] = ball_pos_pred['y'] - (ball_pos_pred['y'] + 0.63)
                df_ball_pos_pred['y'] = -0.63
        else:
            pass
        # else:
        # ball_pos_pred['y'] = ball_pos['y']

        # print(ball_pos_pred['y'], " ", ball_pos['y'])

        self.df_last_ball_pos = df_ball_pos

        df_spin_speed = 6

        df_left_speed = 0
        df_right_speed = 0

        # Adjust Angle Range
        if df_ball_x > 0:
            df_angle_range = 12
        else:
            df_angle_range = 15

        # Mode Switching
        if abs(df_ball_pos_diff['y']) <= 0.0045 and df_ball_pos['x'] < -0.44 and (
                df_ball_pos['y'] > 0.35 or df_ball_pos['y'] < -0.35):
            df_Xdiag1 = -(df_ball_pos['y'] - DF_DIAGONAL_Y) / DF_DIAGONAL_SLOPE + DF_DIAGONAL_X
            df_Xdiag2 = (df_ball_pos['y'] + DF_DIAGONAL_Y) / DF_DIAGONAL_SLOPE + DF_DIAGONAL_X
            df_Ydiag1 = -DF_DIAGONAL_SLOPE * (df_ball_pos['x'] - DF_DIAGONAL_X) + DF_DIAGONAL_Y
            df_Ydiag2 = DF_DIAGONAL_SLOPE * (df_ball_pos['x'] - DF_DIAGONAL_X) - DF_DIAGONAL_Y
        else:
            df_Xdiag1 = -0.44
            df_Xdiag2 = -0.44
            df_Ydiag1 = 0.35
            df_Ydiag2 = -0.35

        if df_ball_pos['x'] < -0.66 and 0.24 > df_ball_pos['y'] > -0.24 and df_robot_pos[
            'x'] < -0.65 and self.df_prev_mode != df_front_defense_mode:
            self.df_mode = df_wall_spin_defense_mode
        elif df_ball_pos['x'] < -0.66 and (df_ball_pos['y'] >= 0.24 or df_ball_pos['y'] <= -0.24):
            self.df_mode = df_wall_side_defense_mode
        elif (df_ball_pos['x'] < df_Xdiag1 and df_ball_pos['y'] > df_Ydiag1):
            self.df_mode = df_diagonal_left_defense_mode
        elif (df_ball_pos['x'] < df_Xdiag2 and df_ball_pos['y'] < df_Ydiag2):
            self.df_mode = df_diagonal_right_defense_mode
        else:
            self.df_mode = df_front_defense_mode

        if (
                self.df_mode == df_wall_side_defense_mode or self.df_mode == df_diagonal_left_defense_mode or self.df_mode == df_diagonal_right_defense_mode) and (
                df_ball_dist > 0.161 and abs(df_ball_pos_diff['y']) < 0.001):
            if time.time_ns() - self.df_set_time >= 2000000000:
                self.df_mode = df_front_defense_mode
        else:
            self.df_set_time = time.time_ns()

        # Wall Defense In Penalty Area (With Deflection)
        if self.df_mode == df_wall_spin_defense_mode:
            df_angle = DefenseYellow.pick_direction_def(self, df_robot_angle, 90, 270)
            if (df_robot_y > -0.21 and df_robot_y < 0) and (df_robot_x < -0.7) and (
                    (df_robot_angle < 180 + df_angle_range and df_robot_angle > 180 - df_angle_range) or (
                    df_robot_angle < 0 + df_angle_range or df_robot_angle > 360 - df_angle_range)):
                df_angle = DefenseYellow.pick_direction_def(self, df_robot_angle, 0, 180)
                df_left_speed, df_right_speed = DefenseYellow.move_def(self, df_robot_angle, 10, 10, df_angle,
                                                           0, False)
            elif (df_robot_y < 0.21 and df_robot_y >= 0) and (df_robot_x < -0.7) and (
                    (df_robot_angle < 180 + df_angle_range and df_robot_angle > 180 - df_angle_range) or (
                    df_robot_angle < 0 + df_angle_range or df_robot_angle > 360 - df_angle_range)):
                df_angle = DefenseYellow.pick_direction_def(self, df_robot_angle, 0, 180)
                df_left_speed, df_right_speed = DefenseYellow.move_def(self, df_robot_angle, -10, -10, df_angle,
                                                           0, False)
            elif df_robot_y > -0.21 and df_robot_y < 0:
                if df_angle == 90:
                    df_left_speed = 5
                    df_right_speed = 10
                else:
                    df_left_speed = -10
                    df_right_speed = -5
            elif df_robot_y < 0.21 and df_robot_y >= 0:
                if df_angle == 90:
                    df_left_speed = 10
                    df_right_speed = 5
                else:
                    df_left_speed = -5
                    df_right_speed = -10
            else:
                df_angle = DefenseYellow.pick_direction_def(self, df_robot_angle, 90, 270)
                if df_robot_angle > df_angle + df_angle_range or df_robot_angle < df_angle - df_angle_range:
                    df_left_speed, df_right_speed = DefenseYellow.turn_to_heading_def(self, df_robot_angle, df_angle, df_left_speed,
                                                               df_right_speed,
                                                               df_spin_speed, df_angle_range)
                else:
                    df_left_speed, df_right_speed = DefenseYellow.move_def(self, df_robot_angle, 10, 10, df_angle,
                                                               90, False)
                    df_left_speed, df_right_speed = DefenseYellow.bound_x(self, df_robot_angle, df_robot_pos['x'],
                                                                   df_left_speed,
                                                                   df_right_speed, 0.718, -0.718, 0,
                                                                   df_return_speed=0)
                    if abs(df_ball_pos['y'] - df_robot_pos['y']) > 0.2:
                        df_left_speed, df_right_speed = DefenseYellow.bound_x(self, df_robot_angle, df_robot_pos['x'],
                                                                       df_left_speed,
                                                                       df_right_speed, 0.695, -0.695, 0.05, -10)

        # Wall Defense Outside Of Penalty Area (With Going To Wall Defense Point)
        elif self.df_mode == df_wall_side_defense_mode:
            if (df_robot_y <= 0.23 or df_robot_y > 0.38) and df_ball_y >= 0.28:
                df_left_speed, df_right_speed = DefenseYellow.drive_to_point_def(self, df_left_speed, df_right_speed, -0.67, 0.3,
                                                              df_robot_pos['x'], df_robot_pos['y'], df_robot_angle,
                                                              df_spin_speed, df_angle_range)
            elif (df_robot_y >= -0.23 or df_robot_y < -0.38) and df_ball_y <= -0.28:
                df_left_speed, df_right_speed = DefenseYellow.drive_to_point_def(self, df_left_speed, df_right_speed, -0.67, -0.3,
                                                              df_robot_pos['x'], df_robot_pos['y'], df_robot_angle,
                                                              df_spin_speed, df_angle_range)
            else:
                df_angle = DefenseYellow.pick_direction_def(self, df_robot_angle, 90, 270)
                if df_robot_angle > df_angle + df_angle_range or df_robot_angle < df_angle - df_angle_range:
                    df_left_speed, df_right_speed = DefenseYellow.turn_to_heading_def(self, df_robot_angle, df_angle, df_left_speed,
                                                               df_right_speed,
                                                               df_spin_speed, df_angle_range)
                else:
                    df_left_speed, df_right_speed = DefenseYellow.move_def(self, df_robot_angle, 10, 10, df_angle,
                                                               90, False)
                    df_left_speed, df_right_speed = DefenseYellow.bound_x(self, df_robot_angle, df_robot_pos['x'],
                                                                   df_left_speed,
                                                                   df_right_speed, 0.718, -0.718, 0, df_return_speed=0)
                    if abs(df_ball_pos['y'] - df_robot_pos['y']) > 0.2:
                        df_left_speed, df_right_speed = DefenseYellow.bound_x(self, df_robot_angle, df_robot_pos['x'],
                                                                       df_left_speed,
                                                                       df_right_speed, 0.695, -0.695, 0.05, -10)

        # Diagonal Ball Tracking On The Robot's Left Side (Blue)
        elif self.df_mode == df_diagonal_left_defense_mode:
            if not (DF_ROBOT_WALL_MIN_X <= df_robot_x <= DF_ROBOT_WALL_MAX_X):
                df_left_speed, df_right_speed = DefenseYellow.drive_to_point_def(self, df_left_speed, df_right_speed, -0.64, 0.28,
                                                              df_robot_pos['x'], df_robot_pos['y'], df_robot_angle,
                                                              df_spin_speed, df_angle_range)
            else:
                df_angle = DefenseYellow.pick_direction_def(self, df_robot_angle, 120, 300)
                if df_robot_angle > df_angle + df_angle_range or df_robot_angle < df_angle - df_angle_range:
                    df_left_speed, df_right_speed = DefenseYellow.turn_to_heading_def(self, df_robot_angle, df_angle, df_left_speed, df_right_speed,
                                                               df_spin_speed,
                                                               df_angle_range)
                else:
                    _, df_solution_y = DefenseYellow.solve_defense_point(self, df_robot_angle, df_robot_x, df_robot_y, df_ball_x,
                                                                   df_ball_pos_pred['y'],
                                                                   -0.745, 0)
                    df_left_speed, df_right_speed = DefenseYellow.chase_ball(self, df_robot_angle, df_robot_pos['y'],
                                                                df_solution_y)
                    df_left_speed, df_right_speed = DefenseYellow.bound_y(self, df_robot_angle, df_robot_pos['y'],
                                                                   df_left_speed, df_right_speed, -0.39, 0.39, 0.05, 10)

        # Diagonal Ball Tracking On The Robot's Right Side (Blue)
        elif self.df_mode == df_diagonal_right_defense_mode:
            if not (DF_ROBOT_WALL_MIN_X <= df_robot_x <= DF_ROBOT_WALL_MAX_X):
                df_left_speed, df_right_speed = DefenseYellow.drive_to_point_def(self, df_left_speed, df_right_speed, -0.64, -0.28,
                                                              df_robot_pos['x'], df_robot_pos['y'], df_robot_angle,
                                                              df_spin_speed, df_angle_range)
            else:
                df_angle = DefenseYellow.pick_direction_def(self, df_robot_angle, 60, 240)
                if df_robot_angle > df_angle + df_angle_range or df_robot_angle < df_angle - df_angle_range:
                    df_left_speed, df_right_speed = DefenseYellow.turn_to_heading_def(self, df_robot_angle, df_angle, df_left_speed,
                                                               df_right_speed, df_spin_speed, df_angle_range)
                else:
                    _, df_solution_y = DefenseYellow.solve_defense_point(self, df_robot_angle, df_robot_x, df_robot_y, df_ball_x,
                                                                   df_ball_pos_pred['y'], -0.745, 0)
                    df_left_speed, df_right_speed = DefenseYellow.chase_ball(self, df_robot_angle, df_robot_pos['y'],
                                                                df_solution_y)
                    df_left_speed, df_right_speed = DefenseYellow.bound_y(self, df_robot_angle, df_robot_pos['y'],
                                                                   df_left_speed, df_right_speed, -0.39, 0.39, 0.05,
                                                                   10)

        # Regular Ball Tracking In A Straight Line
        else:
            if abs(-0.56 - df_robot_pos['x']) > self.df_goalline_range:
                self.df_goalline_range = 0.01

                if df_ball_pos['x'] > 0:
                    df_border = 0.24
                else:
                    df_border = 0.30
                df_temp_ball_pos_y = df_ball_pos['y'] + df_coverage_bias
                df_temp_ball_pos_y = max(DF_COVERAGE_MIN_Y, min(DF_COVERAGE_MAX_Y, df_temp_ball_pos_y))
                if df_temp_ball_pos_y > df_border:
                    df_temp_ball_pos_y = df_border
                elif df_temp_ball_pos_y < -df_border:
                    df_temp_ball_pos_y = -df_border

                df_left_speed, df_right_speed = DefenseYellow.drive_to_point_def(self, df_left_speed, df_right_speed, -0.56, df_temp_ball_pos_y,
                                                              df_robot_pos['x'], df_robot_pos['y'], df_robot_angle,
                                                              df_spin_speed, df_angle_range)
            # stay parallel to goal
            else:
                if df_robot_angle > 180:
                    df_robot_angle -= 360
                if df_ball_pos['x'] > 0:
                    df_border = 0.24
                else:
                    df_border = 0.30
                self.df_goalline_range = 0.01

                if df_robot_angle < 178 and df_robot_angle >= 90:
                    df_left_speed = 10 - (df_robot_angle - 90) * 9.5 / 90
                    df_right_speed = -10 + (df_robot_angle - 90) * 9.5 / 90
                    self.df_face = 1
                elif df_robot_angle > -178 and df_robot_angle <= -90:
                    df_left_speed = -10 - (df_robot_angle + 90) * 9.5 / 90
                    df_right_speed = 10 + (df_robot_angle + 90) * 9.5 / 90
                    self.df_face = 1
                elif df_robot_angle < 90 and df_robot_angle > 2:
                    df_left_speed = -10 + (9 - df_robot_angle * 9.5 / 90)
                    df_right_speed = 10 - (9 - df_robot_angle * 9.5 / 90)
                    self.df_face = -1
                elif df_robot_angle > -90 and df_robot_angle < -2:
                    df_left_speed = 10 - (9 + df_robot_angle * 9.5 / 90)
                    df_right_speed = -10 + (9 + df_robot_angle * 9.5 / 90)
                    self.df_face = -1
                # follow ball  
                elif df_robot_pos['y'] > df_border:
                    df_left_speed = -10 * self.df_face
                    df_right_speed = -10 * self.df_face
                elif df_robot_pos['y'] < -df_border:
                    df_left_speed = 10 * self.df_face
                    df_right_speed = 10 * self.df_face
                else:
                    if df_ball_pos['x'] < -0.4:
                        if df_robot_pos['y'] < df_ball_pos['y']:
                            df_left_speed = 10 * self.df_face
                            df_right_speed = 10 * self.df_face
                        elif df_robot_pos['y'] > df_ball_pos['y']:
                            df_left_speed = -10 * self.df_face
                            df_right_speed = -10 * self.df_face
                        else:
                            df_left_speed = 0
                            df_right_speed = 0
                    else:
                        if df_robot_pos['y'] < df_ball_pos_pred['y']:
                            df_left_speed = 10 * self.df_face
                            df_right_speed = 10 * self.df_face
                        elif df_robot_pos['y'] > df_ball_pos_pred['y']:
                            df_left_speed = -10 * self.df_face
                            df_right_speed = -10 * self.df_face
                        else:
                            df_left_speed = 0
                            df_right_speed = 0

            if df_left_speed > 10:
                df_left_speed = 10
            elif df_left_speed < -10:
                df_left_speed = -10
            if df_right_speed > 10:
                df_right_speed = 10
            elif df_right_speed < -10:
                df_right_speed = -10

        self.df_prev_mode = self.df_mode

        df_robot.left_motor.setVelocity(df_left_speed)
        df_robot.right_motor.setVelocity(df_right_speed)

        # Checks if the location of the ball can be found
        if df_detect:
            pass
        else:
            pass

    def reset_defense(self):
        self.df_mode = 0
        self.df_switch_mode = False
        self.df_last_ball_pos = {'x': 0.0, 'y': 0.0}
        self.df_pred_n = 40
        self.df_prev_mode = 0

    # Add any additional functions here
    def relative_coords_def(self, df_robot_angle, df_robot_x, df_robot_y, df_dest_x, df_dest_y):
        df_angleA = (math.degrees(math.atan2(df_robot_x - df_dest_x, df_robot_y - df_dest_y)) + 270) % 360
        df_angleB = (math.degrees(math.atan2(df_robot_x - df_dest_x, df_robot_y - df_dest_y)) + 180) % 360
        if df_robot_angle - df_angleA >= 0:
            df_leftA_total = df_robot_angle - df_angleA
        else:
            df_leftA_total = df_robot_angle + 360 - df_angleA
        if df_angleA - df_robot_angle >= 0:
            df_rightA_total = df_angleA - df_robot_angle
        else:
            df_rightA_total = df_angleA + 360 - df_robot_angle
        if df_robot_angle - df_angleB >= 0:
            df_leftB_total = df_robot_angle - df_angleB
        else:
            df_leftB_total = df_robot_angle + 360 - df_angleB
        if df_angleB - df_robot_angle >= 0:
            df_rightB_total = df_angleB - df_robot_angle
        else:
            df_rightB_total = df_angleB + 360 - df_robot_angle
        if df_leftA_total < df_rightA_total:
            df_a_close = df_leftA_total
        else:
            df_a_close = df_rightA_total
        if df_leftB_total < df_rightB_total:
            df_b_close = df_leftB_total
        else:
            df_b_close = df_rightB_total
        if df_a_close < df_b_close:
            df_angle = df_angleA
        else:
            df_angle = df_angleB
        return df_angle, df_angleA

    def drive_to_point_def(self, df_left_speed, df_right_speed, df_dest_x, df_dest_y, df_robot_pos_x, df_robot_pos_y, df_robot_angle, df_spin_speed,
                df_angle_range):
        df_angle, df_absolute_angle = DefenseYellow.relative_coords_def2(self, df_robot_angle, df_robot_pos_x, df_robot_pos_y,
                                                    df_dest_x,
                                                    df_dest_y)
        df_movement_range = df_angle_range + 1.0
        if df_robot_angle > df_angle + df_movement_range or df_robot_angle < df_angle - df_movement_range:
            df_left_speed, df_right_speed = DefenseYellow.turn_to_heading_def(self, df_robot_angle, df_angle, df_left_speed,
                                                       df_right_speed,
                                                       df_spin_speed, df_angle_range)
        else:
            df_left_speed, df_right_speed = DefenseYellow.move_def(self, df_robot_angle, 10, 10, df_angle,
                                                       df_absolute_angle,
                                                       False)
        return df_left_speed, df_right_speed

    def turn_to_heading_def(self, df_robot_angle, df_angle, df_left_speed, df_right_speed, df_spin_speed, df_angle_range, df_spin_speedL=None,
             df_spin_speedR=None):
        if df_robot_angle - df_angle >= 0:
            df_left_total = df_robot_angle - df_angle
        else:
            df_left_total = df_robot_angle + 360 - df_angle
        if df_angle - df_robot_angle >= 0:
            df_right_total = df_angle - df_robot_angle
        else:
            df_right_total = df_angle + 360 - df_robot_angle
        if df_robot_angle < df_angle - df_angle_range or df_robot_angle > df_angle + df_angle_range:
            if (df_left_total > df_right_total):
                if df_spin_speedL != None and df_spin_speedR != None:
                    df_left_speed = df_spin_speedL
                    df_right_speed = df_spin_speedR
                else:
                    df_left_speed = df_spin_speed
                    df_right_speed = -df_spin_speed
            else:
                if df_spin_speedL != None and df_spin_speedR != None:
                    df_left_speed = df_spin_speedL
                    df_right_speed = df_spin_speedR
                else:
                    df_left_speed = -df_spin_speed
                    df_right_speed = df_spin_speed
        # print("Spin")
        return df_left_speed, df_right_speed

    def pick_direction_def(self, df_robot_angle, df_angleA, df_angleB):
        if df_robot_angle - df_angleA >= 0:
            df_leftA_total = df_robot_angle - df_angleA
        else:
            df_leftA_total = df_robot_angle + 360 - df_angleA
        if df_angleA - df_robot_angle >= 0:
            df_rightA_total = df_angleA - df_robot_angle
        else:
            df_rightA_total = df_angleA + 360 - df_robot_angle
        if df_robot_angle - df_angleB >= 0:
            df_leftB_total = df_robot_angle - df_angleB
        else:
            df_leftB_total = df_robot_angle + 360 - df_angleB
        if df_angleB - df_robot_angle >= 0:
            df_rightB_total = df_angleB - df_robot_angle
        else:
            df_rightB_total = df_angleB + 360 - df_robot_angle
        if df_leftA_total < df_rightA_total:
            df_a_close = df_leftA_total
        else:
            df_a_close = df_rightA_total
        if df_leftB_total < df_rightB_total:
            df_b_close = df_leftB_total
        else:
            df_b_close = df_rightB_total
        if df_a_close < df_b_close:
            df_angle = df_angleA
        else:
            df_angle = df_angleB
        # print("Direction")
        return df_angle

    def move_def(self, df_robot_angle, df_left_run, df_right_run, df_relative, df_absolute, df_straight):
        if df_straight == True:
            df_angle = DefenseYellow.pick_direction_def(self, df_robot_angle, 0, 180)
            if df_angle == 180:
                df_left_speed = df_left_run
                df_right_speed = df_right_run
            else:
                df_left_run2 = df_left_run
                df_left_speed = -df_right_run
                df_right_speed = -df_left_run2
        else:
            df_angle = DefenseYellow.pick_direction_def(self, df_robot_angle, df_relative, df_absolute)
            if df_angle == df_absolute:
                df_left_speed = df_left_run
                df_right_speed = df_right_run
            else:
                df_left_run2 = df_left_run
                df_left_speed = -df_right_run
                df_right_speed = -df_left_run2
        # print("Move")
        return df_left_speed, df_right_speed

    def relative_coords_def2(self, df_robot_angle, df_robot_x, df_robot_y, df_dest_x, df_dest_y):
        df_angleA = (math.degrees(math.atan2(df_robot_x - df_dest_x, df_robot_y - df_dest_y))) % 360
        df_angleB = (math.degrees(math.atan2(df_robot_x - df_dest_x, df_robot_y - df_dest_y)) + 180) % 360
        if df_robot_angle - df_angleA >= 0:
            df_leftA_total = df_robot_angle - df_angleA
        else:
            df_leftA_total = df_robot_angle + 360 - df_angleA
        if df_angleA - df_robot_angle >= 0:
            df_rightA_total = df_angleA - df_robot_angle
        else:
            df_rightA_total = df_angleA + 360 - df_robot_angle
        if df_robot_angle - df_angleB >= 0:
            df_leftB_total = df_robot_angle - df_angleB
        else:
            df_leftB_total = df_robot_angle + 360 - df_angleB
        if df_angleB - df_robot_angle >= 0:
            df_rightB_total = df_angleB - df_robot_angle
        else:
            df_rightB_total = df_angleB + 360 - df_robot_angle
        if df_leftA_total < df_rightA_total:
            df_a_close = df_leftA_total
        else:
            df_a_close = df_rightA_total
        if df_leftB_total < df_rightB_total:
            df_b_close = df_leftB_total
        else:
            df_b_close = df_rightB_total
        if df_a_close < df_b_close:
            df_angle = df_angleA
        else:
            df_angle = df_angleB
        # print("Coordinates")
        return df_angle, df_angleA

    def drive_to_point_def2(self, df_left_speed, df_right_speed, df_dest_x, df_dest_y, df_robot_pos_x, df_robot_pos_y, df_robot_angle):
        df_spin_speed = 6
        df_angle_range = 15
        df_angle, df_absolute_angle = DefenseYellow.relative_coords_def2(self, df_robot_angle, df_robot_pos_x, df_robot_pos_y, df_dest_x, df_dest_y)
        if df_robot_angle > df_angle + df_angle_range or df_robot_angle < df_angle - df_angle_range:
            df_left_speed, df_right_speed = DefenseYellow.turn_to_heading_def(self, df_robot_angle, df_angle, df_left_speed,
                                                       df_right_speed,
                                                       df_spin_speed, df_angle_range)
        else:
            df_left_speed, df_right_speed = DefenseYellow.move_def(self, df_robot_angle, 10, 10, df_angle,
                                                       df_absolute_angle,
                                                       False)
        return df_left_speed, df_right_speed

    def chase_ball(self, df_robot_angle, df_robot_y, df_ball_y):
        if df_ball_y < df_robot_y:
            df_left_speed, df_right_speed = DefenseYellow.move_def(self, df_robot_angle, -7, -7, 0, 0, True)
        elif df_ball_y > df_robot_y:
            df_left_speed, df_right_speed = DefenseYellow.move_def(self, df_robot_angle, 7, 7, 0, 0, True)
        else:
            df_left_speed, df_right_speed = DefenseYellow.move_def(self, df_robot_angle, 0, 0, 0, 0, True)
        # print("Chase")
        return df_left_speed, df_right_speed

    def bound_x(self, df_robot_angle, df_robot_x, df_left_speed, df_right_speed, df_limit1, df_limit2, df_given_range, df_return_speed=10):
        if df_limit2 < df_limit1:
            df_limitA = df_limit2
            df_limitB = df_limit1
        elif df_limit1 < df_limit2:
            df_limitA = df_limit1
            df_limitB = df_limit2
        else:
            return df_left_speed, df_right_speed
        if df_robot_x < df_limitA:
            df_left_speed, df_right_speed = DefenseYellow.move_def(self, df_robot_angle, df_return_speed, df_return_speed, 0, 0, True)
        elif df_robot_x > df_limitB:
            df_left_speed, df_right_speed = DefenseYellow.move_def(self, df_robot_angle, -df_return_speed, -df_return_speed, 0, 0, True)
        elif df_robot_x > df_limitA and df_robot_x < df_limitA + df_given_range and df_left_speed < 0:
            df_angle = DefenseYellow.pick_direction_def(self, df_robot_angle, 0, 180)
            if (df_angle == 180 and df_left_speed < 0 or df_angle == 0 and df_left_speed > 0):
                df_left_speed = 1
                df_right_speed = 1
                if df_return_speed == 0:
                    df_left_speed = 0
                    df_right_speed = 0
        elif df_robot_x < df_limitB and df_robot_x > df_limitB - df_given_range and df_left_speed > 0:
            df_angle = DefenseYellow.pick_direction_def(self, df_robot_angle, 0, 180)
            if (df_angle == 180 and df_left_speed > 0 or df_angle == 0 and df_left_speed < 0):
                df_left_speed = -1
                df_right_speed = -1
                if df_return_speed == 0:
                    df_left_speed = 0
                    df_right_speed = 0
        return df_left_speed, df_right_speed

    def bound_y(self, df_robot_angle, df_robot_y, df_left_speed, df_right_speed, df_limit1, df_limit2, df_given_range, df_return_speed=10):
        if df_limit2 < df_limit1:
            df_limitA = df_limit2
            df_limitB = df_limit1
        elif df_limit1 < df_limit2:
            df_limitA = df_limit1
            df_limitB = df_limit2
        else:
            return df_left_speed, df_right_speed
        if df_robot_y < df_limitA:
            df_left_speed, df_right_speed = DefenseYellow.move_def(self, df_robot_angle, df_return_speed, df_return_speed, 0, 0, True)
        elif df_robot_y > df_limitB:
            df_left_speed, df_right_speed = DefenseYellow.move_def(self, df_robot_angle, -df_return_speed, -df_return_speed, 0, 0, True)
        elif df_robot_y > df_limitA and df_robot_y < df_limitA + df_given_range and df_left_speed < 0:
            df_angle = DefenseYellow.pick_direction_def(self, df_robot_angle, 0, 180)
            if (df_angle == 180 and df_left_speed < 0 or df_angle == 0 and df_left_speed > 0):
                df_left_speed = 1
                df_right_speed = 1
        elif df_robot_y < df_limitB and df_robot_y > df_limitB - df_given_range and df_left_speed > 0:
            df_angle = DefenseYellow.pick_direction_def(self, df_robot_angle, 0, 180)
            if (df_angle == 180 and df_left_speed > 0 or df_angle == 0 and df_left_speed < 0):
                df_left_speed = -1
                df_right_speed = -1
        # print("Border Y")
        return df_left_speed, df_right_speed

    def solve_defense_point(self, df_robot_angle, df_robot_x, df_robot_y, df_point_x, df_point_y, df_dest_x, df_dest_y):
        df_robot_x, df_robot_y = df_robot_x, -df_robot_y
        df_point_x, df_point_y = df_point_x, -df_point_y
        df_dest_x, df_dest_y = df_dest_x, -df_dest_y
        df_robot_angle = (df_robot_angle + 270) % 360
        df_robot_radian = math.radians(df_robot_angle)

        df_robot_slope = math.tan(df_robot_radian)
        df_robot_intercept = df_robot_y - df_robot_slope * df_robot_x
        df_point_angle = math.atan2(df_point_y, df_point_x - df_dest_x)
        df_point_slope = math.tan(df_point_angle)
        df_point_intercept = df_point_y - df_point_slope * df_point_x

        df_left_matrix = np.array([[-df_robot_slope, 1], [-df_point_slope, 1]])
        df_right_matrix = np.array([df_robot_intercept, df_point_intercept])
        df_solution = np.linalg.solve(df_left_matrix, df_right_matrix).tolist()
        df_solution_x, df_solution_y = df_solution[0], df_solution[1]
        # print("Find Point")
        return df_solution_x, -df_solution_y

    def scale_defense(self, df_dist, df_minScale, df_maxScale, df_minLimit, df_maxLimit):
        if df_dist >= df_maxLimit:
            return df_maxScale
        elif df_dist <= df_minLimit:
            return df_minScale
        else:
            df_scale = (df_maxScale - df_minScale) / (df_maxLimit - df_minLimit)
            df_pred = df_scale * df_dist + df_minScale
        return df_pred