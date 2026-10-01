import math
import traceback

from rcj_soccer_robot import TIME_STEP
from rcj_soccer_robot_compat import CompatibleRCJSoccerRobot
from data import Data
from offense1_yellow import OffenseYellow1
from offense2_yellow import OffenseYellow2
from defense_yellow import DefenseYellow
from progress import Progress
from roles import Roles

class MyRobot(CompatibleRCJSoccerRobot):
    def _stop_motors_safely(self):
        """Stop both drive motors after an unexpected controller exception."""
        try:
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
        except Exception:
            # If the motor devices themselves are unavailable, there is no
            # additional safe action we can take here.
            pass

    def _handle_runtime_error(self):
        """Fail safe for one control cycle and report the real exception.

        The old bare ``except: pass`` hid every controller error and could
        leave the motors running with the previous command.  We still keep
        the controller alive so a transient error does not terminate the
        match, but we stop the motors and print a throttled traceback.
        """
        self._stop_motors_safely()

        try:
            current_time = self.robot.getTime()
        except Exception:
            current_time = 0.0

        last_report_time = getattr(self, "_last_error_report_time", -math.inf)
        if current_time - last_report_time >= 1.0:
            self._last_error_report_time = current_time
            print("[MyRobot] Runtime error; motors stopped for this cycle:")
            traceback.print_exc()

    def run(self):
        if "Y" in self.name:
            offense1 = OffenseYellow1()
            offense2 = OffenseYellow2()
            defense = DefenseYellow()

        data = Data(self.name, 80)
        progress1 = Progress(self.name, math.ceil(15 / (TIME_STEP / 1000.0)), math.ceil(10 / (TIME_STEP / 1000.0)), 0.5,
                            math.ceil(10 / (TIME_STEP / 1000.0)), math.ceil(2 / (TIME_STEP / 1000.0)), 0.5 * 0.25)
        progress2 = Progress(self.name, math.ceil(15 / (TIME_STEP / 1000.0)), math.ceil(10 / (TIME_STEP / 1000.0)), 0.5,
                            math.ceil(10 / (TIME_STEP / 1000.0)), math.ceil(2 / (TIME_STEP / 1000.0)), 0.5 * 0.2)
        roles = Roles(self.name)
        last_role = 0
        self._last_error_report_time = -math.inf
        self.role_packet_age = math.inf
        last_role_packet_time = None
        while self.robot.step(TIME_STEP) != -1:
            try:
                current_time = self.robot.getTime()
                if last_role_packet_time is None:
                    self.role_packet_age = math.inf
                else:
                    self.role_packet_age = max(0.0, current_time - last_role_packet_time)

                if self.is_new_data():
                    team_data = []
                    while self.is_new_team_data():
                        team_data.append(self.get_new_team_data())

                    if self.is_new_ball_data() and not self.get_robot_flipped():
                        ball_data = self.get_new_ball_data()
                        data.process(self.get_gps_coordinates(), self.get_compass_heading_updated(), team_data, ball_data["direction"], ball_data["strength"], current_time)
                        ball_x, ball_y = data.getBallPos()
                        ball_x, ball_y = data.convert(ball_x, ball_y, 1)
                        self.send_data_to_team(self.player_id * 10, ball_x, ball_y, ball_data["strength"])
                    elif self.is_new_ball_data() and self.get_robot_flipped():
                        self.get_new_ball_data()
                        data.process(self.get_gps_coordinates(), -1, team_data, current_time=current_time)
                        self.send_data_to_team(self.player_id * 10, 0, 0)
                    else:
                        data.process(self.get_gps_coordinates(), self.get_compass_heading_updated(), team_data, current_time=current_time)
                        self.send_data_to_team(self.player_id * 10, 0, 0)

                    progress1.update(data.getAll())
                    progress2.update(data.getAll())

                    if self.player_id != 3:
                        for packet in team_data:
                            if packet["id"] == -1:
                                if roles.receiveRoles(packet["x"], packet["y"], packet["value"]):
                                    last_role_packet_time = current_time
                                    self.role_packet_age = 0.0

                    MyRobot.resetProgram(self, last_role, roles, offense1, offense2, defense)
                    last_role = roles.getRoles()[self.player_id]

                    trap_stage = MyRobot.updateTrapStage(self, team_data, roles, offense1, offense2)

                    MyRobot.runProgram(self, self, data, progress1, progress2, roles, offense1, offense2, defense, trap_stage)

                    trap_stage = MyRobot.updateTrapStage(self, team_data, roles, offense1, offense2)

                    if self.player_id == 3:
                        roles.update(data, trap_stage)
                        self.send_data_to_team(-1, roles.getRobotToRole(1), roles.getRobotToRole(2),
                                               roles.getRoleToRobot(3))

                    role = roles.getRoles()[self.player_id]
                    if role == 1:
                        self.send_data_to_team(self.player_id * 100, offense1.o1_trap_stage, 0, 0)
                    elif role == 2:
                        self.send_data_to_team(self.player_id * 100, offense2.o2_trap_stage, 0, 0)

                    robot_x, robot_y = data.getRobotPos()
                    robot_x, robot_y = data.convert(robot_x, robot_y, 1)
                    robot_angle = data.getRobotHeading()

                    if self.get_robot_flipped():
                        self.send_data_to_team(self.player_id, robot_x, robot_y, -1)
                    else:
                        self.send_data_to_team(self.player_id, robot_x, robot_y, robot_angle)
            except Exception:
                self._handle_runtime_error()

    def getRolePacketAge(self):
        return self.role_packet_age

    def runProgram(self, robot, data, progress1, progress2, roles, offense1, offense2, defense, trap_stage):
        role = roles.getRoles()[self.player_id]
        if role == 1:
            offense1.execute_strategy(robot, data, progress1.getProgress(), progress2.getProgress(), roles, trap_stage)
        elif role == 2:
            offense2.execute_strategy_two(robot, data, progress1.getProgress(), progress2.getProgress(), roles, trap_stage)
        else:
            defense.execute_defense(robot, data, progress1.getProgress(), progress2.getProgress(), roles)

    def resetProgram(self, last_role, roles, offense1, offense2, defense):
        role = roles.getRoles()[self.player_id]
        if role != last_role:
            if role == 1:
                offense1.reset_strategy()
            elif role == 2:
                offense2.reset_strategy_two()
            else:
                defense.reset_defense()

    def updateTrapStage(self, team_data, roles, offense1, offense2):
        trap_stage = -1
        for packet in team_data:
            if (packet["id"] == 100 or packet["id"] == 200 or packet["id"] == 300) and packet["x"] != -1:
                trap_stage = packet["x"]
        if trap_stage != -1:
            return trap_stage
        role = roles.getRoles()[self.player_id]
        if role == 1 and offense1.o1_trap_stage != -1:
            return offense1.o1_trap_stage
        elif role == 2 and offense2.o2_trap_stage != -1:
            return offense2.o2_trap_stage
        return trap_stage

