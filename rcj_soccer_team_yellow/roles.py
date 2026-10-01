import math

offense1_role = 1
offense2_role = 2
defense_role = 3

class Roles:
    team = ""
    player_id = ""
    roles = {}
    robot1_data = {}
    robot2_data = {}
    robot3_data = {}
    offense1_data = {}
    offense2_data = {}
    defense_data = {}
    ball_data = {}
    team_detect = False
    trapped_activate = False
    temp_activate = False

    def __init__(self, name):
        if "B" in name:
            self.team = "B"
            self.roles = {1: offense2_role, 2: defense_role, 3: offense1_role}
        else:
            self.team = "Y"
            self.roles = {1: offense1_role, 2: defense_role, 3: offense2_role}
        if "1" in name:
            self.player_id = 1
        elif "2" in name:
            self.player_id = 2
        else:
            self.player_id = 3

    def processData(self, data, trapped_stage):
        self.robot1_data = data.getStandardSearch(1, True)
        self.robot2_data = data.getStandardSearch(2, True)
        self.robot3_data = data.getStandardSearch(3, True)
        self.offense1_data = data.getStandardSearch(self.getRoleToRobot(1), True)
        self.offense2_data = data.getStandardSearch(self.getRoleToRobot(2), True)
        self.defense_data = data.getStandardSearch(self.getRoleToRobot(3), True)
        self.ball_data = {"ball_x": data.getStandardSearch(self.player_id, False)["ball_x"],
                          "ball_y": data.getStandardSearch(self.player_id, False)["ball_y"]}
        self.team_detect = (self.robot1_data["detect"] or self.robot2_data["detect"] or self.robot3_data["detect"])
        if trapped_stage != -1:
            self.trapped_activate = True
        else:
            self.trapped_activate = False
        # print(trapped_stage, self.trapped_activate, self.temp_activate)

    def update(self, data, trapped_stage=-1):
        self.processData(data, trapped_stage)
        # print(self.roles)
        if "B" in self.team:
            if not self.trapped_activate and not self.temp_activate:
                point_x, point_y = 0, 0.58
                offense1_dist = self.getDistance(self.offense1_data['robot_x'], self.offense1_data['robot_y'], point_x,
                                                 point_y)
                offense2_dist = self.getDistance(self.offense2_data['robot_x'], self.offense2_data['robot_y'], point_x,
                                                 point_y)
                defense_dist = self.getDistance(self.defense_data['robot_x'], self.defense_data['robot_y'], point_x,
                                                point_y)
                minimum_dist = min([offense1_dist, offense2_dist, defense_dist])
                if self.defense_data["flipped"] and not self.offense1_data["flipped"] and not self.offense2_data[
                    "flipped"]:
                    if offense1_dist < offense2_dist:
                        self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(defense_role))
                    else:
                        self.swapRoles(self.getRoleToRobot(offense2_role), self.getRoleToRobot(defense_role))
                elif self.defense_data["flipped"] and self.offense2_data["flipped"] and not self.offense1_data[
                    "flipped"]:
                    self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(defense_role))
                elif self.defense_data["flipped"] and self.offense1_data["flipped"] and not self.offense2_data[
                    "flipped"]:
                    self.swapRoles(self.getRoleToRobot(offense2_role), self.getRoleToRobot(defense_role))
                elif self.offense1_data["flipped"] and not self.offense2_data["flipped"] and not self.defense_data[
                    "flipped"] and self.ball_data["ball_x"] <= 0:
                    self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(offense2_role))
                elif self.offense2_data["flipped"] and not self.offense1_data["flipped"] and not self.defense_data[
                    "flipped"] and self.ball_data["ball_x"] > 0:
                    self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(offense2_role))
                elif not self.defense_data["flipped"] and not self.offense1_data["flipped"] and not self.offense2_data[
                    "flipped"]:
                    if self.offense1_data["robot_y"] > 0.67 and self.defense_data["robot_y"] > 0.67 and self.ball_data[
                        "ball_y"] > 0.7 and self.offense1_data["robot_x"] < self.ball_data["ball_x"] < \
                            self.defense_data["robot_x"] and self.ball_data["ball_x"] < -0.22 and self.offense1_data[
                        "robot_y"] < self.ball_data["ball_y"] and self.defense_data["robot_y"] < self.ball_data[
                        "ball_y"] and 0.01 < self.getDistance(self.offense1_data["robot_x"],
                                                              self.offense1_data["robot_y"],
                                                              self.defense_data["robot_x"],
                                                              self.defense_data["robot_y"]) < 0.16:
                        self.swapAllRoles(self.getRoleToRobot(defense_role), self.getRoleToRobot(offense2_role),
                                          self.getRoleToRobot(offense1_role))
                        self.temp_activate = True
                    elif self.offense2_data["robot_y"] > 0.67 and self.defense_data["robot_y"] > 0.67 and \
                            self.ball_data["ball_y"] > 0.7 and self.offense2_data["robot_x"] > self.ball_data[
                        "ball_x"] > self.defense_data["robot_x"] and self.ball_data["ball_x"] > 0.22 and \
                            self.offense2_data["robot_y"] < self.ball_data["ball_y"] and self.defense_data["robot_y"] < \
                            self.ball_data["ball_y"] and 0.01 < self.getDistance(self.defense_data["robot_x"],
                                                                                 self.defense_data["robot_y"],
                                                                                 self.offense2_data["robot_x"],
                                                                                 self.offense2_data["robot_y"]) < 0.16:
                        self.swapAllRoles(self.getRoleToRobot(defense_role), self.getRoleToRobot(offense1_role),
                                          self.getRoleToRobot(offense2_role))
                        self.temp_activate = True
                    elif self.ball_data["ball_x"] < -0.25 and self.ball_data["ball_y"] > 0.65 and self.defense_data[
                        "robot_x"] > self.offense2_data["robot_x"] and self.offense2_data["robot_y"] > \
                            self.defense_data["robot_y"] and self.offense2_data["robot_x"] < -0.15 and \
                            self.offense2_data["robot_y"] > 0.6:
                        self.swapAllRoles(self.getRoleToRobot(defense_role), self.getRoleToRobot(offense1_role),
                                          self.getRoleToRobot(offense2_role))
                    elif self.ball_data["ball_x"] > 0.25 and self.ball_data["ball_y"] > 0.65 and self.defense_data[
                        "robot_x"] < self.offense1_data["robot_x"] and self.offense1_data["robot_y"] > \
                            self.defense_data["robot_y"] and self.offense1_data["robot_x"] > 0.15 and \
                            self.offense1_data["robot_y"] > 0.6:
                        self.swapAllRoles(self.getRoleToRobot(defense_role), self.getRoleToRobot(offense2_role),
                                          self.getRoleToRobot(offense1_role))
                    elif (self.defense_data["robot_y"] < 0.2 or self.defense_data["robot_x"] > 0.45 or
                          self.defense_data["robot_x"] < -0.45) and minimum_dist == offense1_dist:
                        self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(defense_role))
                    elif (self.defense_data["robot_y"] < 0.2 or self.defense_data["robot_x"] > 0.45 or
                          self.defense_data["robot_x"] < -0.45) and minimum_dist == offense2_dist:
                        self.swapRoles(self.getRoleToRobot(offense2_role), self.getRoleToRobot(defense_role))
                    elif self.ball_data["ball_y"] < 0.4 and self.ball_data["ball_x"] > 0.15 and self.offense1_data[
                        "robot_y"] < 0.6 and self.defense_data["robot_y"] > 0.64 and self.offense2_data[
                        "robot_y"] > 0.64 and minimum_dist == offense1_dist:
                        self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(defense_role))
                    elif self.ball_data["ball_y"] < 0.4 and self.ball_data["ball_x"] < -0.15 and self.offense2_data[
                        "robot_y"] < 0.6 and self.defense_data["robot_y"] > 0.64 and self.offense1_data[
                        "robot_y"] > 0.64 and minimum_dist == offense2_dist:
                        self.swapRoles(self.getRoleToRobot(offense2_role), self.getRoleToRobot(defense_role))
                    elif self.ball_data["ball_x"] < -0.25 and self.ball_data["ball_y"] > 0.68 and self.defense_data[
                        "robot_x"] < self.offense1_data["robot_x"]:
                        self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(defense_role))
                    elif self.ball_data["ball_x"] > 0.25 and self.ball_data["ball_y"] > 0.68 and self.defense_data[
                        "robot_x"] > self.offense2_data["robot_x"]:
                        self.swapRoles(self.getRoleToRobot(offense2_role), self.getRoleToRobot(defense_role))
                    elif self.rangePosition(self.robot1_data["robot_x"], self.robot1_data["robot_y"], 0.3, 0.3, 0.1,
                                            0.1) and \
                            self.rangePosition(self.robot2_data["robot_x"], self.robot2_data["robot_y"], -0.3, 0.3, 0.1,
                                               0.1) and \
                            (self.rangePosition(self.robot3_data["robot_x"], self.robot3_data["robot_y"], 0, 0.3, 0.1,
                                                0.1) or
                             self.rangePosition(self.robot3_data["robot_x"], self.robot3_data["robot_y"], 0, 0.1, 0.1,
                                                0.1)):
                        self.resetRoles(self.team, True)
                    elif self.ball_data["ball_x"] > 0 and self.ball_data["ball_y"] < -0.64 and self.offense1_data[
                        "robot_y"] < -0.64 and self.offense2_data["robot_y"] < -0.64 and \
                            self.offense1_data["robot_x"] < 0 and 0 > self.offense2_data["robot_x"] > \
                            self.offense1_data["robot_x"]:
                        self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(offense2_role))
                    elif self.ball_data["ball_x"] < 0 and self.ball_data["ball_y"] < -0.64 and self.offense1_data[
                        "robot_y"] < -0.64 and self.offense2_data["robot_y"] < -0.64 and \
                            self.offense1_data["robot_x"] > 0 and 0 < self.offense2_data["robot_x"] < \
                            self.offense1_data["robot_x"]:
                        self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(offense2_role))
                    elif self.offense1_data["robot_x"] < -0.2 and self.offense2_data["robot_x"] < -0.2 and \
                            self.offense1_data["robot_y"] > -0.56 and self.offense2_data["robot_y"] > -0.56 and \
                            self.offense1_data["robot_x"] > self.offense2_data["robot_x"] and self.getDistance(
                        self.offense1_data["robot_x"], self.offense1_data["robot_y"], self.offense2_data["robot_x"],
                        self.offense2_data["robot_y"]) < 0.2:
                        self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(offense2_role))
                    elif self.offense1_data["robot_x"] > 0.2 and self.offense2_data["robot_x"] > 0.2 and \
                            self.offense1_data["robot_y"] > -0.56 and self.offense2_data["robot_y"] > -0.56 and \
                            self.offense2_data["robot_x"] < self.offense1_data["robot_x"] and self.getDistance(
                        self.offense1_data["robot_x"], self.offense1_data["robot_y"], self.offense2_data["robot_x"],
                        self.offense2_data["robot_y"]) < 0.2:
                        self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(offense2_role))
                    elif self.offense1_data["ball_distance"] > 0.3 and self.offense1_data["ball_distance"] > \
                            self.offense2_data["ball_distance"] + 0.1 and self.ball_data[
                        "ball_x"] < -0.1 and self.getDistance(self.offense1_data["robot_x"],
                                                              self.offense1_data["robot_y"],
                                                              self.offense2_data["robot_x"],
                                                              self.offense2_data["robot_y"]) >= 0.2 and \
                            self.ball_data[
                                "ball_y"] > -0.4 and self.offense1_data[
                        "robot_y"] > -0.4 and self.offense2_data[
                        "robot_y"] > -0.4 and (self.offense1_data[
                                                   "robot_y"] > 0 or self.offense2_data[
                                                   "robot_y"] > 0):
                        self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(offense2_role))
                    elif self.offense2_data["ball_distance"] > 0.3 and self.offense2_data["ball_distance"] > \
                            self.offense1_data["ball_distance"] + 0.1 and self.ball_data[
                        "ball_x"] > 0.1 and self.getDistance(self.offense1_data["robot_x"],
                                                             self.offense1_data["robot_y"],
                                                             self.offense2_data["robot_x"],
                                                             self.offense2_data["robot_y"]) >= 0.2 and \
                            self.ball_data[
                                "ball_y"] > -0.4 and self.offense1_data[
                        "robot_y"] > -0.4 and self.offense2_data[
                        "robot_y"] > -0.4 and (self.offense1_data[
                                                   "robot_y"] > 0 or self.offense2_data[
                                                   "robot_y"] > 0):
                        self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(offense2_role))
            elif self.temp_activate:
                self.temp_activate = False
        else:
            if not self.trapped_activate and not self.temp_activate:
                point_x, point_y = 0, -0.58
                offense1_dist = self.getDistance(self.offense1_data['robot_x'], self.offense1_data['robot_y'], point_x,
                                                 point_y)
                offense2_dist = self.getDistance(self.offense2_data['robot_x'], self.offense2_data['robot_y'], point_x,
                                                 point_y)
                defense_dist = self.getDistance(self.defense_data['robot_x'], self.defense_data['robot_y'], point_x,
                                                point_y)
                minimum_dist = min([offense1_dist, offense2_dist, defense_dist])
                if self.defense_data["flipped"] and not self.offense1_data["flipped"] and not self.offense2_data[
                    "flipped"]:
                    if offense1_dist < offense2_dist:
                        self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(defense_role))
                    else:
                        self.swapRoles(self.getRoleToRobot(offense2_role), self.getRoleToRobot(defense_role))
                elif self.defense_data["flipped"] and self.offense2_data["flipped"] and not self.offense1_data[
                    "flipped"]:
                    self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(defense_role))
                elif self.defense_data["flipped"] and self.offense1_data["flipped"] and not self.offense2_data[
                    "flipped"]:
                    self.swapRoles(self.getRoleToRobot(offense2_role), self.getRoleToRobot(defense_role))
                elif self.offense1_data["flipped"] and not self.offense2_data["flipped"] and not self.defense_data[
                    "flipped"] and self.ball_data["ball_x"] <= 0:
                    self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(offense2_role))
                elif self.offense2_data["flipped"] and not self.offense1_data["flipped"] and not self.defense_data[
                    "flipped"] and self.ball_data["ball_x"] > 0:
                    self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(offense2_role))
                elif not self.defense_data["flipped"] and not self.offense1_data["flipped"] and not self.offense2_data[
                    "flipped"]:
                    if self.offense1_data["robot_y"] < -0.67 and self.defense_data["robot_y"] < -0.67 and self.ball_data[
                        "ball_y"] < -0.7 and self.offense1_data["robot_x"] < self.ball_data["ball_x"] < \
                            self.defense_data["robot_x"] and self.ball_data["ball_x"] < -0.22 and self.offense1_data[
                        "robot_y"] > self.ball_data["ball_y"] and self.defense_data["robot_y"] > self.ball_data[
                        "ball_y"] and 0.01 < self.getDistance(self.offense1_data["robot_x"],
                                                              self.offense1_data["robot_y"],
                                                              self.defense_data["robot_x"],
                                                              self.defense_data["robot_y"]) < 0.16:
                        self.swapAllRoles(self.getRoleToRobot(defense_role), self.getRoleToRobot(offense2_role),
                                          self.getRoleToRobot(offense1_role))
                        self.temp_activate = True
                    elif self.offense2_data["robot_y"] < -0.67 and self.defense_data["robot_y"] < -0.67 and \
                            self.ball_data["ball_y"] < -0.7 and self.offense2_data["robot_x"] > self.ball_data[
                        "ball_x"] > self.defense_data["robot_x"] and self.ball_data["ball_x"] > 0.22 and \
                            self.offense2_data["robot_y"] > self.ball_data["ball_y"] and self.defense_data["robot_y"] > \
                            self.ball_data["ball_y"] and 0.01 < self.getDistance(self.defense_data["robot_x"],
                                                                                 self.defense_data["robot_y"],
                                                                                 self.offense2_data["robot_x"],
                                                                                 self.offense2_data["robot_y"]) < 0.16:
                        self.swapAllRoles(self.getRoleToRobot(defense_role), self.getRoleToRobot(offense1_role),
                                          self.getRoleToRobot(offense2_role))
                        self.temp_activate = True
                    elif self.ball_data["ball_x"] < -0.25 and self.ball_data["ball_y"] < -0.65 and self.defense_data[
                        "robot_x"] > self.offense2_data["robot_x"] and self.offense2_data["robot_y"] < \
                            self.defense_data["robot_y"] and self.offense2_data["robot_x"] < -0.15 and \
                            self.offense2_data["robot_y"] < -0.6:
                        self.swapAllRoles(self.getRoleToRobot(defense_role), self.getRoleToRobot(offense1_role),
                                          self.getRoleToRobot(offense2_role))
                    elif self.ball_data["ball_x"] > 0.25 and self.ball_data["ball_y"] < -0.65 and self.defense_data[
                        "robot_x"] < self.offense1_data["robot_x"] and self.offense1_data["robot_y"] < \
                            self.defense_data["robot_y"] and self.offense1_data["robot_x"] > 0.15 and \
                            self.offense1_data["robot_y"] < -0.6:
                        self.swapAllRoles(self.getRoleToRobot(defense_role), self.getRoleToRobot(offense2_role),
                                          self.getRoleToRobot(offense1_role))
                    elif (self.defense_data["robot_y"] > -0.2 or self.defense_data["robot_x"] > 0.45 or
                          self.defense_data["robot_x"] < -0.45) and minimum_dist == offense1_dist:
                        self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(defense_role))
                    elif (self.defense_data["robot_y"] > -0.2 or self.defense_data["robot_x"] > 0.45 or
                          self.defense_data["robot_x"] < -0.45) and minimum_dist == offense2_dist:
                        self.swapRoles(self.getRoleToRobot(offense2_role), self.getRoleToRobot(defense_role))
                    elif self.ball_data["ball_y"] > -0.4 and self.ball_data["ball_x"] > 0.15 and self.offense1_data[
                        "robot_y"] > -0.6 and self.defense_data["robot_y"] < -0.64 and self.offense2_data[
                        "robot_y"] < -0.64 and minimum_dist == offense1_dist:
                        self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(defense_role))
                    elif self.ball_data["ball_y"] > -0.4 and self.ball_data["ball_x"] < -0.15 and self.offense2_data[
                        "robot_y"] > -0.6 and self.defense_data["robot_y"] < -0.64 and self.offense1_data[
                        "robot_y"] < -0.64 and minimum_dist == offense2_dist:
                        self.swapRoles(self.getRoleToRobot(offense2_role), self.getRoleToRobot(defense_role))
                    elif self.ball_data["ball_x"] < -0.25 and self.ball_data["ball_y"] < -0.68 and self.defense_data[
                        "robot_x"] < self.offense1_data["robot_x"]:
                        self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(defense_role))
                    elif self.ball_data["ball_x"] > 0.25 and self.ball_data["ball_y"] < -0.68 and self.defense_data[
                        "robot_x"] > self.offense2_data["robot_x"]:
                        self.swapRoles(self.getRoleToRobot(offense2_role), self.getRoleToRobot(defense_role))
                    elif self.rangePosition(self.robot1_data["robot_x"], self.robot1_data["robot_y"], -0.3, -0.3, 0.1,
                                            0.1) and \
                            self.rangePosition(self.robot2_data["robot_x"], self.robot2_data["robot_y"], 0.3, -0.3, 0.1,
                                               0.1) and \
                            (self.rangePosition(self.robot3_data["robot_x"], self.robot3_data["robot_y"], 0, -0.3, 0.1,
                                                0.1) or
                             self.rangePosition(self.robot3_data["robot_x"], self.robot3_data["robot_y"], 0, -0.1, 0.1,
                                                0.1)):
                        self.resetRoles(self.team, True)
                    elif self.ball_data["ball_x"] > 0 and self.ball_data["ball_y"] > 0.64 and self.offense1_data[
                        "robot_y"] > 0.64 and self.offense2_data["robot_y"] > 0.64 and \
                            self.offense1_data["robot_x"] < 0 and 0 > self.offense2_data["robot_x"] > \
                            self.offense1_data["robot_x"]:
                        self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(offense2_role))
                    elif self.ball_data["ball_x"] < 0 and self.ball_data["ball_y"] > 0.64 and self.offense1_data[
                        "robot_y"] > 0.64 and self.offense2_data["robot_y"] > 0.64 and \
                            self.offense1_data["robot_x"] > 0 and 0 < self.offense2_data["robot_x"] < \
                            self.offense1_data["robot_x"]:
                        self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(offense2_role))
                    elif self.offense1_data["robot_x"] < -0.2 and self.offense2_data["robot_x"] < -0.2 and \
                            self.offense1_data["robot_y"] < 0.56 and self.offense2_data["robot_y"] < 0.56 and \
                            self.offense1_data["robot_x"] > self.offense2_data["robot_x"] and self.getDistance(
                        self.offense1_data["robot_x"], self.offense1_data["robot_y"], self.offense2_data["robot_x"],
                        self.offense2_data["robot_y"]) < 0.2:
                        self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(offense2_role))
                    elif self.offense1_data["robot_x"] > 0.2 and self.offense2_data["robot_x"] > 0.2 and \
                            self.offense1_data["robot_y"] < 0.56 and self.offense2_data["robot_y"] < 0.56 and \
                            self.offense2_data["robot_x"] < self.offense1_data["robot_x"] and self.getDistance(
                        self.offense1_data["robot_x"], self.offense1_data["robot_y"], self.offense2_data["robot_x"],
                        self.offense2_data["robot_y"]) < 0.2:
                        self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(offense2_role))
                    elif self.offense1_data["ball_distance"] > 0.3 and self.offense1_data["ball_distance"] > \
                            self.offense2_data["ball_distance"] + 0.1 and self.ball_data[
                        "ball_x"] < -0.1 and self.getDistance(self.offense1_data["robot_x"],
                                                              self.offense1_data["robot_y"],
                                                              self.offense2_data["robot_x"],
                                                              self.offense2_data["robot_y"]) >= 0.2 and \
                            self.ball_data[
                                "ball_y"] < 0.4 and self.offense1_data[
                        "robot_y"] < 0.4 and self.offense2_data[
                        "robot_y"] < 0.4 and (self.offense1_data[
                                                   "robot_y"] < 0 or self.offense2_data[
                                                   "robot_y"] < 0):
                        self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(offense2_role))
                    elif self.offense2_data["ball_distance"] > 0.3 and self.offense2_data["ball_distance"] > \
                            self.offense1_data["ball_distance"] + 0.1 and self.ball_data[
                        "ball_x"] > 0.1 and self.getDistance(self.offense1_data["robot_x"],
                                                             self.offense1_data["robot_y"],
                                                             self.offense2_data["robot_x"],
                                                             self.offense2_data["robot_y"]) >= 0.2 and \
                            self.ball_data[
                                "ball_y"] < 0.4 and self.offense1_data[
                        "robot_y"] < 0.4 and self.offense2_data[
                        "robot_y"] < 0.4 and (self.offense1_data[
                                                   "robot_y"] < 0 or self.offense2_data[
                                                   "robot_y"] < 0):
                        self.swapRoles(self.getRoleToRobot(offense1_role), self.getRoleToRobot(offense2_role))
            elif self.temp_activate:
                self.temp_activate = False

    def swapRoles(self, id_1, id_2):
        temp_id_1_role = self.roles[id_1]
        self.roles[id_1] = self.roles[id_2]
        self.roles[id_2] = temp_id_1_role

    def swapAllRoles(self, id_1, id_2, id_3):
        temp_id_1_role = self.roles[id_1]
        temp_id_2_role = self.roles[id_2]
        temp_id_3_role = self.roles[id_3]
        self.roles[id_1] = temp_id_3_role
        self.roles[id_2] = temp_id_1_role
        self.roles[id_3] = temp_id_2_role

    def resetRoles(self, team, conditional=False):
        if conditional:
            if "B" in team:
                if self.robot3_data["robot_x"] < 0:
                    self.roles = {1: defense_role, 2: offense1_role, 3: offense2_role}
                else:
                    self.roles = {1: offense2_role, 2: defense_role, 3: offense1_role}
            else:
                if self.robot3_data["robot_x"] < 0:
                    self.roles = {1: offense1_role, 2: defense_role, 3: offense2_role}
                else:
                    self.roles = {1: defense_role, 2: offense2_role, 3: offense1_role}
        else:
            if "B" in team:
                self.roles = {1: offense2_role, 2: defense_role, 3: offense1_role}
            else:
                self.roles = {1: offense1_role, 2: defense_role, 3: offense2_role}

    def rangePosition(self, pos_x, pos_y, dest_x, dest_y, range_x=0.2, range_y=0.2, range_x_up=None, range_x_down=None,
                      range_y_up=None, range_y_down=None):
        if range_x_up is None:
            range_x_up = range_x
        if range_x_down is None:
            range_x_down = range_x
        if range_y_up is None:
            range_y_up = range_y
        if range_y_down is None:
            range_y_down = range_y
        if pos_x > dest_x + range_x_up or pos_x < dest_x - range_x_down or pos_y > dest_y + range_y_up or pos_y < dest_y - range_y_down:
            return False
        return True

    def receiveRoles(self, robot1_role, robot2_role, robot3_role):
        self.roles = self.roles = {1: robot1_role, 2: robot2_role, 3: robot3_role}

    def getRoles(self):
        return self.roles

    def getRobotToRole(self, id):
        if id == 1:
            return self.roles[1]
        elif id == 2:
            return self.roles[2]
        else:
            return self.roles[3]

    def getRoleToRobot(self, role):
        if role == self.roles[1]:
            return 1
        elif role == self.roles[2]:
            return 2
        else:
            return 3

    def getDistance(self, point_a_x, point_a_y, point_b_x, point_b_y):
        return math.sqrt(math.pow(point_a_x - point_b_x, 2) + math.pow(point_a_y - point_b_y, 2))
