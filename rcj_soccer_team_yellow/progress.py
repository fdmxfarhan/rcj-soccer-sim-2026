import math

class Progress:

    def __init__(self, name, robot_steps, ball_steps, threshold, robot_steps_set, ball_steps_set, threshold_set):
        if "B" in name:
            self.team = "B"
        else:
            self.team = "Y"

        self.name = name
        self.threshold = threshold
        self.threshold_set = threshold_set
        self.robot_steps = robot_steps
        self.ball_steps = ball_steps
        self.robot_steps_set = robot_steps_set
        self.ball_steps_set = ball_steps_set
        self.states = {self.team + "1": False, self.team + "2": False, self.team + "3": False, "ball": False}
        self.states_set = {self.team + "1": False, self.team + "2": False, self.team + "3": False, "ball": False}
        self.samples = {self.team + "1": [0 for _ in range(self.robot_steps)],
                        self.team + "2": [0 for _ in range(self.robot_steps)],
                        self.team + "3": [0 for _ in range(self.robot_steps)],
                        "ball": [0 for _ in range(self.ball_steps)]}
        self.samples_set = {self.team + "1": [0 for _ in range(self.robot_steps_set)],
                        self.team + "2": [0 for _ in range(self.robot_steps_set)],
                        self.team + "3": [0 for _ in range(self.robot_steps_set)],
                        "ball": [0 for _ in range(self.ball_steps_set)]}
        self.iterators = {self.team + "1": 0, self.team + "2": 0, self.team + "3": 0, "ball": 0}
        self.previous_pos = {}
        self._initialized = False

    def _get_position(self, data, player):
        if str(data["robot_id"]) in player:
            return data["robot_pos"]
        if str(data["teammate1_id"]) in player:
            return data["teammate1_pos"]
        if str(data["teammate2_id"]) in player:
            return data["teammate2_pos"]
        return data["ball_pos"]

    def track(self, data):
        if not self._initialized:
            for player in self.samples:
                self.previous_pos[player] = self._get_position(data, player)
            self._initialized = True
            return

        for player in self.samples:
            current_pos = self._get_position(data, player)
            prev_pos = self.previous_pos[player]
            delta = math.hypot(current_pos["x"] - prev_pos["x"],
                               current_pos["y"] - prev_pos["y"])

            if player == "ball":
                self.samples[player][self.iterators[player] % self.ball_steps] = delta
                self.samples_set[player][self.iterators[player] % self.ball_steps_set] = delta
            else:
                self.samples[player][self.iterators[player] % self.robot_steps] = delta
                self.samples_set[player][self.iterators[player] % self.robot_steps_set] = delta

            self.previous_pos[player] = current_pos
            self.iterators[player] += 1

    def update(self, data):
        for player in self.samples:
            if self.states[player]:
                self.reset(player)
        self.track(data)
        for player in self.samples:
            s = sum(self.samples[player])
            if player == "ball":
                if self.iterators[player] < self.ball_steps:
                    self.states[player] = False
                else:
                    self.states[player] = (s < self.threshold)
            else:
                if self.iterators[player] < self.robot_steps:
                    self.states[player] = False
                else:
                    self.states[player] = (s < self.threshold)
        for player in self.samples_set:
            s = sum(self.samples_set[player])
            if player == "ball":
                if self.iterators[player] < self.ball_steps_set:
                    self.states_set[player] = False
                else:
                    self.states_set[player] = (s < self.threshold_set)
            else:
                if self.iterators[player] < self.robot_steps_set:
                    self.states_set[player] = False
                else:
                    self.states_set[player] = (s < self.threshold_set)
        return self.states_set

    def reset(self, player):
        self.states[player] = False
        self.states_set[player] = False
        self.iterators[player] = 0
        if player == "ball":
            self.samples[player] = [0 for _ in range(self.ball_steps)]
            self.samples_set[player] = [0 for _ in range(self.ball_steps_set)]
        else:
            self.samples[player] = [0 for _ in range(self.robot_steps)]
            self.samples_set[player] = [0 for _ in range(self.robot_steps_set)]

    def resetAll(self):
        self.states = {self.team + "1": False, self.team + "2": False, self.team + "3": False, "ball": False}
        self.states_set = {self.team + "1": False, self.team + "2": False, self.team + "3": False, "ball": False}
        self.samples = {self.team + "1": [0 for _ in range(self.robot_steps)],
                        self.team + "2": [0 for _ in range(self.robot_steps)],
                        self.team + "3": [0 for _ in range(self.robot_steps)],
                        "ball": [0 for _ in range(self.ball_steps)]}
        self.samples_set = {self.team + "1": [0 for _ in range(self.robot_steps_set)],
                            self.team + "2": [0 for _ in range(self.robot_steps_set)],
                            self.team + "3": [0 for _ in range(self.robot_steps_set)],
                            "ball": [0 for _ in range(self.ball_steps_set)]}
        self.iterators = {self.team + "1": 0, self.team + "2": 0, self.team + "3": 0, "ball": 0}

    def getProgress(self):
        return self.states_set
