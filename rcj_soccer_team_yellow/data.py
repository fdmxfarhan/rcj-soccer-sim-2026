import math
from rcj_soccer_robot import TIME_STEP

class Data:
    # Team-observation freshness is intentionally conservative. Teammate
    # packets are normally received every control cycle (~32 ms), so a 0.5 s
    # window tolerates short communication gaps without treating old data as
    # current forever.
    TEAM_DATA_FRESHNESS_TIMEOUT = 0.5
    BALL_SIGNAL_FRESHNESS_TIMEOUT = 0.5
    BALL_AGREEMENT_DISTANCE = 0.20
    TEAM_CONFIDENCE_CORROBORATION_WEIGHT = 0.25
    team_side = ""
    use_legacy_coords = False
    unit_id = 0
    self_pos_x = 0
    self_pos_y = 0
    heading = 0
    is_flipped = False
    ball_pos_x = 0
    ball_pos_y = 0
    ball_bearing = 0
    ball_range = 0
    ball_visible = False
    prev_robot_x = 0
    prev_robot_y = 0
    prev_robot_heading = 0
    prev_ball_x = 0
    prev_ball_y = 0
    prev_ball_range = 0
    prev_ball_detect = False
    prev_flipped = False
    predicted_ball_x = 0
    predicted_ball_y = 0
    predicted_ball_bearing = 0
    predicted_ball_range = 0
    forecast_prev_ball_x = 0
    forecast_prev_ball_y = 0
    forecast_initialized = False
    forecast_prev_source = None
    prediction_gain = 0
    mate1_id = 0
    mate1_pos_x = 0
    mate1_pos_y = 0
    mate1_heading = 0
    mate1_ball_x = 0
    mate1_ball_y = 0
    mate1_signal = 0
    mate1_ball_visible = False
    mate1_flipped = False
    mate1_ball_range = 0
    mate2_id = 0
    mate2_pos_x = 0
    mate2_pos_y = 0
    mate2_heading = 0
    mate2_ball_x = 0
    mate2_ball_y = 0
    mate2_signal = 0
    mate2_ball_visible = False
    mate2_flipped = False
    mate2_ball_range = 0

    def __init__(self, name, prediction_gain=40, use_legacy_coords=False):
        if "B" in name:
            self.team_side = "B"
        else:
            self.team_side = "Y"

        if "1" in name:
            self.unit_id = 1
            self.mate1_id = 2
            self.mate2_id = 3
        elif "2" in name:
            self.unit_id = 2
            self.mate1_id = 1
            self.mate2_id = 3
        else:
            self.unit_id = 3
            self.mate1_id = 1
            self.mate2_id = 2

        self.prediction_gain = prediction_gain
        self.ball_signal = 0.0
        self.primary_ball_observer = None
        self.team_ball_confidence = 0.0
        self.ball_agreement = 1.0
        self.team_ball_state = "STALE"
        self.forecast_prev_ball_x = 0.0
        self.forecast_prev_ball_y = 0.0
        self.forecast_initialized = False
        self.forecast_prev_source = None

        self.mate1_position_age = math.inf
        self.mate2_position_age = math.inf
        self.mate1_ball_age = math.inf
        self.mate2_ball_age = math.inf
        self.mate1_ball_state = "STALE"
        self.mate2_ball_state = "STALE"
        self._last_update_time = None
        self._mate1_position_time = None
        self._mate2_position_time = None
        self._mate1_ball_time = None
        self._mate2_ball_time = None

        if use_legacy_coords:
            self.use_legacy_coords = True

    def updatePosition(self, robot_pos):
        self.self_pos_x = robot_pos[0]
        self.self_pos_y = robot_pos[1]
        return True

    def updateHeading(self, heading):
        if heading == -1:
            self.is_flipped = True
            self.ball_visible = False
            return False
        else:
            self.is_flipped = False
            if self.team_side == "B":
                self.heading = (-math.degrees(heading) + 180 + 360) % 360
            else:
                self.heading = (-math.degrees(heading) + 360) % 360
            return True

    def updateBallAngle(self, ball_vector):
        ball_radian = math.atan2(ball_vector[0], ball_vector[1]) + (math.pi / 2)
        if ball_radian < -math.pi:
            ball_radian = ball_radian + (2 * math.pi)
        self.ball_bearing = (math.degrees(ball_radian) + 360) % 360
        return True

    def updateBallDistance(self, strength):
        self.ball_range = math.pow(1/strength, 0.5)
        return True

    def updateBallCoords(self):
        if self.team_side == "B":
            heading_radian = math.radians((self.heading + self.ball_bearing) % 360)
            x_distance = math.sin(heading_radian) * self.ball_range
            y_distance = math.cos(heading_radian) * self.ball_range
            ball_pos_x = self.self_pos_x - x_distance
            ball_pos_y = self.self_pos_y - y_distance
        else:
            heading_radian = math.radians((self.heading + self.ball_bearing) % 360)
            x_distance = math.sin(heading_radian) * self.ball_range
            y_distance = math.cos(heading_radian) * self.ball_range
            ball_pos_x = self.self_pos_x + x_distance
            ball_pos_y = self.self_pos_y + y_distance

        self.ball_pos_x = ball_pos_x
        self.ball_pos_y = ball_pos_y
        return True

    def updateBallPredCoords(self):
        active_source = self.unit_id if self.ball_visible else self.receiveActive()
        ball_pos_diff = {'x': self.ball_pos_x - self.forecast_prev_ball_x,
                         'y': self.ball_pos_y - self.forecast_prev_ball_y}

        # The first sample and a large jump after changing observation source are
        # not reliable enough to extrapolate. Use the current position for that
        # frame instead of turning a sensor/teammate handoff into a fake velocity.
        source_changed = (self.forecast_prev_source is not None and
                          active_source is not None and
                          active_source != self.forecast_prev_source)
        jump_distance = math.hypot(ball_pos_diff['x'], ball_pos_diff['y'])
        if (not self.forecast_initialized or
                (source_changed and jump_distance > 0.10) or
                jump_distance > 0.20):
            ball_pos_diff = {'x': 0.0, 'y': 0.0}

        ball_dist = math.hypot(self.self_pos_x - self.ball_pos_x,
                               self.self_pos_y - self.ball_pos_y)
        ball_pos_pred = {'x': self.ball_pos_x + ball_pos_diff['x'] * ball_dist * self.prediction_gain,
                         'y': self.ball_pos_y + ball_pos_diff['y'] * ball_dist * self.prediction_gain}
        self.forecast_prev_ball_x, self.forecast_prev_ball_y = self.ball_pos_x, self.ball_pos_y
        self.forecast_initialized = True
        self.forecast_prev_source = active_source
        if ball_pos_pred['x'] > 0.625:
            ball_pos_pred['x'] = 0.625 - abs(0.625 - ball_pos_pred['x'])
        elif ball_pos_pred['x'] < -0.625:
            ball_pos_pred['x'] = -0.625 + abs(-0.625 - ball_pos_pred['x'])
        if ball_pos_pred['y'] > 0.725:
            ball_pos_pred['y'] = 0.725 - abs(0.725 - ball_pos_pred['y'])
        elif ball_pos_pred['y'] < -0.725:
            ball_pos_pred['y'] = -0.725 + abs(-0.725 - ball_pos_pred['y'])
        self.predicted_ball_x, self.predicted_ball_y = ball_pos_pred['x'], ball_pos_pred['y']
        if self.predicted_ball_x < -0.63:
            self.predicted_ball_x = -0.63
        elif self.predicted_ball_x > 0.63:
            self.predicted_ball_x = 0.63
        if self.predicted_ball_y < -0.74:
            self.predicted_ball_y = -0.74
        elif self.predicted_ball_y > 0.74:
            self.predicted_ball_y = 0.74
        return True

    def updateBallPredDistance(self):
        self.predicted_ball_range = math.sqrt(math.pow(self.predicted_ball_x - self.self_pos_x, 2) + math.pow(self.predicted_ball_y - self.self_pos_y, 2))
        return True

    def updateBallPredAngle(self):
        if self.team_side == "B":
            self.predicted_ball_bearing = (math.degrees(math.atan2(self.predicted_ball_x - self.self_pos_x, self.predicted_ball_y - self.self_pos_y)) - self.heading + 180) % 360
        else:
            self.predicted_ball_bearing = (math.degrees(math.atan2(self.predicted_ball_x - self.self_pos_x, self.predicted_ball_y - self.self_pos_y)) - self.heading + 360) % 360
        return True

    def updatePredN(self, prediction_gain):
        self.prediction_gain = prediction_gain
        return True

    def updateLastValues(self):
        self.prev_robot_x = self.self_pos_x
        self.prev_robot_y = self.self_pos_y
        self.prev_robot_heading = self.heading
        self.prev_ball_x = self.ball_pos_x
        self.prev_ball_y = self.ball_pos_y
        self.prev_ball_range = self.ball_range
        self.prev_ball_detect = self.ball_visible
        self.prev_flipped = self.is_flipped
        return True

    def _update_observation_ages(self, current_time=None):
        if current_time is None:
            return
        self._last_update_time = current_time
        if self._mate1_position_time is None:
            self.mate1_position_age = math.inf
        else:
            self.mate1_position_age = max(0.0, current_time - self._mate1_position_time)
        if self._mate2_position_time is None:
            self.mate2_position_age = math.inf
        else:
            self.mate2_position_age = max(0.0, current_time - self._mate2_position_time)
        if self._mate1_ball_time is None:
            self.mate1_ball_age = math.inf
        else:
            self.mate1_ball_age = max(0.0, current_time - self._mate1_ball_time)
        if self._mate2_ball_time is None:
            self.mate2_ball_age = math.inf
        else:
            self.mate2_ball_age = max(0.0, current_time - self._mate2_ball_time)

        self.mate1_ball_state = self._get_ball_state(self.mate1_ball_visible, self.mate1_ball_age)
        self.mate2_ball_state = self._get_ball_state(self.mate2_ball_visible, self.mate2_ball_age)

    def _get_ball_state(self, visible, age):
        if age > self.BALL_SIGNAL_FRESHNESS_TIMEOUT:
            return "STALE"
        return "FRESH_VISIBLE" if visible else "FRESH_NOT_VISIBLE"

    def _is_fresh_position(self, age):
        return age <= self.TEAM_DATA_FRESHNESS_TIMEOUT

    def _is_fresh_ball(self, age):
        return age <= self.BALL_SIGNAL_FRESHNESS_TIMEOUT

    def _effective_signal(self, signal, age):
        if not self._is_fresh_ball(age):
            return 0.0
        freshness = max(0.0, 1.0 - (age / self.BALL_SIGNAL_FRESHNESS_TIMEOUT))
        return signal * freshness

    def _observation_confidence(self, signal, age):
        """Return a bounded confidence score for one ball observation."""
        effective_signal = self._effective_signal(signal, age)
        if effective_signal <= 0.0:
            return 0.0
        return effective_signal / (effective_signal + 1.0)

    def _get_ball_observations(self):
        """Return fresh visible ball observations as (robot_id, x, y, confidence)."""
        observations = []
        if self.ball_visible:
            observations.append((self.unit_id, self.ball_pos_x, self.ball_pos_y,
                                 self._observation_confidence(self.ball_signal, 0.0)))
        if self.mate1_ball_state == "FRESH_VISIBLE":
            observations.append((self.mate1_id, self.mate1_ball_x, self.mate1_ball_y,
                                 self._observation_confidence(self.mate1_signal, self.mate1_ball_age)))
        if self.mate2_ball_state == "FRESH_VISIBLE":
            observations.append((self.mate2_id, self.mate2_ball_x, self.mate2_ball_y,
                                 self._observation_confidence(self.mate2_signal, self.mate2_ball_age)))
        return observations

    def _update_team_ball_awareness(self):
        """Update C1-C4 awareness data without changing strategy behavior."""
        observations = self._get_ball_observations()
        if not observations:
            self.primary_ball_observer = None
            self.team_ball_confidence = 0.0
            self.ball_agreement = 1.0
            has_fresh_not_visible = (self.mate1_ball_state == "FRESH_NOT_VISIBLE" or
                                     self.mate2_ball_state == "FRESH_NOT_VISIBLE")
            self.team_ball_state = "LOST" if has_fresh_not_visible else "STALE"
            return

        # Preserve the controller's existing priority: a fresh local sensor
        # observation is preferred over teammate observations. If the local
        # sensor cannot see the ball, choose the strongest fresh teammate
        # observation using the same freshness-aware score as receiveActive().
        if self.ball_visible:
            primary = observations[0]
        else:
            primary = max(observations, key=lambda item: (item[3], -item[0]))
        self.primary_ball_observer = primary[0]

        if len(observations) < 2:
            self.ball_agreement = 1.0
        else:
            max_separation = 0.0
            for index, first in enumerate(observations):
                for second in observations[index + 1:]:
                    separation = math.hypot(first[1] - second[1], first[2] - second[2])
                    max_separation = max(max_separation, separation)
            self.ball_agreement = max(0.0,
                                      1.0 - max_separation / self.BALL_AGREEMENT_DISTANCE)

        # The primary observation provides the base confidence. Fresh,
        # agreeing teammate observations add a deliberately small
        # corroboration bonus instead of simply averaging all observations.
        # This keeps one strong observation from being weakened by a weaker
        # teammate, while still rewarding independent confirmation.
        primary_confidence = primary[3]
        secondary_confidence = max(
            (item[3] for item in observations if item[0] != primary[0]),
            default=0.0
        )
        corroboration = (
            (1.0 - primary_confidence) *
            secondary_confidence *
            self.TEAM_CONFIDENCE_CORROBORATION_WEIGHT
        )
        # Confidence measures how trustworthy the available observations are
        # on their own. Observation disagreement is kept separate in
        # ``ball_agreement`` so C2 does not hide the distinct C3 signal. C4
        # uses both values when deciding the overall team state.
        self.team_ball_confidence = min(1.0,
                                        primary_confidence + corroboration)
        if len(observations) >= 2 and self.ball_agreement <= 0.0:
            self.team_ball_state = "CONTESTED"
        elif self.team_ball_confidence >= 0.50:
            self.team_ball_state = "KNOWN"
        else:
            self.team_ball_state = "UNCERTAIN"

    def receiveData(self, team_data, current_time=None):
        if team_data:
            for packet in team_data:
                # ID Meaning
                # single-digit: robot location (1, 2, 3),
                # double-digit: ball location (10, 20, 30),
                id = packet["id"]
                x = packet["x"]
                y = packet["y"]
                value = packet["value"]
                if id == self.mate1_id:
                    self.mate1_pos_x = x
                    self.mate1_pos_y = y
                    if current_time is not None:
                        self._mate1_position_time = current_time
                    if value == -1:
                        self.mate1_flipped = True
                    else:
                        self.mate1_heading = value
                        self.mate1_flipped = False
                    self.mate1_ball_range = math.sqrt(math.pow(self.mate1_pos_x - self.ball_pos_x, 2) + math.pow(self.mate1_pos_y - self.ball_pos_y, 2))
                elif id == self.mate2_id:
                    self.mate2_pos_x = x
                    self.mate2_pos_y = y
                    if current_time is not None:
                        self._mate2_position_time = current_time
                    if value == -1:
                        self.mate2_flipped = True
                    else:
                        self.mate2_heading = value
                        self.mate2_flipped = False
                    self.mate2_ball_range = math.sqrt(math.pow(self.mate2_pos_x - self.ball_pos_x, 2) + math.pow(self.mate2_pos_y - self.ball_pos_y, 2))
                elif id == self.mate1_id * 10:
                    if current_time is not None:
                        self._mate1_ball_time = current_time
                    if value >= 0:
                        self.mate1_ball_visible = True
                        self.mate1_ball_x = x
                        self.mate1_ball_y = y
                        self.mate1_signal = value
                    else:
                        self.mate1_ball_visible = False
                elif id == self.mate2_id * 10:
                    if current_time is not None:
                        self._mate2_ball_time = current_time
                    if value >= 0:
                        self.mate2_ball_visible = True
                        self.mate2_ball_x = x
                        self.mate2_ball_y = y
                        self.mate2_signal = value
                    else:
                        self.mate2_ball_visible = False
            self._update_observation_ages(current_time)
            return True
        self._update_observation_ages(current_time)
        return False

    def receiveActive(self):
        mate1_active = (self.mate1_ball_state == "FRESH_VISIBLE")
        mate2_active = (self.mate2_ball_state == "FRESH_VISIBLE")

        if mate1_active and mate2_active:
            mate1_effective = self._effective_signal(self.mate1_signal, self.mate1_ball_age)
            mate2_effective = self._effective_signal(self.mate2_signal, self.mate2_ball_age)
            if mate1_effective > mate2_effective:
                return self.mate1_id
            return self.mate2_id
        elif mate1_active:
            return self.mate1_id
        elif mate2_active:
            return self.mate2_id
        return None

    def receiveBallCoords(self):
        active = self.receiveActive()
        if active:
            if active == self.mate1_id:
                self.ball_pos_x = self.mate1_ball_x
                self.ball_pos_y = self.mate1_ball_y
            elif active == self.mate2_id:
                self.ball_pos_x = self.mate2_ball_x
                self.ball_pos_y = self.mate2_ball_y
            else:
                return False
            return True
        else:
            return False

    def receiveBallDistance(self):
        active = self.receiveActive()
        if active:
            if active == self.mate1_id:
                x = self.mate1_ball_x
                y = self.mate1_ball_y
            elif active == self.mate2_id:
                x = self.mate2_ball_x
                y = self.mate2_ball_y
            else:
                return False
            self.ball_range = math.sqrt(math.pow(x - self.self_pos_x, 2) + math.pow(y - self.self_pos_y, 2))
            return True
        else:
            return False

    def receiveBallAngle(self):
        active = self.receiveActive()
        if active:
            if active == self.mate1_id:
                x = self.mate1_ball_x
                y = self.mate1_ball_y
            elif active == self.mate2_id:
                x = self.mate2_ball_x
                y = self.mate2_ball_y
            else:
                return False
            if self.team_side == "B":
                self.ball_bearing = (math.degrees(math.atan2(x - self.self_pos_x, y - self.self_pos_y)) - self.heading + 180) % 360
            else:
                self.ball_bearing = (math.degrees(math.atan2(x - self.self_pos_x, y - self.self_pos_y)) - self.heading + 360) % 360
            return True
        else:
            return False

    def getTeam(self):
        return self.team_side

    def getRobotID(self):
        return self.unit_id

    def getRobotPos(self):
        if self.use_legacy_coords:
            return self.self_pos_y, self.self_pos_x
        return self.self_pos_x, self.self_pos_y

    def getRobotHeading(self):
        return self.heading

    def getRobotFlipped(self):
        return self.is_flipped

    def getBallPos(self):
        if self.use_legacy_coords:
            return self.ball_pos_y, self.ball_pos_x
        return self.ball_pos_x, self.ball_pos_y

    def getBallAngle(self):
        return self.ball_bearing

    def getBallDistance(self):
        return self.ball_range

    def getBallPredPos(self):
        if self.use_legacy_coords:
            return self.predicted_ball_y, self.predicted_ball_x
        return self.predicted_ball_x, self.predicted_ball_y

    def getBallPredAngle(self):
        return self.predicted_ball_bearing

    def getBallPredDistance(self):
        return self.predicted_ball_range

    def getLastRobotPos(self):
        if self.use_legacy_coords:
            return self.prev_robot_y, self.prev_robot_x
        return self.prev_robot_x, self.prev_robot_y

    def getLastRobotHeading(self):
        return self.prev_robot_heading

    def getLastBallPos(self):
        if self.use_legacy_coords:
            return self.prev_ball_y, self.prev_ball_x
        return self.prev_ball_x, self.prev_ball_y

    def getLastDetect(self):
        return self.prev_ball_detect

    def getLastFlipped(self):
        return self.prev_flipped

    def getLastBallDistance(self):
        return self.prev_ball_range

    def getBallDetect(self):
        return self.ball_visible

    def getTeammate1ID(self):
        return self.mate1_id

    def getTeammate1Pos(self):
        if self.use_legacy_coords:
            return self.mate1_pos_y, self.mate1_pos_x
        return self.mate1_pos_x, self.mate1_pos_y

    def getTeammate1Heading(self):
        return self.mate1_heading

    def getTeammate1Detect(self):
        return self.mate1_ball_state == "FRESH_VISIBLE"

    def getTeammate1Flipped(self):
        return self.mate1_flipped

    def getTeammate1BallDistance(self):
        return self.mate1_ball_range

    def getTeammate2ID(self):
        return self.mate2_id

    def getTeammate2Pos(self):
        if self.use_legacy_coords:
            return self.mate2_pos_y, self.mate2_pos_x
        return self.mate2_pos_x, self.mate2_pos_y

    def getTeammate2Heading(self):
        return self.mate2_heading

    def getTeammate2Detect(self):
        return self.mate2_ball_state == "FRESH_VISIBLE"

    def getTeammate2Flipped(self):
        return self.mate2_flipped

    def getTeammate2BallDistance(self):
        return self.mate2_ball_range

    def getTeamDetect(self):
        return self.ball_visible or self.receiveActive() is not None

    def getIndividualDetect(self):
        return (self.ball_visible,
                self.mate1_ball_state == "FRESH_VISIBLE",
                self.mate2_ball_state == "FRESH_VISIBLE")

    def getActiveDetect(self):
        if self.ball_visible:
            return self.unit_id
        return self.receiveActive()

    def getPrimaryBallObserver(self):
        return self.primary_ball_observer

    def getTeamBallConfidence(self):
        return self.team_ball_confidence

    def getBallAgreement(self):
        return self.ball_agreement

    def getTeamBallState(self):
        return self.team_ball_state
    
    def getTeammate1PositionAge(self):
        return self.mate1_position_age

    def getTeammate2PositionAge(self):
        return self.mate2_position_age

    def getTeammate1BallAge(self):
        return self.mate1_ball_age

    def getTeammate2BallAge(self):
        return self.mate2_ball_age

    def getTeammate1BallState(self):
        return self.mate1_ball_state

    def getTeammate2BallState(self):
        return self.mate2_ball_state

    def getRobotPosSearch(self, id, last=False):
        if id == self.mate1_id:
            if self.use_legacy_coords:
                return self.mate1_pos_y, self.mate1_pos_x
            return self.mate1_pos_x, self.mate1_pos_y
        elif id == self.mate2_id:
            if self.use_legacy_coords:
                return self.mate2_pos_y, self.mate2_pos_x
            return self.mate2_pos_x, self.mate2_pos_y
        elif id == self.unit_id:
            if last:
                if self.use_legacy_coords:
                    return self.prev_robot_y, self.prev_robot_x
                return self.prev_robot_x, self.prev_robot_y
            else:
                if self.use_legacy_coords:
                    return self.self_pos_y, self.self_pos_x
                return self.self_pos_x, self.self_pos_y
        return None

    def getHeadingSearch(self, id, last=False):
        if id == self.mate1_id:
            return self.mate1_heading
        elif id == self.mate2_id:
            return self.mate2_heading
        elif id == self.unit_id:
            if last:
                return self.prev_robot_heading
            else:
                return self.heading
        return None

    def getDetectSearch(self, id, last=False):
        if id == self.mate1_id:
            return self.mate1_ball_visible
        elif id == self.mate2_id:
            return self.mate2_ball_visible
        elif id == self.unit_id:
            if last:
                return self.prev_ball_detect
            else:
                return self.ball_visible
        return None

    def getFlippedSearch(self, id, last=False):
        if id == self.mate1_id:
            return self.mate1_flipped
        elif id == self.mate2_id:
            return self.mate2_flipped
        elif id == self.unit_id:
            if last:
                return self.prev_flipped
            else:
                return self.is_flipped
        return None
    
    def getBallPosSearch(self, id, last=False):
        if id == self.mate1_id:
            if self.use_legacy_coords:
                return self.mate1_ball_y, self.mate1_ball_x
            return self.mate1_ball_x, self.mate1_ball_y
        elif id == self.mate2_id:
            if self.use_legacy_coords:
                return self.mate2_ball_y, self.mate2_ball_x
            return self.mate2_ball_x, self.mate2_ball_y
        elif id == self.unit_id:
            if last:
                if self.use_legacy_coords:
                    return self.prev_ball_y, self.prev_ball_x
                return self.prev_ball_x, self.prev_ball_y
            else:
                if self.use_legacy_coords:
                    return self.ball_pos_y, self.ball_pos_x
                return self.ball_pos_x, self.ball_pos_y
        return None

    def getBallDistanceSearch(self, id, last=False):
        if id == self.mate1_id:
            return self.mate1_ball_range
        elif id == self.mate2_id:
            return self.mate2_ball_range
        elif id == self.unit_id:
            if last:
                return self.prev_ball_range
            else:
                return self.ball_range
        return None

    def getSearch(self, id, last=False):
        search_robot_x, search_robot_y = self.getRobotPosSearch(id, last)
        search_ball_x, search_ball_y = self.getBallPosSearch(id, last)
        search_heading = self.getHeadingSearch(id, last)
        search_ball_distance = self.getBallDistanceSearch(id, last)
        search_detect = self.getDetectSearch(id, last)
        search_flipped = self.getFlippedSearch(id, last)
        return {"robot_x": search_robot_x,
                "robot_y": search_robot_y,
                "heading": search_heading,
                "ball_x": search_ball_x,
                "ball_y": search_ball_y,
                "ball_distance": search_ball_distance,
                "detect": search_detect,
                "flipped": search_flipped
                }

    def getStandardSearch(self, id, last=False):
        if self.use_legacy_coords:
            search_robot_y, search_robot_x = self.getRobotPosSearch(id, last)
            search_ball_y, search_ball_x = self.getBallPosSearch(id, last)
        else:
            search_robot_x, search_robot_y = self.getRobotPosSearch(id, last)
            search_ball_x, search_ball_y = self.getBallPosSearch(id, last)
        search_heading = self.getHeadingSearch(id, last)
        search_ball_distance = self.getBallDistanceSearch(id, last)
        search_detect = self.getDetectSearch(id, last)
        search_flipped = self.getFlippedSearch(id, last)
        return {"robot_x": search_robot_x,
                "robot_y": search_robot_y,
                "heading": search_heading,
                "ball_x": search_ball_x,
                "ball_y": search_ball_y,
                "ball_distance": search_ball_distance,
                "detect": search_detect,
                "flipped": search_flipped
                }

    def getAll(self):
        if self.use_legacy_coords:
            return {"team": self.team_side,
                    "robot_heading": self.heading,
                    "robot_id": self.unit_id,
                    "robot_pos": {"x": self.self_pos_y, "y": self.self_pos_x},
                    "robot_flipped": self.is_flipped,
                    "ball_pos": {"x": self.ball_pos_y, "y": self.ball_pos_x},
                    "ball_angle": self.ball_bearing,
                    "ball_distance": self.ball_range,
                    "ball_pred_pos": {"x": self.predicted_ball_y, "y": self.predicted_ball_x},
                    "ball_pred_angle": self.predicted_ball_bearing,
                    "ball_pred_distance": self.predicted_ball_range,
                    "ball_detect": self.ball_visible,
                    "last_robot_pos": {"x": self.prev_robot_y, "y": self.prev_robot_x},
                    "last_heading": self.prev_robot_heading,
                    "last_detect": self.prev_ball_detect,
                    "last_flipped": self.prev_flipped,
                    "last_ball_pos": {"x": self.prev_ball_y, "y": self.prev_ball_x},
                    "last_ball_distance": self.prev_ball_range,
                    "teammate1_id": self.mate1_id,
                    "teammate1_pos": {"x": self.mate1_pos_y, "y": self.mate1_pos_x},
                    "teammate1_heading": self.mate1_heading,
                    "teammate1_detect": self.getTeammate1Detect(),
                    "teammate1_position_age": self.mate1_position_age,
                    "teammate1_ball_age": self.mate1_ball_age,
                    "teammate1_ball_state": self.mate1_ball_state,
                    "teammate1_flipped": self.mate1_flipped,
                    "teammate1_ball_pos": {"x": self.mate1_ball_y, "y": self.mate1_ball_x},
                    "teammate1_ball_distance": self.mate1_ball_range,
                    "teammate2_id": self.mate2_id,
                    "teammate2_pos": {"x": self.mate2_pos_y, "y": self.mate2_pos_x},
                    "teammate2_heading": self.mate2_heading,
                    "teammate2_detect": self.getTeammate2Detect(),
                    "teammate2_position_age": self.mate2_position_age,
                    "teammate2_ball_age": self.mate2_ball_age,
                    "teammate2_ball_state": self.mate2_ball_state,
                    "teammate2_flipped": self.mate2_flipped,
                    "teammate2_ball_pos": {"x": self.mate2_ball_y, "y": self.mate2_ball_x},
                    "teammate2_ball_distance": self.mate2_ball_range,
                    "team_detect": self.getTeamDetect(),
                    "primary_ball_observer": self.primary_ball_observer,
                    "team_ball_confidence": self.team_ball_confidence,
                    "ball_agreement": self.ball_agreement,
                    "team_ball_state": self.team_ball_state
                    }
        return {"team": self.team_side,
                "robot_heading": self.heading,
                "robot_id": self.unit_id,
                "robot_pos": {"x": self.self_pos_x, "y": self.self_pos_y},
                "robot_flipped": self.is_flipped,
                "ball_pos": {"x": self.ball_pos_x, "y": self.ball_pos_y},
                "ball_angle": self.ball_bearing,
                "ball_distance": self.ball_range,
                "ball_pred_pos": {"x": self.predicted_ball_x, "y": self.predicted_ball_y},
                "ball_pred_angle": self.predicted_ball_bearing,
                "ball_pred_distance": self.predicted_ball_range,
                "ball_detect": self.ball_visible,
                "last_robot_pos": {"x": self.prev_robot_x, "y": self.prev_robot_y},
                "last_heading": self.prev_robot_heading,
                "last_detect": self.prev_ball_detect,
                "last_flipped": self.prev_flipped,
                "last_ball_pos": {"x": self.prev_ball_x, "y": self.prev_ball_y},
                "last_ball_distance": self.prev_ball_range,
                "teammate1_id": self.mate1_id,
                "teammate1_pos": {"x": self.mate1_pos_x, "y": self.mate1_pos_y},
                "teammate1_heading": self.mate1_heading,
                "teammate1_detect": self.mate1_ball_visible,
                "teammate1_position_age": self.mate1_position_age,
                "teammate1_ball_age": self.mate1_ball_age,
                "teammate1_ball_state": self.mate1_ball_state,
                "teammate1_flipped": self.mate1_flipped,
                "teammate1_ball_pos": {"x": self.mate1_ball_x, "y": self.mate1_ball_y},
                "teammate1_ball_distance": self.mate1_ball_range,
                "teammate2_id": self.mate2_id,
                "teammate2_pos": {"x": self.mate2_pos_x, "y": self.mate2_pos_y},
                "teammate2_heading": self.mate2_heading,
                "teammate2_detect": self.mate2_ball_visible,
                "teammate2_position_age": self.mate2_position_age,
                "teammate2_ball_age": self.mate2_ball_age,
                "teammate2_ball_state": self.mate2_ball_state,
                "teammate2_flipped": self.mate2_flipped,
                "teammate2_ball_pos": {"x": self.mate2_ball_x, "y": self.mate2_ball_y},
                "teammate2_ball_distance": self.mate2_ball_range,
                "team_detect": self.getTeamDetect(),
                "primary_ball_observer": self.primary_ball_observer,
                "team_ball_confidence": self.team_ball_confidence,
                "ball_agreement": self.ball_agreement,
                "team_ball_state": self.team_ball_state
                }

    def convert(self, x, y, coord_system):
        # Coordinate System: Old = 0, New = 1
        if (coord_system == 1 and self.use_legacy_coords) or (coord_system == 0 and not self.use_legacy_coords):
            return y, x
        return x, y

    def process(self, robot_pos, heading, team_data, ball_vector=None, strength=None, current_time=None):
        self.updateLastValues()
        self.updatePosition(robot_pos)
        self.updateHeading(heading)
        receive = self.receiveData(team_data, current_time)
        if ball_vector and strength:
            self.ball_visible = True
            self.updateBallAngle(ball_vector)
            self.ball_signal = strength
            self.updateBallDistance(strength)
            self.updateBallCoords()
            self.updateBallPredCoords()
            self.updateBallPredDistance()
            self.updateBallPredAngle()
            self.ball_visible = True
        else:
            self.ball_visible = False
            if receive and self.receiveActive():
                self.receiveBallCoords()
                self.receiveBallDistance()
                self.receiveBallAngle()
                self.updateBallPredCoords()
                self.updateBallPredDistance()
                self.updateBallPredAngle()
            else:
                self._update_team_ball_awareness()
                return False
        self._update_team_ball_awareness()
        return True