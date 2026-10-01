import json

import math

from rcj_soccer_robot import RCJSoccerRobot


class CompatibleRCJSoccerRobot(RCJSoccerRobot):

    def parse_team_msg(self, data: str) -> dict:
        unpacked = json.loads(data)

        if isinstance(unpacked, (list, tuple)) and len(unpacked) == 4:
            return {
                "id": unpacked[0],
                "x": unpacked[1],
                "y": unpacked[2],
                "value": unpacked[3],
            }

        if isinstance(unpacked, dict) and all(
                key in unpacked for key in ("id", "x", "y", "value")):
            return {
                "id": unpacked["id"],
                "x": unpacked["x"],
                "y": unpacked["y"],
                "value": unpacked["value"],
            }

        raise ValueError("Unsupported team packet format")

    def send_data_to_team(self, packet_id, x, y, value=-1) -> list:
        data = [packet_id, x, y, value]
        self.team_emitter.send(json.dumps(data))
        return data

    def get_robot_flipped(self) -> bool:
        compass_values = self.compass.getValues()
        return not (-0.5 < compass_values[2] < 0.5)

    def get_compass_heading_updated(self) -> float:
        """Get compass heading in radians

        Returns:
            float: Compass value in radians
        """
        compass_values = self.compass.getValues()

        # Add math.pi/2 (90) so that the heading 0 is facing opponent's goal
        rad = math.atan2(compass_values[0], compass_values[1]) + (math.pi / 2)
        if rad < -math.pi:
            rad = rad + (2 * math.pi)

        return rad
