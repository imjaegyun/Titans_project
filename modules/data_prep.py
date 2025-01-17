# Titans_project/modules/data_prep.py

def build_scene_description(row):
    track_id = int(row.get('track_id', 0))
    lane = row.get('lane', 'Unknown')
    first_xy = (row.get('x1', 0.0), row.get('y1', 0.0))
    last_xy  = (row.get('x4', 0.0), row.get('y4', 0.0))
    avg_speed= row.get('avg_speed', 30.0)
    ego_speed= row.get('ego_speed', 25.0)
    vehicle_type = row.get('vehicle_type', 'Sedan')

    template = """# Role:
You are an assistant who analyzes traffic conditions and predicts the trajectory of vehicles in a drone view environment.
# Instruction:
1. Predict the next trajectory.
2. Predict whether the vehicle will change lanes.
3. Provide an explanation of the lane change intention.
# Context:
The ego vehicle {tid} is currently located in lane {lane} and has driven from coordinates {fc} to {lc}.
The average vehicle speed in the area is {av:.2f}, and the speed of the ego vehicle is {eg:.2f}.
The type of vehicle is {vtype}.
"""
    return template.format(
        tid=track_id,
        lane=lane,
        fc=first_xy,
        lc=last_xy,
        av=avg_speed,
        eg=ego_speed,
        vtype=vehicle_type
    )
