# Titans_project/modules/data_prep.py

def build_scene_description(row, previous_row=None):
    track_id = int(row.get('track_id', 0))
    lane = row.get('lane', 'Unknown')
    first_xy = (row.get('x1', 0.0), row.get('y1', 0.0))
    last_xy  = (row.get('x4', 0.0), row.get('y4', 0.0))
    
    # avg_speed을 사전 정의된 값으로 설정하거나 계산
    avg_speed = 30.0  # 또는 다른 계산 로직
    
    # ego_speed 계산 (이전 프레임과 현재 프레임의 위치 차이를 기반으로)
    if previous_row is not None:
        prev_x = float(previous_row.get('center_x', 0.0))
        prev_y = float(previous_row.get('center_y', 0.0))
        curr_x = float(row.get('center_x', 0.0))
        curr_y = float(row.get('center_y', 0.0))
        # 단순 유클리드 거리 기반 속도 계산 (예: 한 프레임 당 거리)
        ego_speed = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2)**0.5
    else:
        ego_speed = 25.0  # 기본값
    
    # vehicle_type 매핑 (class_id를 기반으로)
    class_id = row.get('class_id', 1)
    vehicle_type_mapping = {1: 'Sedan', 2: 'SUV', 3: 'Truck'}  # 예시 매핑
    vehicle_type = vehicle_type_mapping.get(int(class_id), 'Unknown')
    
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
