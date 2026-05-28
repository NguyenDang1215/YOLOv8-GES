from ultralytics.nn.tasks import DetectionModel
import torch

# 1. Đường dẫn tới file yaml của bạn
cfg_path = 'ultralytics/cfg/models/v8/yolov8m-goldP2.yaml' 

# 2. Khởi tạo model từ cấu trúc YAML
# Lưu ý: Các class SimFusion, IFM... phải được import hoặc nằm trong 
# thư viện ultralytics/nn/modules thì model mới build được.
model = DetectionModel(cfg_path)

# 3. In bảng thống kê (Summary)
# Bảng này sẽ cho bạn biết chính xác Parameter và FLOPs của từng dòng YAML
model.info() 

# 4. In chi tiết cấu trúc các lớp
print(model)