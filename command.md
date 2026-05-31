### Streamlit
python -m streamlit run dashboard/app.py


### Evaluation
python -m model.training.evaluate_detection --checkpoint checkpoints/ssd_mobilenetv3/b3206eba-5f2a-4367-8b08-09199d1c6e77/best_model.pt --split train

python -m model.training.evaluate_detection --checkpoint checkpoints/ssd_mobilenetv3/b3206eba-5f2a-4367-8b08-09199d1c6e77/best_model.pt --split val

## Training

## mobilenet
python -m model.training.train_detection --config model/configs/train_ssd_mobilenet.yaml
## Yolo
wsl bash -c "cd /mnt/c/Users/Jean/Documents/GitHub/Tesis-modelo && conda activate tesis && yolo detect train model=yolo26m.pt data=model/data/rdd2022/sample_yolo/data.yaml epochs=10 batch=16 imgsz=640 project=checkpoints/yolo26 name=sample_run"



### Convert data
python -m model.scripts.convert_supervisely_to_yolo --src model/data/rdd2022/sample --dst model/data/rdd2022/sample_yolo

