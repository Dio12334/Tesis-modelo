### Streamlit
python -m streamlit run dashboard/src/app.py

### Evaluation
python -m model.training.evaluate_detection --checkpoint checkpoints/ssd_mobilenetv3/b3206eba-5f2a-4367-8b08-09199d1c6e77/best_model.pt --split train

python -m model.training.evaluate_detection --checkpoint checkpoints/ssd_mobilenetv3/b3206eba-5f2a-4367-8b08-09199d1c6e77/best_model.pt --split val

## Training

## mobilenet
python -m model.training.evaluate_detection --config model/configs/train_ssd_mobilenet.yaml --run-id 299b83b9-2b59-457b-875b-bfedd092d4e3 --split val --dataset model/data/rdd2022/complete

## Yolo
python -m model.training.evaluate_detection --config model/configs/train_yolo26.yaml --run-id 299b83b9-2b59-457b-875b-bfedd092d4e3 --split val --dataset model/data/rdd2022/complete


### Convert data
python -m model.scripts.convert_supervisely_to_yolo --src model/data/rdd2022/sample --dst model/data/rdd2022/sample_yolo

