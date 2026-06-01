### Streamlit
python -m streamlit run dashboard/src/app.py

### Evaluation
python -m model.training.evaluate_detection --config model/configs/train_ssd_mobilenet.yaml --run-id 299b83b9-2b59-457b-875b-bfedd092d4e3 --split val --dataset model/data/rdd2022/complete

python -m model.training.evaluate_detection --config model/configs/train_yolo26.yaml --run-id 299b83b9-2b59-457b-875b-bfedd092d4e3 --split val --dataset model/data/rdd2022/complete

## Training

## mobilenet
python -m model.training.train_detection --config model/configs/train_ssd_mobilenet.yaml -v

## Yolo
python -m model.training.train_detection --config model/configs/train_yolo26.yaml -v


### Convert data
python -m model.scripts.convert_supervisely_to_yolo --src model/data/rdd2022/sample --dst model/data/rdd2022/sample_yolo

