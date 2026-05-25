### Streamlit
python -m streamlit run dashboard/app.py


### Evaluation
python -m model.training.evaluate_detection --checkpoint checkpoints/ssd_mobilenetv3/b3206eba-5f2a-4367-8b08-09199d1c6e77/best_model.pt --split train

python -m model.training.evaluate_detection --checkpoint checkpoints/ssd_mobilenetv3/b3206eba-5f2a-4367-8b08-09199d1c6e77/best_model.pt --split val

### Training
python -m model.training.train_detection --config model/configs/train_ssd_mobilenet.yaml