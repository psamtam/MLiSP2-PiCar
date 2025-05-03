"""


ps aux | grep runserver

python3 run.py --model sam_tpu --mode drive --duration 60

python3 run.py --model sam_tpu --mode drive --duration 60 --max_speed 50



"""


import numpy as np
import tensorflow as tf
import os
import time


class Model:
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"  # enable gpu
    my_model = "/home/pi/autopilot/autopilot/models/sam_tpu/merged_model_120x160_14heads.tflite"

    def __init__(self):
        try:  # load edge TPU model
            # raise ValueError("Use CPU") # Use CPU
            delegate = tf.lite.experimental.load_delegate(
                "libedgetpu.so.1"
            )  #'libedgetpu.1.dylib' for mac or 'libedgetpu.so.1' for linux
            print("Using TPU")
            self.interpreter = tf.lite.Interpreter(
                model_path=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), self.my_model
                ),
                experimental_delegates=[delegate],
            )

        except ValueError:
            print("Fallback to CPU")
            self.interpreter = tf.lite.Interpreter(
                model_path=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), self.my_model
                )
            )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.output_details_sorted = sorted(
            self.output_details, key=lambda x: int(x["name"].split(":")[-1])
        )

    def preprocess(self, image):
        # im = tf.image.convert_image_dtype(image, tf.float32)
        im = tf.image.resize(image, (120, 160))
        im = tf.divide(im, 255)  # Normalize, need to check if needed
        im = tf.multiply(im, 2) - 1  # Normalise to -1 to 1
        im = tf.expand_dims(im, axis=0)  # add batch dimension
        return im

    def predict(self, image):
        image = self.preprocess(image)

        self.interpreter.set_tensor(
            self.input_details[0]["index"], image
        )  # original might work
        self.interpreter.invoke()

        all_outputs = [
            self.interpreter.get_tensor(out["index"]) for out in self.output_details_sorted
        ]

        # Split into angle_outputs (0-9) and speed_outputs (10-19)
        pred_angle = float(np.mean(np.squeeze(all_outputs[0:3])))
        pred_angle = np.where(abs(pred_angle-0.5)<0.01, pred_angle, np.clip((pred_angle-0.5)*1.3, -0.5, 0.5)+0.5)
        pred_angle = np.clip(pred_angle, 0, 1)

        pred_speed = int(np.mean(np.squeeze(all_outputs[-5:])[:,0]) > 0.5)

        pred_speed_s = np.squeeze(all_outputs[-5:])[:,0]

        for s in pred_speed_s:
            filled_length = int(s * 10)  # Calculate filled length based on speed
            progress_bar = '█' * filled_length + '-' * (10 - filled_length)
            print(f"|{progress_bar}| ", end="")

        print()

        progress_bar_length = 20



        arrow_left = int(np.mean(np.squeeze(all_outputs[3:6])[:, 1]) > 0.25)
        arrow_right = int(np.mean(np.squeeze(all_outputs[3:6])[:, 2]) > 0.25)
        arrow_left_raw = (np.mean(np.squeeze(all_outputs[3:6])[:, 1]))
        arrow_right_raw = np.mean(np.squeeze(all_outputs[3:6])[:, 2])

        arrow_turn_left = int(np.mean(np.squeeze(all_outputs[6:9])[:, 1]) > 0.25)
        arrow_turn_right = int(np.mean(np.squeeze(all_outputs[6:9])[:, 2]) > 0.25)
        arrow_turn_left_raw = (np.mean(np.squeeze(all_outputs[6:9])[:, 1]))
        arrow_turn_right_raw = (np.mean(np.squeeze(all_outputs[6:9])[:, 2]))

        if arrow_turn_right_raw > 0.3 and arrow_right_raw > 0.5:
            pred_angle = 1
        elif arrow_turn_left_raw > 0.3 and arrow_left_raw > 0.5:
            pred_angle = 0
        # elif arrow_turn_right_raw > 0.3 and arrow_right_raw > 0.3:
        #     pred_angle = 1

        angle = pred_angle * 80 + 50
        speed = np.around(pred_speed).astype(int) * 50

        # print(f"{arrow_left:.2f}, {arrow_right:.2f}, {arrow_turn_left:.2f}, {arrow_turn_right:.2f}")

        # Define the total length of the progress bar
        progress_bar_length = 20

        # Calculate the number of filled blocks based on the raw values
        arrow_left_progress = int(arrow_left_raw / 0.05)
        arrow_right_progress = int(arrow_right_raw / 0.05)
        arrow_turn_left_progress = int(arrow_turn_left_raw / 0.05)
        arrow_turn_right_progress = int(arrow_turn_right_raw / 0.05)

        # Create the progress bars
        left_bar = '█' * min(arrow_left_progress, progress_bar_length)
        right_bar = '█' * min(arrow_right_progress, progress_bar_length)
        turn_left_bar = '█' * min(arrow_turn_left_progress, progress_bar_length)
        turn_right_bar = '█' * min(arrow_turn_right_progress, progress_bar_length)

        # Print the progress bars
        print(f"{arrow_left_progress:2} [{left_bar:>{progress_bar_length}}] | "
            f"[{right_bar:<{progress_bar_length}}] {arrow_right_progress:2}")
        print(f"{arrow_turn_left_progress:2} [{turn_left_bar:>{progress_bar_length}}] | "
            f"[{turn_right_bar:<{progress_bar_length}}] {arrow_turn_right_progress:2}")        
        
        # print([f"{speed:.3f}" for speed in pred_speed_s])

        return angle, speed
