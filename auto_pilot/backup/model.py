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
    my_model = "/home/pi/autopilot/autopilot/models/sam_tpu/merged_model_120x160_2heads.tflite"

    def __init__(self):
        try:  # load edge TPU model
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
        pred_angle = float(np.squeeze(all_outputs[0]))
        pred_angle = np.where(abs(pred_angle-0.5)<0.01, pred_angle, np.clip((pred_angle-0.5)*1.3, -0.5, 0.5)+0.5)
        pred_angle = np.clip(pred_angle, 0, 1)
        pred_speed = int(np.squeeze(all_outputs[-1])[0] > 0.5)

        # arrow_left = float(np.squeeze(all_outputs[1]))
        # arrow_right = float(np.squeeze(all_outputs[2]))

        # arrow_left_turn = float(np.squeeze(all_outputs[3]))
        # arrow_right_turn = float(np.squeeze(all_outputs[4]))

        # if arrow_left_turn > 0.5 and arrow_left > 0.5:
        #     pred_angle = 0
        # elif arrow_right_turn > 0.5 and arrow_right > 0.5:
        #     pred_angle = 1

        # pred_angle_tuned = np.where(np.abs(pred_angle - 0.5) < 0.1, pred_angle, (pred_angle - 0.5) * 1.1 + 0.5)

        # pred_speed = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        # pred_angle = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        angle = pred_angle * 80 + 50
        speed = np.around(pred_speed).astype(int) * 50

        # print(f"{arrow_left:.2f}, {arrow_right:.2f}, {arrow_left_turn:.2f}, {arrow_right_turn:.2f}")

        return angle, speed
