#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: Description

import numpy as np
import tensorflow as tf

class Classifier(object):

    def __init__(self, model_path, num_threads=1):
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, pre_processed_list, output_result_count):
        self.interpreter.set_tensor(self.input_details[0]['index'], np.array([pre_processed_list], dtype=np.float32))
        self.interpreter.invoke()

        result = self.interpreter.get_tensor(self.output_details[0]['index'])

        N_result_index, N_result_probs = self.maxNRes(np.squeeze(result), output_result_count)

        return N_result_index, N_result_probs

        # result_index = np.argmax(np.squeeze(result))

        # return result_index
    
    def maxNRes(self, result_probs, N):
    
        result_probs_sorted = np.argsort(result_probs)
        result_probs_descen_sorted = result_probs_sorted[::-1] #index

        maxNIndex = result_probs_descen_sorted[:N]
        maxNProbs = result_probs[maxNIndex]
        
        
        return maxNIndex, maxNProbs
