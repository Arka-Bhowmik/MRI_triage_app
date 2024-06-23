#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:45:53 2024

@author: unknown
"""
# This code allow streamlite progress bar to take function
# and display function progress
#
import logging, time
logging.disable(logging.WARNING)
import streamlit as st

class Progress:
    """Progress bar and Status update."""
    def __init__(self, number_of_functions: int):
        self.n = number_of_functions
        self.bar = st.progress(0)
        self.progress = 1
        self.message = ""
        self.message_container = st.empty()

    def go(self, msg, function, *args, **kwargs):
        self.message += msg
        self.message_container.info(self.message)
        s = time.time()
        result = function(*args, **kwargs)
        spent_time = (time.time() - s)/60
        self.message += f" {round(spent_time, 2)} min. "
        # self.message += f" [{time.time() - s:.2f}min]. "
        self.message_container.info(self.message)
        self.bar.progress(self.progress / self.n)
        self.progress += 1
        # if self.progress > self.n:
        #     self.bar.empty()
        return result
#