#!/usr/bin/env python

import threading
import time


class MyThread(threading.Thread):
    val = "-"



    def run(self):
        print(self.val)
        print("{} started!".format(self.getName()))  # "Thread-x started!"
        time.sleep(1)  # Pretend to work for a second
        print("{} finished!".format(self.getName()))  # "Thread-x finished!"


def main():
    for x in range(10):  # Four times...
        mythread = MyThread(name="Thread-{}".format(x + 1))  # ...Instantiate a thread and pass a unique ID to it
        mythread.val='as'
        mythread.start()  # ...Start the thread
        #time.sleep(.9)  # ...Wait 0.9 seconds before starting another


if __name__ == '__main__': main()
