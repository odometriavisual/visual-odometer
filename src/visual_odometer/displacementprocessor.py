import threading


class DisplacementProcessor(threading.Thread):
    def __init__(self, owner):
        super(DisplacementProcessor, self).__init__()
        self.event = threading.Event()
        self.terminated = False
        self.owner = owner
        self.start()
        self.img_beg, self.img_end = None, None

    def feed_image(self, img_beg, img_end) -> None:
        self.img_beg, self.img_end = img_beg, img_end

    def start_processing(self) -> None:
        self.event.set()

    def run(self):
        while not self.terminated:
            if self.event.wait(
                    1):  # Wait until the VisualOdometer wakes this thread to start processing a new pair of frames.
                try:
                    displacement = self.owner._estimate_displacement(self.img_beg, self.img_end)
                    self.owner.displacements.append(displacement)

                finally:
                    # Reset the event
                    self.event.clear()

                    # Clears the thread image buffer:
                    self.img_beg, self.img_end = None, None

                    # Return ourselves to the available pool
                    with self.owner.lock:
                        self.owner.pool.append(self)
