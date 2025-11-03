import numpy as np
from .preprocessing import image_preprocessing
from .dsp import crop_two_imgs_with_displacement

class SyncOdometerHooks:
    """
    Manages the state and logic for the synchronous (blocking) mode.
    """
    def __init__(self, odometer_instance):
        self.odometer = odometer_instance
        self._setup_sync_mode()

    def _setup_sync_mode(self):
        # Binding the main methods to the synchronous implementations
        self.odometer.feed_image = self.feed_image_sync
        self.odometer.get_displacement = self.get_displacement_sync

    def feed_image_sync(self, img) -> None:
        """
        Send the next image in "Sequential" mode for processing (Synchronous).
        :param img: Next image in the stream
        """
        img_spectrum = image_preprocessing(img, self.odometer.configs)

        if self.odometer.imgs_processed[0] is None:
            # The first iteration
            self.odometer.imgs_processed[0] = img_spectrum
            self.odometer.imgs_original[0] = img
        else:
            # Update the current image:
            with self.odometer.imgs_lock:
                self.odometer.imgs_processed[1] = img_spectrum
                self.odometer.imgs_original[1] = img

    def get_displacement_sync(self):
        """
        Get the next displacement in "Sequential" mode (Synchronous).
        :return: Next x and y displacements in mm
        """
        try:
            reprocess_displacement = self.odometer.configs["Displacement Estimation"]["reprocess_displacement"]
            skip_frames = self.odometer.configs["Displacement Estimation"]["skip_frames"]

            if self.odometer.imgs_processed[0] is None or self.odometer.imgs_processed[1] is None:
                return 0.0, 0.0

            spectrum_beg = self.odometer.imgs_processed[0]
            original_img_beg = self.odometer.imgs_original[0]

            with self.odometer.imgs_lock:
                spectrum_end = self.odometer.imgs_processed[1].copy()
                original_img_end = self.odometer.imgs_original[1].copy()

            # Estimate gross displacement
            displacement = self.odometer._estimate_displacement(spectrum_beg, spectrum_end)

            if reprocess_displacement:
                count = self.odometer.configs["Displacement Estimation"]["params"].get("reprocess_displacement_count", 1)
                for _ in range(count):
                    round_dx = int(round(displacement[0]))
                    round_dy = int(round(displacement[1]))
                    crop_img_beg, crop_img_end = crop_two_imgs_with_displacement(
                        original_img_beg, original_img_end, round_dx, round_dy
                    )
                    # Note: estimate_displacement_between calls _estimate_displacement
                    new_displacement = self.odometer.estimate_displacement_between(crop_img_beg, crop_img_end)
                    displacement = [round_dx + new_displacement[0], round_dy + new_displacement[1]]

            if skip_frames:
                threshold = self.odometer.configs["Displacement Estimation"]["params"]["skip_frames_threshold"]
                if np.linalg.norm(displacement) < threshold:
                    # Do not update the base image (keep img_beg)
                    return 0.0, 0.0

            # Update base image only if displacement was accepted
            self.odometer.imgs_processed[0] = spectrum_end
            self.odometer.imgs_original[0] = original_img_end

            # Update accumulated position in the main object
            self.odometer.current_position[0] += displacement[0]
            self.odometer.current_position[1] += displacement[1]
            self.odometer.number_of_displacements += 1

            return displacement
        except NotImplementedError:
            return None, None