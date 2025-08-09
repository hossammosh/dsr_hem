from lib.train.admin.environment import env_settings

class Settings:
    """ Training settings, e.g. the paths to datasets and networks."""
    def __init__(self):
        self.set_default()

    def set_default(self):
        self.env = env_settings()
        self.use_gpu = True
        self.selected_sampling = False
        self.sample_per_epoch = 0
        # Phase manager will be set in train_script.py
        self.phase_manager = None
