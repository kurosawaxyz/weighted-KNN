class Hyperparam:
    """Configuration class for hyperparameters and settings"""
    def __init__(self, **kwargs):
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
        else: 
            self.hidden_dim = 64
            self.latent_dim = 32
            self.learning_rate = 1e-3
            self.epochs = 10
            self.batch_size = 32
            self.dropout_rate = 0.2
            self.weight_decay = 1e-4
            self.k_neighbors = 5
            self.temperature = 1.0
            self.capacity_reg = 0.1
            self.monotonicity_reg = 1.0