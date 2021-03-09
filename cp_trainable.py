import os
from ray.tune import Trainable
from ray.util.sgd import TorchTrainer
from ray.tune.trial import Resources


class ClausePredictionTrainable(Trainable):
    # @classmethod
    # def default_resource_request(cls, config):
    #     return Resources(
    #         cpu=0,
    #         gpu=0,
    #         memory=2**34,
    #         extra_cpu=config["num_replicas"]+3,             # data loader
    #         extra_gpu=int(config["use_gpu"]) * config["num_replicas"])

    def _setup(self, config):
        self._trainer = TorchTrainer(**config)

    def _train(self):
        train_stats = self._trainer.train()
        validation_stats = self._trainer.validate()

        train_stats.update(validation_stats)

        # output {"mean_loss": test_loss, "mean_accuracy": accuracy}
        return train_stats

    def _save(self, checkpoint_dir):
        return self._trainer.save(os.path.join(checkpoint_dir, "model.pth"))

    def _restore(self, checkpoint_path):
        return self._trainer.restore(checkpoint_path)

    def _stop(self):
        self._trainer.shutdown()
