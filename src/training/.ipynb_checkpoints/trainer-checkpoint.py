import torch
import torch.nn as nn
import json
import os
import statistics
from src.training.online_stability_tracker import OnlineStabilityTracker
from src.training.progressive_freezer import ProgressiveFreezer


class FusionAwareTrainer:
    """
    QAT trainer with progressive static scale conversion.
    Tracks activation stability during training and progressively
    freezes stable layers, letting remaining weights adapt.
    """

    def __init__(self,
                 model,
                 train_batches,
                 val_batches,
                 device,
                 lr=2e-5,
                 cv_threshold=5.0,
                 patience=50,
                 freeze_tier_size=5,
                 adaptation_steps=100):

        self.model         = model.to(device)
        self.train_batches = train_batches
        self.val_batches   = val_batches
        self.device        = device

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        # stability tracking
        self.tracker  = OnlineStabilityTracker(
            cv_threshold=cv_threshold,
            patience=patience
        )
        self.tracker.attach_hooks(model)

        self.freezer = ProgressiveFreezer(
            model=model,
            tracker=self.tracker,
            freeze_tier_size=freeze_tier_size,
            adaptation_steps=adaptation_steps
        )

        # logging
        self.log = {
            'train_loss': [], 'val_acc': [],
            'frozen_count': [], 'step': []
        }
        self.global_step = 0
        self.best_acc    = 0.0

    def train(self, epochs=5):
        os.makedirs('results/checkpoints', exist_ok=True)
        os.makedirs('results/figures',     exist_ok=True)

        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []

            for batch in self.train_batches:
                inputs = {k: v.to(self.device) for k, v in batch.items()
                          if k != 'labels'}
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()
                out  = self.model(**inputs)
                loss = self.criterion(out.logits, labels)
                loss.backward()
                self.optimizer.step()

                # update stability tracker
                newly_eligible = self.tracker.step()

                # inside the batch loop, after tracker.step()
                if self.global_step % 20 == 0:
                    candidates = self.tracker.get_freeze_candidates()
                    stable_counts = {
                        n: self.tracker.stable_steps[n] 
                        for n in list(self.tracker.stable_steps.keys())[:3]
                    }
                    print(f"  step={self.global_step} "
                          f"candidates={len(candidates)} "
                          f"sample_stable_steps={stable_counts}")
                    
                # maybe freeze a tier of layers
                self.freezer.maybe_freeze(self.global_step)

                epoch_losses.append(loss.item())
                self.global_step += 1

                # log every 50 steps
                if self.global_step % 50 == 0:
                    frozen = sum(self.tracker.frozen.values())
                    self.log['train_loss'].append(
                        statistics.mean(epoch_losses[-50:]))
                    self.log['frozen_count'].append(frozen)
                    self.log['step'].append(self.global_step)

            # end of epoch — validate
            val_acc = self._validate()
            self.log['val_acc'].append(val_acc)

            frozen_count = sum(self.tracker.frozen.values())
            print(f"Epoch {epoch+1}/{epochs}  "
                  f"loss={statistics.mean(epoch_losses):.4f}  "
                  f"val_acc={val_acc:.2f}%  "
                  f"frozen={frozen_count}/38")

            # save best
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(
                    self.model.state_dict(),
                    'results/checkpoints/phase3_best.pt'
                )

        # save training log
        with open('results/profiles/phase3_training_log.json', 'w') as f:
            json.dump(self.log, f, indent=2)

        # save freeze schedule
        with open('results/profiles/phase3_freeze_schedule.json', 'w') as f:
            json.dump(self.freezer.get_freeze_schedule(), f, indent=2)

        print(f"\nTraining complete. Best val acc: {self.best_acc:.2f}%")
        print(f"Final frozen layers: "
              f"{sum(self.tracker.frozen.values())}/38")
        self.tracker.remove_hooks()
        return self.model

    def _validate(self):
        self.model.eval()
        correct = 0
        total   = 0
        with torch.no_grad():
            for batch in self.val_batches:
                inputs = {k: v.to(self.device) for k, v in batch.items()
                          if k != 'labels'}
                labels = batch['labels'].to(self.device)
                out    = self.model(**inputs)
                correct += (out.logits.argmax(-1) == labels).sum().item()
                total   += labels.size(0)
        self.model.train()
        return correct / total * 100