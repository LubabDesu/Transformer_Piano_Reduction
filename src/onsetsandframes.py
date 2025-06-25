import torch.nn.functional as F
import torch

class OnsetsAndFrames :
    #Onset Frame Loss function 
    def onsetFrameLoss(self, batch, predictions):
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        velocity_label = batch['velocity']

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
            'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }

        return losses

    '''
    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return torch.tensor(0.0, device=velocity_pred.device)
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator
    '''
