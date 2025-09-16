import torch.nn.functional as F
import torch

class OnsetsAndFrames :
    #Onset Frame Loss function 
    def onsetFrameLoss(self, batch, predictions):
        onset_label_orig = batch['onset']
        offset_label_orig = batch['offset']
        frame_label_orig = batch['frame']

        # Permute labels to match prediction shape -> [B, Time, Pitches]
        onset_label = onset_label_orig.permute(0, 2, 1).float()
        offset_label = offset_label_orig.permute(0, 2, 1).float()
        frame_label = frame_label_orig.permute(0, 2, 1).float()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        pw_frame  = torch.tensor([min(32.0, 50.0)], device=device)
        pw_onset  = torch.tensor([max(262.2, 50.0)], device=device)   # capped to 50
        pw_offset = torch.tensor([max(266.8, 50.0)], device=device)   # capped to 50
        
        crit_frame  = torch.nn.BCEWithLogitsLoss(pos_weight=pw_frame)
        crit_onset  = torch.nn.BCEWithLogitsLoss(pos_weight=pw_onset)
        crit_offset = torch.nn.BCEWithLogitsLoss(pos_weight=pw_offset)


        # losses = {
        #     'loss/onset': F.binary_cross_entropy_with_logits(predictions['onset'], onset_label),
        #     'loss/offset': F.binary_cross_entropy_with_logits(predictions['offset'], offset_label),
        #     'loss/frame': F.binary_cross_entropy_with_logits(predictions['frame'], frame_label),
        #     # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        # }

        λF, λO, λOff = 1.0, 3.0, 1.0  # relative task weights

        losses = (λF*crit_frame(predictions['frame'], frame_label) +
                λO*crit_onset(predictions['onset'], onset_label) +
                λOff*crit_offset(predictions['offset'], offset_label)) / (λF+λO+λOff)

        return losses

    
    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return torch.tensor(0.0, device=velocity_pred.device)
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator
    
