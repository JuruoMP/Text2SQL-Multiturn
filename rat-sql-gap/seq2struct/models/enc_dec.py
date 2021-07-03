import torch
import torch.utils.data

from seq2struct.models import abstract_preproc
from seq2struct.utils import registry

class ZippedDataset(torch.utils.data.Dataset):
    def __init__(self, *components):
        assert len(components) >= 1
        lengths = [len(c) for c in components]
        assert all(
            lengths[0] == other for other in lengths[1:]), "Lengths don't match: {}".format(lengths)
        self.components = components
    
    def __getitem__(self, idx):
        return tuple(c[idx] for c in self.components)
    
    def __len__(self):
        return len(self.components[0])


@registry.register('model', 'EncDec')
class EncDecModel(torch.nn.Module):
    class Preproc(abstract_preproc.AbstractPreproc):
        def __init__(
                self,
                encoder,
                decoder,
                encoder_preproc,
                decoder_preproc):
            super().__init__()

            self.enc_preproc = registry.lookup('encoder', encoder['name']).Preproc(**encoder_preproc)
            self.dec_preproc = registry.lookup('decoder', decoder['name']).Preproc(**decoder_preproc)
        
        def validate_item(self, item, section):
            enc_result, enc_info = self.enc_preproc.validate_item(item, section)
            dec_result, dec_info = self.dec_preproc.validate_item(item, section)
            
            return enc_result and dec_result, (enc_info, dec_info)
        
        def add_item(self, item, section, validation_info):
            enc_info, dec_info = validation_info
            self.enc_preproc.add_item(item, section, enc_info)
            self.dec_preproc.add_item(item, section, dec_info)
        
        def clear_items(self):
            self.enc_preproc.clear_items()
            self.dec_preproc.clear_items()

        def save(self):
            self.enc_preproc.save()
            self.dec_preproc.save()
        
        def load(self):
            self.enc_preproc.load()
            self.dec_preproc.load()
        
        def dataset(self, section):
            return ZippedDataset(self.enc_preproc.dataset(section), self.dec_preproc.dataset(section))
        
    def __init__(self, preproc, device, encoder, decoder):
        super().__init__()
        self.preproc = preproc
        self.encoder = registry.construct(
                'encoder', encoder, device=device, preproc=preproc.enc_preproc)
        self.decoder = registry.construct(
                'decoder', decoder, device=device, preproc=preproc.dec_preproc)
        self.decoder.visualize_flag = False
        
        if getattr(self.encoder, 'batched'):
            self.compute_loss = self._compute_loss_enc_batched
        else:
            self.compute_loss = self._compute_loss_unbatched

        self.align_weight = 100

    def _compute_loss_enc_batched(self, batch, debug=False):
        losses = []
        d = [enc_input for enc_input, dec_output in batch]
        enc_states = self.encoder(d)

        align_losses = []
        final_d_id = [i for i, (enc_input, dec_output) in enumerate(batch) if 'final_nl' in enc_input is not None]
        final_d = [enc_input['final_nl'] for enc_input, dec_output in batch if 'final_nl' in enc_input]
        final_enc_states = [None for _ in range(len(d))]
        if len(final_d_id) > 0:
            for i, state in zip(final_d_id, self.encoder(final_d)):
                final_enc_states[i] = state
        for enc, final_enc in zip(enc_states, final_enc_states):
            if final_enc is not None:
                enc_cls, final_enc_cls = enc.cls_vec, final_enc.cls_vec
                align_losses.append(torch.nn.functional.mse_loss(enc_cls, final_enc_cls))
        align_loss = torch.stack(align_losses).mean() if len(align_losses) > 0 else 0

        for enc_state, (enc_input, dec_output) in zip(enc_states, batch):
            loss = self.decoder.compute_loss(enc_input, dec_output, enc_state, debug)
            losses.append(loss)
        if debug:
            return losses
        else:
            return torch.mean(torch.stack(losses, dim=0), dim=0) + self.align_weight * align_loss

    def _compute_loss_enc_batched2(self, batch, debug=False):
        losses = []
        for enc_input, dec_output in batch:
            enc_state, = self.encoder([enc_input])
            loss = self.decoder.compute_loss(enc_input, dec_output, enc_state, debug)
            losses.append(loss)
        if debug:
            return losses
        else:
            return torch.mean(torch.stack(losses, dim=0), dim=0)

    def _compute_loss_unbatched(self, batch, debug=False):
        losses = []
        for enc_input, dec_output in batch:
            enc_state = self.encoder(enc_input)
            loss = self.decoder.compute_loss(enc_input, dec_output, enc_state, debug)
            losses.append(loss)
        if debug:
            return losses
        else:
            return torch.mean(torch.stack(losses, dim=0), dim=0)

    def eval_on_batch(self, batch):
        mean_loss = self.compute_loss(batch).item()
        batch_size = len(batch)
        result = {'loss': mean_loss * batch_size, 'total': batch_size}
        return result

    def begin_inference(self, orig_item, preproc_item):
        ## TODO: Don't hardcode train
        #valid, validation_info =  self.preproc.enc_preproc.validate_item(item, 'train')
        #if not valid:
        #    return None
        #enc_input = self.preproc.enc_preproc.preprocess_item(item, validation_info)

        enc_input, _ = preproc_item
        if self.decoder.visualize_flag:
            print('question:')
            print(enc_input['question'])
            print('columns:')
            print(enc_input['columns'])
            print('tables:')
            print(enc_input['tables'])
        if getattr(self.encoder, 'batched'):
            enc_state, = self.encoder([enc_input])
        else:
            enc_state = self.encoder(enc_input)
        return self.decoder.begin_inference(enc_state, orig_item)
