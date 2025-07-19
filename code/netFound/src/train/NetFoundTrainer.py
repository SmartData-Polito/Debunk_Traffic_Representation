from transformers import Trainer


class NetfoundTrainer(Trainer):

    extraFields = {}

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        self._signature_columns += {
            "direction",
            "iats",
            "bytes",
            "pkt_count",
            "total_bursts",
            "ports",
            "stats",
            "protocol",
        }
        self._signature_columns += self.extraFields
        self._signature_columns = list(set(self._signature_columns))

    def __init__(self, label_names=None, extraFields = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if extraFields is not None:
            self.extraFields = extraFields
        if label_names is not None:
            self.label_names.extend(label_names)
