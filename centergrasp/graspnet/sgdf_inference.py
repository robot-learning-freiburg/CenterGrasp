import torch
from centergrasp.configs import DEVICE
from centergrasp.sgdf.sgdf_inference import SGDFInference
from centergrasp.sgdf.data_structures import SGDFPrediction


class SGDFInferenceGT(SGDFInference):
    def get_embeddings_from_objidx(self, obj_idx: int) -> torch.Tensor:
        idx_th = torch.tensor(obj_idx).to(DEVICE)
        shape_embedding = self.lit_model.embeddings(idx_th)
        return shape_embedding.unsqueeze(0)

    def predict_from_objidx(self, obj_idx: int) -> SGDFPrediction:
        shape_embedding = self.get_embeddings_from_objidx(obj_idx)
        return self.predict_reconstruction(shape_embedding)
