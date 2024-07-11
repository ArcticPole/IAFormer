# comprehension perceptron
import torch
import torch.nn as nn
import torch.nn.functional as F


class CP(nn.Module):
    """
    Interaction Prior Learning Module
    """
    def __init__(self, opt):
        super(CP, self).__init__()
        self.codebook_EMA = EMA(opt=opt, num_embeddings=opt.codebook_size, embedding_dim=opt.latent_dim, commitment_cost=0.25, decay=0.99)

    def forward(self, IP_score):
        k_loss, feature_EMA, _, _, sec_EMA = self.codebook_EMA(IP_score)

        cp_score = 1-torch.norm(feature_EMA-IP_score)/torch.norm(sec_EMA-IP_score)

        return cp_score, k_loss, feature_EMA


class Codebook(nn.Module):
    def __init__(self, opt):
        super(Codebook, self).__init__()
        self.codebook_size = opt.codebook_size  # Number of codewords
        self.latent_dim = opt.latent_dim  # Size of codewords
        self.beta = opt.beta
        self.embedding = nn.Embedding(self.codebook_size, self.latent_dim)  # Codebook used to store prior information
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def update_codebook(self, inputs):

        inputs = inputs.view(-1)
        print(inputs, inputs.shape)

        indices = self.embedding(inputs).argmin(dim=1)
        print(indices, indices.shape)

        counts = torch.bincount(indices, minlength=self.codebook_size)
        print(counts, counts.shape)
        sum_vectors = torch.bincount(indices, weights=inputs, minlength=self.codebook_size)
        print(sum_vectors, sum_vectors.shape)
        new_codebook = sum_vectors / counts.unsqueeze(1)
        print(self.embedding.weight.data.shape, new_codebook.shape)

        self.embedding.weight.data.copy_(new_codebook)

    def forward(self, z):

        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * (torch.matmul(z_flattened, self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_encoding_indices, loss


class EMA(nn.Module):
    def __init__(self, opt, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(EMA, self).__init__()

        self.opt = opt
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))  # weight
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        bs, kpts, seq = inputs.shape
        inputs = inputs.reshape([bs, kpts // 3, 3, seq]).contiguous()
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input,
                                        self._embedding.weight.t()))  # compute distance between input and embedding vector

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # index of nearst encoding
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)  # set 1 at a zero matrix with same size as ema


        # confidence
        sort_coding, coding_idx = torch.sort(distances, dim=1, descending=False)
        second_idx = coding_idx[:, 2].unsqueeze(1)
        second = torch.zeros(second_idx.shape[0], self._num_embeddings, device=inputs.device)
        second.scatter_(1, second_idx, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        second_quantized = torch.matmul(second, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        second_quantized = inputs + (second_quantized - inputs).detach()

        # convert quantized from BHWC -> BCHW
        quantized = quantized.reshape([bs, kpts, seq])
        second_quantized = second_quantized.reshape([bs, kpts, seq])

        return loss, quantized.contiguous(), perplexity, encodings, second_quantized.contiguous()
