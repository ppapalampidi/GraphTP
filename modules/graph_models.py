from torch import nn
from modules.layers import SelfAttention, MLP
from torch.nn import functional as F

import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops

from torch_geometric.nn.inits import glorot, zeros


class ModelHelper:

    @staticmethod
    def script_context_interaction(script, context):
        product = context * script
        diff = context - script
        cos = F.cosine_similarity(script, context, dim=1)
        out = torch.cat([script, context, product, diff, cos.unsqueeze(1)], -1)

        return out


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, normalize=True, **kwargs):
        super(GCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None


    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                             edge_weight, self.improved,
                                             x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)



class GraphTP(nn.Module):
    def __init__(self, config):
        super(GraphTP, self).__init__()

        # --------------------------------------------------
        # Estimate RNN encoding sizes
        # --------------------------------------------------

        self.type = config["type"]

        scene_size = config["scene_encoder_size"]
        if config["scene_encoder_bidirectional"]:
            scene_size *= 2

        script_size = config["script_encoder_size"]
        if config["script_encoder_bidirectional"]:
            script_size *= 2

        # --------------------------------------------------
        # Script Encoder
        # --------------------------------------------------
        # Reads a sequence of scenes and produces
        # context-aware scene representations
        # --------------------------------------------------
        self.scene_encoder = nn.LSTM(
            input_size=config["input_size"],
            hidden_size=config["scene_encoder_size"],
            num_layers=config["scene_encoder_layers"],
            bidirectional=config["scene_encoder_bidirectional"],
            batch_first=True)

        self.scene_attention = SelfAttention(scene_size)

        self.script_encoder1 = nn.LSTM(
            input_size=scene_size,
            hidden_size=config["script_encoder_size"],
            num_layers=config["script_encoder_layers"],
            bidirectional=config["script_encoder_bidirectional"],
            batch_first=True)

        # --------------------------------------------------
        # Audiovisual encoding
        # --------------------------------------------------
        # Intra-scene audio and visual attention
        # --------------------------------------------------

        activation = nn.Tanh()
        dropout = 0.2

        self.audio_projection = nn.Sequential(
            nn.Linear(config["audio_size"], script_size),
            activation
        )

        self.vision_projection = nn.Sequential(
            nn.Linear(config["vision_size"], script_size),
            activation
        )

        modules = []

        modules.append(nn.Linear(script_size, 1))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))

        self.scene_attention_audio = nn.Sequential(*modules)

        self.scene_softmax_audio = nn.Softmax(dim=-1)

        modules = []

        modules.append(nn.Linear(script_size, 1))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))

        self.scene_attention_vision = nn.Sequential(*modules)

        self.scene_softmax_vision = nn.Softmax(dim=-1)

        # --------------------------------------------------
        # Graph construction
        # --------------------------------------------------
        # Selection of neighborhood size and computation of probabilities of
        # forming an edge between any two scenes in the graph
        # --------------------------------------------------

        self.neighbor_decision = nn.Linear(400, 6)

        self.core_scene = nn.Sequential(
            nn.Linear(scene_size, 1),
            activation
        )

        self.neighbor_scene = nn.Sequential(
            nn.Linear(scene_size, 1),
            activation
        )

        # --------------------------------------------------
        # Graph encoder
        # --------------------------------------------------
        # One layer GCN that takes as input the normalized edges
        #  in the graph and the contextualized scene representations
        # and computes a neighborhood (local) representation per scene
        # --------------------------------------------------

        self.graph1 = GCNConv(script_size, script_size, improved=True,
                              normalize=False)

        # --------------------------------------------------
        # Output layers
        # --------------------------------------------------
        # Linear output layer per TP
        # --------------------------------------------------

        interaction_size = script_size * 2
        self.TP1 = nn.Linear(interaction_size, 1)
        self.TP2 = nn.Linear(interaction_size, 1)
        self.TP3 = nn.Linear(interaction_size, 1)
        self.TP4 = nn.Linear(interaction_size, 1)
        self.TP5 = nn.Linear(interaction_size, 1)

        self.softmax = nn.Softmax(dim=-1)
        print(self)


    def forward(self, script, scene_lens, audio, vision, device):
        """

        :param script: sequence of scenes in the screenplay.
        Each scene contains a sequence of sentences/utterances
        :param scene_lens: true length (# of sentences) per scene
        :param audio: sequence of scenes: Each scene contains a sequence of
        representations for the corresponding audio segments
        :param vision: sequence of scenes: Each scene contains a sequence of
        representations for the corresponding visual frames
        :return: scene probabilities per TP, probability distribution per scene
        to be the neighbor of each other scene in the graph and the final
        sparse connections of the learnt graph
        """

        # contextualized textual scene representations

        script_outs, _ = self.scene_encoder(script)

        scene_embeddings, sa = self.scene_attention(script_outs, scene_lens)

        script_embeddings, _ = self.script_encoder1(scene_embeddings.unsqueeze(0))
        script_embeddings = script_embeddings.squeeze()

        # audiovisual projection and intra-scene attention

        audio = self.audio_projection(audio)
        vision = self.vision_projection(vision)

        audio_energies = self.scene_attention_audio(audio).squeeze()

        audio_energies = self.scene_softmax_audio(audio_energies)

        audio = (audio * audio_energies.unsqueeze(-1)).sum(1)

        vis_energies = self.scene_attention_vision(vision).squeeze()

        vis_energies = self.scene_softmax_vision(vis_energies)

        vision = (vision * vis_energies.unsqueeze(-1)).sum(1)

        ### bi-directional flow

        vision = F.normalize(vision, p=2, dim=-1)
        audio = F.normalize(audio, p=2, dim=-1)

        ## vision attended

        vision_audio = torch.matmul(vision, audio.t())

        vision_weights = vision_audio.softmax(1)

        vision_attended_audio = torch.matmul(vision_weights, audio)

        vision_attended = vision_attended_audio + vision

        ## audio attended

        audio_weights = vision_audio.softmax(0)

        audio_attended_vision = torch.matmul(audio_weights.transpose(-2, -1),
                                             vision)

        audio_attended = audio_attended_vision + audio

        audiovisual = torch.cat([audio_attended, vision_attended], -1)

        scenes_audiovisual_1 = audiovisual
        scenes_audiovisual_2 = audiovisual.t()

        ## multimodal similarity between all pairs of scenes

        multimodal_similarity = torch.matmul(scenes_audiovisual_1,
                                             scenes_audiovisual_2).squeeze()

        edge_index = []

        core_scenes = self.core_scene(scene_embeddings)
        neighbors = self.neighbor_scene(scene_embeddings)
        neigbors_similarity = core_scenes*neighbors.t()

        if self.type == 'multimodal':
            scores = neigbors_similarity * multimodal_similarity
        else:
            scores = neigbors_similarity

        scores -= torch.mean(scores, 0)

        ## Graph construction

        # 1. Select neighborhood size

        neighborhood_sizes = [1, 2, 3, 4, 5, 6]

        scores_for_nei = scores

        if device == torch.device("cuda"):
            scores_for_nei = torch.cat([scores_for_nei,
                                    torch.zeros(scores_for_nei.size(0),
                                                (400 - scores_for_nei.size(0))).cuda(
                                        scene_embeddings.get_device())], dim=-1)
        else:
            scores_for_nei = torch.cat([scores_for_nei,
                                    torch.zeros(scores_for_nei.size(0),
                                                (400 - scores_for_nei.size(0)))],
                                       dim=-1)

        nei_logits = self.neighbor_decision(scores_for_nei)

        tau = 0.1
        gumbels = -torch.empty_like(nei_logits.contiguous()).exponential_().log()
        nei_logits = (nei_logits + gumbels) / tau
        soft = nei_logits.softmax(-1)

        index_nei = soft.max(1, keepdim=True)[1]
        nei_hard = torch.zeros_like(nei_logits.contiguous()).scatter_(-1,
                                                                      index_nei,
                                                                      1.0)
        nei_indices = nei_hard - soft.detach() + soft

        reverse_nei = nei_indices.t()

        # 2. Find neighbors for all different neighborhood sizes

        rets = []

        y_softs = []
        y_hards = []
        for i in range(len(neighborhood_sizes)):
            tau = 0.5
            gumbels = -torch.empty_like(scores.contiguous()).exponential_().log()
            gumbels = (scores + gumbels) / tau
            y_soft = gumbels.softmax(-1)
            indices = torch.topk(y_soft, dim=-1, k=neighborhood_sizes[i])[1]
            value = 1 / neighborhood_sizes[i]
            y_hard = torch.zeros_like(scores.contiguous()).scatter_(-1,
                                                                    indices,
                                                                    value)
            ret = y_hard - y_soft.detach() + y_soft
            rets.append(ret)
            y_softs.append(y_soft)
            y_hards.append(y_hard)

        # 4. Use the neighborhoods per scene for the true selected size

        rets = torch.stack(rets, dim=0)
        reverse_nei = reverse_nei.unsqueeze(-1).repeat(1, 1, scores.size(0))
        rets = rets * reverse_nei
        ret = rets.mean(dim=0)

        y_hards = torch.stack(y_hards, dim=0)
        y_hards = y_hards * reverse_nei
        y_hard = y_hards.max(dim=0)[0]

        # 5. Construct sparse edges for the graph

        for i in range(ret.size(0)):
            for j in range(ret.size(1)):
                edge_index.append([i, j])

        if device == torch.device("cuda"):
            edge_index = torch.LongTensor(edge_index).cuda(scene_embeddings.
                                                           get_device())
        else:
            edge_index = torch.LongTensor(edge_index)

        edge_weight = torch.flatten(ret)

        # 6. Use the contextualized scene embeddings and the constructed graph
        # for the one-layer GCN

        script_embeddings_gcn = self.graph1(script_embeddings,
                                            edge_index.t().contiguous(),
                                            edge_weight)

        ## Finally, concatenate local and global scene representations

        script_embeddings = torch.cat((script_embeddings, script_embeddings_gcn),
                                      dim=-1)

        ## Binary classification per TP

        contextualized_script_embeddings = []

        for j in range(5):
            if j == 0:
                u = self.TP1(script_embeddings)
            elif j == 1:
                u = self.TP2(script_embeddings)
            elif j == 2:
                u = self.TP3(script_embeddings)
            elif j == 3:
                u = self.TP4(script_embeddings)
            else:
                u = self.TP5(script_embeddings)

            u = torch.transpose(u, 0, 1)
            u = self.softmax(u).squeeze()
            contextualized_script_embeddings.append(u)

        contextualized_script_embeddings = torch.stack(
            contextualized_script_embeddings, dim=0)

        y = contextualized_script_embeddings

        return y, scores, y_hard


class TAM(nn.Module,ModelHelper):
    def __init__(self, config, window_length):
        super(TAM, self).__init__()

        self.window_length = window_length

        scene_size = config["scene_encoder_size"]
        if config["scene_encoder_bidirectional"]:
            scene_size *= 2

        script_size = config["script_encoder_size"]
        if config["script_encoder_bidirectional"]:
            script_size *= 2
        # --------------------------------------------------
        # Scene Encoder
        # --------------------------------------------------
        # Reads a sequence of scene sentences and produces
        # context-aware scene sentence representations
        # --------------------------------------------------
        self.scene_encoder = nn.LSTM(
            input_size=config["input_size"],
            hidden_size=config["scene_encoder_size"],
            num_layers=config["scene_encoder_layers"],
            bidirectional=config["scene_encoder_bidirectional"],
            batch_first=True)
        # --------------------------------------------------
        # Scene Attention
        # --------------------------------------------------
        # Self-attention over the sentences contained in a scene
        # --------------------------------------------------
        self.scene_attention = SelfAttention(scene_size)
        # --------------------------------------------------
        # Script Encoder
        # --------------------------------------------------
        # Reads a sequence of scenes and produces
        # context-aware scene representations
        # --------------------------------------------------
        self.script_encoder = nn.LSTM(
            input_size=scene_size*3,
            hidden_size=config["script_encoder_size"],
            num_layers=config["script_encoder_layers"],
            bidirectional=config["script_encoder_bidirectional"],
            batch_first=True)
        # --------------------------------------------------
        # Audiovisual encoding
        # --------------------------------------------------
        # Intra-scene audio and visual attention
        # --------------------------------------------------
        activation = nn.Tanh()
        dropout = 0.2

        self.audio_projection = nn.Sequential(
            nn.Linear(config["audio_size"], script_size),
            activation
        )

        self.vision_projection = nn.Sequential(
            nn.Linear(config["vision_size"], script_size),
            activation
        )

        modules = []

        modules.append(nn.Linear(script_size, 1))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))

        self.scene_attention_audio = nn.Sequential(*modules)

        self.scene_softmax_audio = nn.Softmax(dim=-1)

        modules = []

        modules.append(nn.Linear(script_size, 1))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))

        self.scene_attention_vision = nn.Sequential(*modules)

        self.scene_softmax_vision = nn.Softmax(dim=-1)

        # --------------------------------------------------
        # TP-specific prediction layers
        # --------------------------------------------------
        interaction_size = (script_size * 4 + 1) * 2 + script_size
        self.TP1 = nn.Linear(interaction_size, 1)
        self.TP2 = nn.Linear(interaction_size, 1)
        self.TP3 = nn.Linear(interaction_size, 1)
        self.TP4 = nn.Linear(interaction_size, 1)
        self.TP5 = nn.Linear(interaction_size, 1)

        self.softmax = nn.Softmax(dim=-1)

        print(self)

    def forward(self, script, scene_lens, audio, vision, device):

        # ----------------------------------------------------------
        # 1 - encode the scenes
        # ----------------------------------------------------------
        script_outs, _ = self.scene_encoder(script)

        scene_embeddings, sa = self.scene_attention(script_outs, scene_lens)

        # audiovisual projection and intra-scene attention

        audio = self.audio_projection(audio)
        vision = self.vision_projection(vision)

        audio_energies = self.scene_attention_audio(audio).squeeze()

        audio_energies = self.scene_softmax_audio(audio_energies)

        audio = (audio * audio_energies.unsqueeze(-1)).sum(1)

        vis_energies = self.scene_attention_vision(vision).squeeze()

        vis_energies = self.scene_softmax_vision(vis_energies)

        vision = (vision * vis_energies.unsqueeze(-1)).sum(1)

        scene_embeddings = torch.cat([scene_embeddings, audio, vision], -1)
        # ----------------------------------------------------------
        # 2 - contextualize the scene representations
        # ----------------------------------------------------------
        script_outs1, _ = self.script_encoder(scene_embeddings.unsqueeze(0))
        script_embeddings = script_outs1.squeeze()

        num_scenes = script_embeddings.size(0)

        left_context = []
        right_context = []
        for i in range(num_scenes):

            if i == 0:
                if device == torch.device("cuda"):
                    _left_context = torch.zeros((script_embeddings.size(1))). \
                    cuda(script_embeddings.get_device())
                else:
                    _left_context = torch.zeros((script_embeddings.size(1)))
            elif i < int(self.window_length * num_scenes):
                _left_context = torch.mean(
                    script_embeddings.narrow(0, 0, i), dim=0)
            else:
                _left_context = torch.mean(script_embeddings.
                                           narrow(0,
                                                  i - int(self.window_length * num_scenes),
                                                  int(self.window_length * num_scenes)),
                                           dim=0)

            if i == (num_scenes - 1):
                if device == torch.device("cuda"):
                    _right_context = torch.zeros((script_embeddings.size(1))). \
                    cuda(script_embeddings.get_device())
                else:
                    _right_context = torch.zeros((script_embeddings.size(1)))
            elif i > (
                    num_scenes - 1 - int(self.window_length * num_scenes)):
                _right_context = torch.mean(script_embeddings.
                                            narrow(0, i, (num_scenes - i)),
                                            dim=0)
            else:
                _right_context = torch.mean(script_embeddings.narrow(0, i,
                                        int(self.window_length * num_scenes)),
                                            dim=0)

            left_context.append(_left_context)
            right_context.append(_right_context)

        left_context = torch.stack(left_context, dim=0)
        right_context = torch.stack(right_context, dim=0)

        left_interaction = self.script_context_interaction(
            script_embeddings, left_context)
        right_interaction = self.script_context_interaction(
            script_embeddings, right_context)

        u = torch.cat(
            [script_embeddings, left_interaction, right_interaction], -1)
        # ----------------------------------------------------------
        # 3 - probability of each scene to represent each TP
        # ----------------------------------------------------------
        y = []
        # number of output posterior distributions = number of TPs = 5
        for i in range(5):
            if i == 0:
                y_now = self.TP1(u)
            elif i == 1:
                y_now = self.TP2(u)
            elif i == 2:
                y_now = self.TP3(u)
            elif i == 3:
                y_now = self.TP4(u)
            else:
                y_now = self.TP5(u)
            y_now = torch.transpose(y_now, 0, 1)
            y_now = self.softmax(y_now).squeeze()
            y.append(y_now.squeeze())

        y = torch.stack(y, dim=0)

        return y

