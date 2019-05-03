from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np


def make_weights(indim, outdim): 
    """
    Helper function that creates weight matrix

    Args:
        indim: input dimension
        outdim: output dimension
    Returns:
        torch tensor of zero mean variance scaled floats of shape [indim, outdim]
        that is differentiable
    """
    return torch.normal(
                    torch.zeros((indim, outdim)),
                    1/np.sqrt(indim)
                    ).clone().detach().requires_grad_(True)



######################################################################################################################
class imageCaptionModel(nn.Module):
    def __init__(self, config):
        super(imageCaptionModel, self).__init__()
        """
        "imageCaptionModel" is the main module class for the image captioning network
        
        Args:
            config: Dictionary holding neural network configuration

        Returns:
            self.Embedding  : An instance of nn.Embedding, shape[vocabulary_size, embedding_size]
            self.inputLayer : An instance of nn.Linear, shape[VggFc7Size, hidden_state_sizes]
            self.rnn        : An instance of RNN
            self.outputLayer: An instance of nn.Linear, shape[hidden_state_sizes, vocabulary_size]
        """
        self.config = config
        self.vocabulary_size    = config['vocabulary_size']
        self.embedding_size     = config['embedding_size']
        self.VggFc7Size         = config['VggFc7Size']
        self.hidden_state_sizes = config['hidden_state_sizes']
        self.num_rnn_layers     = config['num_rnn_layers']
        self.cell_type          = config['cellType']

        self.Embedding = torch.nn.Embedding(self.vocabulary_size, self.embedding_size)
        self.inputLayer = torch.nn.Linear(self.VggFc7Size, self.hidden_state_sizes)
        self.rnn = RNN(self.embedding_size, self.hidden_state_sizes, self.num_rnn_layers, self.cell_type)
        self.outputLayer = torch.nn.Linear(self.hidden_state_sizes, self.vocabulary_size)

        return

    def forward(self, vgg_fc7_features, xTokens, is_train, current_hidden_state=None):
        """
        Args:
            vgg_fc7_features    : Features from the VGG16 network, shape[batch_size, VggFc7Size]
            xTokens             : Shape[batch_size, truncated_backprop_length]
            is_train            : "is_train" is a flag used to select whether or not to use estimated token as input
            current_hidden_state: If not None, "current_hidden_state" should be passed into the rnn module
                                  shape[num_rnn_layers, batch_size, hidden_state_sizes]

        Returns:
            logits              : Shape[batch_size, truncated_backprop_length, vocabulary_size]
            current_hidden_state: shape[num_rnn_layers, batch_size, hidden_state_sizes]
        """
        # ToDO
        # Get "initial_hidden_state" shape[num_rnn_layers, batch_size, hidden_state_sizes].
        # Remember that each rnn cell needs its own initial state.

        # use self.rnn to calculate "logits" and "current_hidden_state"

        batch_size = vgg_fc7_features.shape[0]
        states_prep = self.inputLayer(vgg_fc7_features)

        if current_hidden_state is None:
            initial_hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_sizes))

            for i in range(self.num_rnn_layers):
                initial_hidden_state[i] = states_prep.clone().detach()
        else:
            initial_hidden_state = current_hidden_state

        logits, current_state = self.rnn.forward(
                                        xTokens,
                                        initial_hidden_state,
                                        self.outputLayer,
                                        self.Embedding,
                                        is_train
                                        )
        return logits, current_state
######################################################################################################################
class RNN(nn.Module):
    def __init__(self, input_size, hidden_state_size, num_rnn_layers, cell_type='RNN'):
        super(RNN, self).__init__()
        """
        Args:
            input_size (Int)        : embedding_size
            hidden_state_size (Int) : Number of features in the rnn cells (will be equal for all rnn layers) 
            num_rnn_layers (Int)    : Number of stacked rnns
            cell_type               : Whether to use vanilla or GRU cells
            
        Returns:
            self.cells              : A nn.ModuleList with entities of "RNNCell" or "GRUCell"
        """
        self.input_size        = input_size
        self.hidden_state_size = hidden_state_size
        self.num_rnn_layers    = num_rnn_layers
        self.cell_type         = cell_type
        
        cell_classes = {
                "RNN": RNNCell,
                "GRU": GRUCell,
                }

        self.cell_class = cell_classes[self.cell_type]
        self.cells = []

        for i in range(num_rnn_layers):
            if i == 0:
                self.cells.append(self.cell_class(hidden_state_size, input_size))
            else:
                self.cells.append(self.cell_class(hidden_state_size, hidden_state_size))

        self.cells = torch.nn.ModuleList(self.cells)

        return 


    def forward(self, xTokens, initial_hidden_state, outputLayer, Embedding, is_train=True):
        """
        Args:
            xTokens:        shape [batch_size, truncated_backprop_length]
            initial_hidden_state:  shape [num_rnn_layers, batch_size, hidden_state_size]
            outputLayer:    handle to the last fully connected layer (an instance of nn.Linear)
            Embedding:      An instance of nn.Embedding. This is the embedding matrix.
            is_train:       flag: whether or not to feed in the predicated token vector as input for next step

        Returns:
            logits        : The predicted logits. shape[batch_size, truncated_backprop_length, vocabulary_size]
            current_state : The hidden state from the last iteration (in time/words).
                            Shape[num_rnn_layers, batch_size, hidden_state_sizes]
        """
        if is_train==True:
            seqLen = xTokens.shape[1] #truncated_backprop_length
        else:
            seqLen = 40 #Max sequence length to be generated

        if is_train:
            return self._train_forward(xTokens, initial_hidden_state, outputLayer, Embedding, seqLen)
        else:
            return self._predict_forward(xTokens, initial_hidden_state, outputLayer, Embedding, seqLen)

    def _train_forward(self, xTokens, initial_hidden_state, outputLayer, Embedding, seqLen):
        batch_size = xTokens.shape[0]
        device = xTokens.device

        #dim: batch_size, seqLen, input_size
        embedded_tokens = Embedding(xTokens)

        #dim: batch_size, seqLen, vocab_size
        logits = torch.zeros((xTokens.shape[0], xTokens.shape[1], outputLayer.out_features)).to(device)

        #dim: seqLen, rnn_layers, batch_size, hidden_state_size
        states = torch.zeros(
                    (seqLen+1, self.num_rnn_layers, batch_size, self.hidden_state_size),
                    ).to(device)

        states[0] = initial_hidden_state

        for i in range(seqLen):
            cell_input = embedded_tokens[:, i, :]

            for j in range(self.num_rnn_layers):
                state = states[i, j].clone().detach()
                cell = self.cells[j]
                states[i+1, j] = cell(cell_input, state)
                cell_input = states[i+1, j].clone().detach()

            logits[:, i, :] = outputLayer(states[i+1, -1]).clone().detach().requires_grad_(True).to(device)

        return logits, states[-1]

    def _predict_forward(self, xTokens, initial_hidden_state, outputLayer, Embedding, seqLen):

        batch_size = xTokens.shape[0]
        #dim: batch_size, seqLen, vocab_size
        logits = torch.zeros((xTokens.shape[0], seqLen, outputLayer.out_features))
        #dim: rnn_layers, batch_size, hidden_state_size
        cell_input = Embedding(xTokens)[:, 0, :]

        states = torch.zeros(
                (seqLen+1, self.num_rnn_layers, batch_size, self.hidden_state_size),
                    )

        states[0] = initial_hidden_state

        for i in range(seqLen):
            for j in range(self.num_rnn_layers):
                state = states[i, j].clone().detach()
                cell = self.cells[j]
                states[i, j] = cell(cell_input, state)
                cell_input = states[i, j].clone().detach()

            logits[:, i, :] = outputLayer(states[i+1, -1])
            output = torch.nn.Softmax(dim=1)(logits[:, i, :])
            words = torch.argmax(output, dim=1)
            cell_input = Embedding(words) 

        return logits, states[-1]

########################################################################################################################
class GRUCell(nn.Module):
    def __init__(self, hidden_state_size, input_size):
        super(GRUCell, self).__init__()
        """
        Args:
            hidden_state_size: Integer defining the size of the hidden state of rnn cell
            inputSize: Integer defining the number of input features to the rnn

        Returns:
            self.weight_u: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                           variance scaling with zero mean. 

            self.weight_r: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                           variance scaling with zero mean. 

            self.weight: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                         variance scaling with zero mean. 

            self.bias_u: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero.

            self.bias_r: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero. 

            self.bias: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero. 

        Tips:
            Variance scaling:  Var[W] = 1/n
        """

        self.hidden_state_sizes = hidden_state_size
        self.rnn_input_size = hidden_state_size + input_size

        self.weight_u = torch.nn.Parameter(make_weights(self.rnn_input_size, hidden_state_size))
        self.bias_u   = torch.nn.Parameter(torch.zeros((1, hidden_state_size), requires_grad=True)) 

        self.weight_r = torch.nn.Parameter(make_weights(self.rnn_input_size, hidden_state_size))
        self.bias_r   = torch.nn.Parameter(torch.zeros((1, hidden_state_size), requires_grad=True))

        self.weight = torch.nn.Parameter(make_weights(self.rnn_input_size, hidden_state_size))
        self.bias   = torch.nn.Parameter(torch.zeros((1, hidden_state_size), requires_grad=True)) 

        return

    def forward(self, x, state_old):
        """
        Args:
            x: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]

        """
        device = x.device

        state_old = state_old.to(device)
        tmp = torch.cat((x, state_old), 1, ).to(device)

        gamma_u = torch.sigmoid(torch.matmul(tmp, self.weight_u) + self.bias_u)
        gamma_r = torch.sigmoid(torch.matmul(tmp, self.weight_r) + self.bias_r)
        
        h_tmp = torch.cat((x, gamma_r * state_old), 1)
        h_tilde = torch.tanh(torch.matmul(h_tmp, self.weight) + self.bias)

        state_new = gamma_u * state_old + ( 1 - gamma_u) * h_tilde
        return state_new

######################################################################################################################
class RNNCell(nn.Module):
    def __init__(self, hidden_state_size, input_size):
        super(RNNCell, self).__init__()
        """
        Args:
            hidden_state_size: Integer defining the size of the hidden state of rnn cell
            inputSize: Integer defining the number of input features to the rnn

        Returns:
            self.weight: A nn.Parameter with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                         variance scaling with zero mean.

            self.bias: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero. 

        Tips:
            Variance scaling:  Var[W] = 1/n
        """
        self.hidden_state_size = hidden_state_size

        self.weight = torch.nn.Parameter(make_weights(hidden_state_size + input_size, hidden_state_size))
        self.bias   = torch.nn.Parameter(torch.zeros((1, hidden_state_size, ) , requires_grad=True))

        return


    def forward(self, x, state_old):
        """
        Args:
            x: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]

        """

        device = x.device
        state_old = state_old.to(device)
        tmp = torch.cat((x, state_old), 1).to(device)
        state_new = torch.tanh(torch.matmul(tmp, self.weight) + self.bias)
        return state_new

######################################################################################################################
def loss_fn(logits, yTokens, yWeights):
    """
    Weighted softmax cross entropy loss.

    Args:
        logits          : shape[batch_size, truncated_backprop_length, vocabulary_size]
        yTokens (labels): Shape[batch_size, truncated_backprop_length]
        yWeights        : Shape[batch_size, truncated_backprop_length]. Add contribution to the total loss only from words exsisting 
                          (the sequence lengths may not add up to #*truncated_backprop_length)

    Returns:
        sumLoss: The total cross entropy loss for all words
        meanLoss: The averaged cross entropy loss for all words

    Tips:
        F.cross_entropy
    """

    softmax = F.log_softmax(logits, dim=2)
    softmax = torch.transpose(softmax, 1, 2)

    loss = torch.nn.NLLLoss(reduction="none")(softmax, yTokens)
    weighted_loss = loss * yWeights

    sumLoss = torch.sum(weighted_loss) 
    meanLoss = sumLoss/torch.sum(yWeights)

    return sumLoss, meanLoss


