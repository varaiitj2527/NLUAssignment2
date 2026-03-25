import os
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

def AnalyzeDataset(FilePath):
    """this function reads the text file calculates dataset statistics converts text to lower case and finds maximum length"""
    # open reads the file and returns a file object
    with open(FilePath, 'r', encoding='utf-8') as File:
        # strip removes leading and trailing whitespaces and lower converts string to lowercase
        Names = [Line.strip().lower() for Line in File.readlines() if Line.strip()]

    MaxLen = max(len(Name) for Name in Names)
    # join concatenates all strings in the list into a single string
    AllChars = "".join(Names)
    CharCounts = Counter(AllChars)
    # sorted returns a new sorted list from the items in iterable
    Vocab = sorted(list(CharCounts.keys()))
    VocabSize = len(Vocab)

    return Names, Vocab, VocabSize, MaxLen

class NameDataset(Dataset):
    """this class creates a pytorch dataset for the names handling character to integer mapping sequence padding and end of sequence tokens"""
    def __init__(self, Names, Vocab, MaxLen):
        self.MaxLen = MaxLen
        self.CharToIdx = {'<PAD>': 0, '<EOS>': 1}

        # enumerate adds a counter to an iterable and returns it
        for Idx, Char in enumerate(Vocab):
            self.CharToIdx[Char] = Idx + 2

        self.VocabSize = len(self.CharToIdx)

        self.Data = []
        for Name in Names:
            IdxSeq = [self.CharToIdx[Char] for Char in Name]
            IdxSeq.append(self.CharToIdx['<EOS>'])
            IdxSeq += [0] * (MaxLen - len(IdxSeq))

            # torch tensor constructs a tensor with the given data
            X = torch.tensor(IdxSeq[:-1], dtype=torch.long)
            Y = torch.tensor(IdxSeq[1:], dtype=torch.long)
            self.Data.append((X, Y))

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, Idx):
        return self.Data[Idx]

class VanillaRnn(nn.Module):
    """this class implements a simple recurrent neural network using standard linear transformations"""
    def __init__(self, VocabSize, EmbedSize, HiddenSize):
        super().__init__()
        self.HiddenSize = HiddenSize
        self.Embed = nn.Embedding(VocabSize, EmbedSize)

        # torch randn returns a tensor filled with random numbers from a normal distribution
        self.Wih = nn.Parameter(torch.randn(HiddenSize, EmbedSize) * 0.1)
        self.Whh = nn.Parameter(torch.randn(HiddenSize, HiddenSize) * 0.1)
        self.Who = nn.Parameter(torch.randn(VocabSize, HiddenSize) * 0.1)

    def forward(self, X, HZero=None):
        SeqLen, BatchSize = X.shape
        # torch zeros returns a tensor filled with the scalar value zero
        Ht = torch.zeros(self.HiddenSize, BatchSize).to(X.device) if HZero is None else HZero
        XEmbed = self.Embed(X)
        Outputs = []

        for T in range(SeqLen):
            Xt = XEmbed[T].T
            
            # computes linear transformation for hidden state
            # torch matmul performs matrix product of two tensors
            Zt = torch.matmul(self.Wih, Xt) + torch.matmul(self.Whh, Ht)
            
            # applies non linearity ht = tanh(wih * xt + whh * ht-1)
            # torch tanh computes the hyperbolic tangent element wise
            Ht = torch.tanh(Zt)
            
            # computes predicted output y_prime_t = who * ht
            YPrimeT = torch.matmul(self.Who, Ht)
            Outputs.append(YPrimeT.T)

        # torch stack concatenates a sequence of tensors along a new dimension
        return torch.stack(Outputs)

class BlstmCell(nn.Module):
    """this class implements a bidirectional lstm layer from scratch"""
    def __init__(self, InputSize, HiddenSize):
        super().__init__()
        self.HiddenSize = HiddenSize
        ConcatSize = HiddenSize + InputSize

        self.WfFwd = nn.Parameter(torch.randn(HiddenSize, ConcatSize) * 0.1)
        self.BfFwd = nn.Parameter(torch.zeros(HiddenSize, 1))
        self.WiFwd = nn.Parameter(torch.randn(HiddenSize, ConcatSize) * 0.1)
        self.BiFwd = nn.Parameter(torch.zeros(HiddenSize, 1))
        self.WcFwd = nn.Parameter(torch.randn(HiddenSize, ConcatSize) * 0.1)
        self.BcFwd = nn.Parameter(torch.zeros(HiddenSize, 1))
        self.WoFwd = nn.Parameter(torch.randn(HiddenSize, ConcatSize) * 0.1)
        self.BoFwd = nn.Parameter(torch.zeros(HiddenSize, 1))

        self.WfBwd = nn.Parameter(torch.randn(HiddenSize, ConcatSize) * 0.1)
        self.BfBwd = nn.Parameter(torch.zeros(HiddenSize, 1))
        self.WiBwd = nn.Parameter(torch.randn(HiddenSize, ConcatSize) * 0.1)
        self.BiBwd = nn.Parameter(torch.zeros(HiddenSize, 1))
        self.WcBwd = nn.Parameter(torch.randn(HiddenSize, ConcatSize) * 0.1)
        self.BcBwd = nn.Parameter(torch.zeros(HiddenSize, 1))
        self.WoBwd = nn.Parameter(torch.randn(HiddenSize, ConcatSize) * 0.1)
        self.BoBwd = nn.Parameter(torch.zeros(HiddenSize, 1))

    def forward(self, X):
        SeqLen, BatchSize, InputSize = X.shape

        HtFwd = torch.zeros(self.HiddenSize, BatchSize).to(X.device)
        CtFwd = torch.zeros(self.HiddenSize, BatchSize).to(X.device)
        ForwardStates = []

        for T in range(SeqLen):
            Xt = X[T].T
            
            # concatenates past hidden state and current input zt = [ht-1, xt]
            # torch cat concatenates the given sequence of tensors in the given dimension
            Zt = torch.cat([HtFwd, Xt], dim=0)

            # forget gate ft = sigmoid(wf * zt + bf)
            # torch sigmoid computes the logistic sigmoid function
            Ft = torch.sigmoid(torch.matmul(self.WfFwd, Zt) + self.BfFwd)
            
            # input gate it = sigmoid(wi * zt + bi)
            It = torch.sigmoid(torch.matmul(self.WiFwd, Zt) + self.BiFwd)
            
            # cell candidate ct_tilde = tanh(wc * zt + bc)
            CtTilde = torch.tanh(torch.matmul(self.WcFwd, Zt) + self.BcFwd)
            
            # cell state ct = ft * ct-1 + it * ct_tilde
            CtFwd = Ft * CtFwd + It * CtTilde
            
            # output gate ot = sigmoid(wo * zt + bo)
            Ot = torch.sigmoid(torch.matmul(self.WoFwd, Zt) + self.BoFwd)
            
            # hidden state ht = ot * tanh(ct)
            HtFwd = Ot * torch.tanh(CtFwd)

            ForwardStates.append(HtFwd.T)

        HtBwd = torch.zeros(self.HiddenSize, BatchSize).to(X.device)
        CtBwd = torch.zeros(self.HiddenSize, BatchSize).to(X.device)
        BackwardStates = []

        for T in range(SeqLen - 1, -1, -1):
            Xt = X[T].T
            # backwards concatenation zt = [ht+1, xt]
            Zt = torch.cat([HtBwd, Xt], dim=0)

            # backward equations identical to forward but reading right to left
            Ft = torch.sigmoid(torch.matmul(self.WfBwd, Zt) + self.BfBwd)
            It = torch.sigmoid(torch.matmul(self.WiBwd, Zt) + self.BiBwd)
            CtTilde = torch.tanh(torch.matmul(self.WcBwd, Zt) + self.BcBwd)
            CtBwd = Ft * CtBwd + It * CtTilde
            Ot = torch.sigmoid(torch.matmul(self.WoBwd, Zt) + self.BoBwd)
            HtBwd = Ot * torch.tanh(CtBwd)

            # insert places the element at the specified position
            BackwardStates.insert(0, HtBwd.T)

        ForwardTensor = torch.stack(ForwardStates)
        BackwardTensor = torch.stack(BackwardStates)
        
        # concatenates the forward and backward representations
        return torch.cat((ForwardTensor, BackwardTensor), dim=2)

class BlstmModel(nn.Module):
    """this class wraps the bidirectional lstm cell and applies the embedding and final linear layer"""
    def __init__(self, VocabSize, EmbedSize, HiddenSize):
        super().__init__()
        self.Embed = nn.Embedding(VocabSize, EmbedSize)
        self.Blstm = BlstmCell(EmbedSize, HiddenSize)
        self.Fc = nn.Linear(HiddenSize * 2, VocabSize)

    def forward(self, X):
        XEmbed = self.Embed(X)
        HiddenStates = self.Blstm(XEmbed)
        return self.Fc(HiddenStates)

class RnnCausalAttention(nn.Module):
    """this class implements a decoder only rnn with causal self attention where each step attends only to its own past and present hidden states"""
    def __init__(self, VocabSize, EmbedSize, HiddenSize):
        super().__init__()
        self.HiddenSize = HiddenSize
        self.Embed = nn.Embedding(VocabSize, EmbedSize)

        self.Wih = nn.Parameter(torch.randn(HiddenSize, EmbedSize) * 0.1)
        self.Whh = nn.Parameter(torch.randn(HiddenSize, HiddenSize) * 0.1)

        self.Wq = nn.Parameter(torch.randn(HiddenSize, HiddenSize) * 0.1)
        self.Wk = nn.Parameter(torch.randn(HiddenSize, HiddenSize) * 0.1)
        self.VAtt = nn.Parameter(torch.randn(HiddenSize, 1) * 0.1)

        self.Fc = nn.Linear(HiddenSize * 2, VocabSize)

    def forward(self, X):
        SeqLen, BatchSize = X.shape
        Ht = torch.zeros(self.HiddenSize, BatchSize).to(X.device)
        XEmbed = self.Embed(X)

        Outputs = []
        PastStates = []

        for T in range(SeqLen):
            Xt = XEmbed[T].T
            
            # generates current hidden state ht = tanh(wih * xt + whh * ht-1)
            Ht = torch.tanh(torch.matmul(self.Wih, Xt) + torch.matmul(self.Whh, Ht))
            PastStates.append(Ht)

            # projects current state into a query vector q = wq * ht
            Query = torch.matmul(self.Wq, Ht)
            EjtList = []

            for J in range(T + 1):
                Hj = PastStates[J]
                
                # projects past state into a key vector k = wk * hj
                Key = torch.matmul(self.Wk, Hj)
                
                # calculates alignment score ejt = vatt * tanh(query + key)
                Score = torch.matmul(self.VAtt.T, torch.tanh(Query + Key))
                EjtList.append(Score)

            Ejt = torch.cat(EjtList, dim=0)
            
            # normalizes scores to sum to one alphajt = softmax(ejt)
            # torch softmax scales output values to a probability distribution
            AlphaJt = torch.softmax(Ejt, dim=0)

            Ct = torch.zeros(self.HiddenSize, BatchSize).to(X.device)
            for J in range(T + 1):
                # computes weighted sum context vector ct = sum(alphajt * hj)
                # unsqueeze returns a new tensor with a dimension of size one inserted at the specified position
                Ct += AlphaJt[J].unsqueeze(0) * PastStates[J]

            # combines hidden state and context vector [ht, ct]
            Combined = torch.cat([Ht, Ct], dim=0)
            
            # final linear projection to vocab size
            Lt = self.Fc(Combined.T)
            Outputs.append(Lt)

        return torch.stack(Outputs)

def GenerateNames(Model, ModelName, Vocab, CharToIdx, NumNames, MaxLen, Device, Temperature=0.85):
    """this function generates text step by step stopping naturally when it predicts the end of sequence token"""
    Model.eval()
    GeneratedNames = []
    IdxToChar = {Idx: Char for Char, Idx in CharToIdx.items()}

    # torch no_grad disables gradient calculation to save memory and compute
    with torch.no_grad():
        for _ in range(NumNames):
            # start with a random character indices 2 to vocabsize
            StartIdx = random.randint(2, len(Vocab) + 1)
            CurrentSeq = [StartIdx]

            for _ in range(MaxLen - 1):
                Inputs = torch.tensor(CurrentSeq, dtype=torch.long).unsqueeze(1).to(Device)
                Outputs = Model(Inputs)

                # always take the prediction for the last time step
                LastOutput = Outputs[-1, 0, :]

                # applies temperature scaling before softmax to control randomness
                Probs = torch.softmax(LastOutput / Temperature, dim=0)
                
                # torch multinomial returns a random sample based on the probability distribution
                NextIdx = torch.multinomial(Probs, 1).item()

                if NextIdx == CharToIdx['<EOS>'] or NextIdx == CharToIdx['<PAD>']:
                    break

                CurrentSeq.append(NextIdx)

            Name = "".join([IdxToChar[Idx] for Idx in CurrentSeq])
            GeneratedNames.append(Name.capitalize())

    return GeneratedNames

def ComputeMetrics(GeneratedNames, TrainingNames):
    """this function computes the novelty rate and diversity of the generated text compared to the training data"""
    GeneratedSet = set(GeneratedNames)
    UniqueGenerated = len(GeneratedSet)
    TotalGenerated = len(GeneratedNames)

    TrainSet = set([Name.lower() for Name in TrainingNames])
    NovelCount = sum(1 for Name in GeneratedNames if Name.lower() not in TrainSet)

    NoveltyRate = (NovelCount / TotalGenerated) * 100 if TotalGenerated > 0 else 0
    Diversity = (UniqueGenerated / TotalGenerated) * 100 if TotalGenerated > 0 else 0

    return NoveltyRate, Diversity

def Main():
    """this function initializes parameters prepares the dataloader trains all models and evaluates the generated metrics"""
    FilePath = input("Enter the path to your corpus file: ")
    Names, Vocab, VocabSize, MaxLen = AnalyzeDataset(FilePath)

    MaxLen += 1

    DatasetInstance = NameDataset(Names, Vocab, MaxLen)
    CharToIdx = DatasetInstance.CharToIdx

    VocabSize = DatasetInstance.VocabSize
    EmbedSize = 32
    HiddenSize = 128
    BatchSize = 32
    Epochs = 50
    LearningRate = 0.001
    WeightDecay = 1e-5
    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Loader = DataLoader(DatasetInstance, batch_size=BatchSize, shuffle=True)

    Models = {
        "VanillaRnn": VanillaRnn(VocabSize, EmbedSize, HiddenSize).to(Device),
        "BlstmModel": BlstmModel(VocabSize, EmbedSize, HiddenSize).to(Device),
        "RnnCausalAttention": RnnCausalAttention(VocabSize, EmbedSize, HiddenSize).to(Device)
    }

    Criterion = nn.CrossEntropyLoss(ignore_index=0)

    for ModelName, Model in Models.items():
        print(f"\n=====================================")
        print(f"--- Training {ModelName} ---")
        
        # calculates total trainable parameters
        # numel returns the total number of elements in the tensor
        TotalParams = sum(Param.numel() for Param in Model.parameters() if Param.requires_grad)
        
        # calculates model size in megabytes using element_size which returns size in bytes
        ModelSizeMB = sum(Param.numel() * Param.element_size() for Param in Model.parameters()) / (1024 * 1024)
        
        print(f"Total Parameters: {TotalParams:,}")
        print(f"Model Size: {ModelSizeMB:.4f} MB")
        
        Optimizer = optim.Adam(Model.parameters(), lr=LearningRate, weight_decay=WeightDecay)

        Model.train()
        for Epoch in range(Epochs):
            TotalLoss = 0
            for Inputs, Targets in Loader:
                Inputs, Targets = Inputs.to(Device), Targets.to(Device)
                Inputs, Targets = Inputs.T, Targets.T

                Optimizer.zero_grad() # clears gradients

                Outputs = Model(Inputs)

                # view reshapes the tensor without copying memory
                Outputs = Outputs.view(-1, VocabSize)
                
                # contiguous returns a contiguous in memory tensor containing the same data
                Targets = Targets.contiguous().view(-1)

                Loss = Criterion(Outputs, Targets)
                Loss.backward() # computes derivative of loss

                torch.nn.utils.clip_grad_norm_(Model.parameters(), max_norm=5.0)
                Optimizer.step() # updates weights
                TotalLoss += Loss.item()

            if (Epoch + 1) % 5 == 0:
                print(f"Epoch [{Epoch+1}/{Epochs}], Loss: {TotalLoss/len(Loader):.4f}")

        print(f"\n--- Metrics for {ModelName} ---")
        GeneratedNames = GenerateNames(Model, ModelName, Vocab, CharToIdx, 100, MaxLen, Device)

        Novelty, Diversity = ComputeMetrics(GeneratedNames, Names)

        print(f"Novelty Rate: {Novelty:.2f}%")
        print(f"Diversity:    {Diversity:.2f}%")

        FileName = f"{ModelName}_generated.txt"
        with open(FileName, 'w', encoding='utf-8') as File:
            for Name in GeneratedNames:
                File.write(Name + '\n')

if __name__ == "__main__":
    Main()