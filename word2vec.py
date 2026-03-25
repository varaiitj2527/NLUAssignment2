# !pip install gensim scikit-learn matplotlib numpy

import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
import matplotlib.patches as Mpatches
from collections import Counter
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

def LoadCorpus(FilePath="corpus.txt"):
    """loads tokenised sentences from a text file into a list of lists"""
    AllSentences = []
    # open is a built in function to interact with files
    with open(FilePath, "r", encoding="utf-8") as CorpusFile:
        for RawLine in CorpusFile:
            # strip removes leading trailing whitespace and split divides the string into a list based on spaces
            Tokens = RawLine.strip().split()
            if len(Tokens) > 1: AllSentences.append(Tokens)
    return AllSentences

def BuildVocabulary(AllSentences, MinCount=2):
    """creates vocabulary mappings and calculates word frequencies filtering out rare words"""
    # Counter tallies occurrences of every word across all sentences
    FreqMap = Counter(Word for Sent in AllSentences for Word in Sent)
    FilteredWords = [Word for Word, Count in FreqMap.items() if Count >= MinCount]
    # enumerate pairs each word with a unique integer index
    Word2Idx = {Word: Idx for Idx, Word in enumerate(FilteredWords)}
    Idx2Word = {Idx: Word for Word, Idx in Word2Idx.items()}
    return Word2Idx, Idx2Word, {W: FreqMap[W] for W in FilteredWords}

def BuildNegativeSamplingTable(FreqMap, Word2Idx, TableSize=1000000):
    """creates a lookup table based on unigram distribution raised to the power of three fourths for fast negative sampling"""
    # np zeros creates an integer array filled with zeros
    Table = np.zeros(TableSize, dtype=np.int32)
    SmoothedFreqs = np.array([FreqMap.get(Word, 0) ** 0.75 for Word in Word2Idx])
    # np sum calculates the total of all array elements
    WordProbs = SmoothedFreqs / SmoothedFreqs.sum()
    
    TableIdx = 0
    for WordIdx, Prob in enumerate(WordProbs):
        # round converts the float to the nearest integer
        for _ in range(int(round(Prob * TableSize))):
            if TableIdx < TableSize:
                Table[TableIdx] = WordIdx
                TableIdx += 1
    Table[TableIdx:] = WordIdx
    return Table

def SampleNegatives(NegTable, ExcludeIdx, NumNeg):
    """fetches random negative word indices from the table ensuring the target word is excluded"""
    Negatives = []
    TableSize = len(NegTable)
    while len(Negatives) < NumNeg:
        # random randint picks a random integer within the specified range
        NegIdx = int(NegTable[random.randint(0, TableSize - 1)])
        if NegIdx != ExcludeIdx: Negatives.append(NegIdx)
    return Negatives

def Sigmoid(X):
    """computes the logistic sigmoid function mathematically clipping values to prevent overflow warnings"""
    # np clip limits values in an array to a specified min and max
    X = np.clip(X, -500, 500)
    # np exp calculates the exponential of all elements in the input array
    return 1.0 / (1.0 + np.exp(-X))

#  Objective (from image 1):
#    J = - log σ(u_o^T v_c)  -  Σ_{k ∈ K sampled} log σ(-u_k^T v_c)
#
#  Gradients (derived analytically):
#    ∂J/∂v_c  = (σ(u_o^T v_c) - 1) u_o  +  Σ_k σ(u_k^T v_c) u_k
#    ∂J/∂u_o  = (σ(u_o^T v_c) - 1) v_c
#    ∂J/∂u_k  = σ(u_k^T v_c) v_c   for each negative k

class ScratchSkipgramNegSampling:
    """trains word embeddings using the skip gram objective with negative sampling from scratch utilizing analytical gradients"""
    def __init__(self, VocabSize, EmbedDim, NegTable, LearningRate=0.025):
        # np random randn returns samples from the standard normal distribution
        self.Vc = np.random.randn(VocabSize, EmbedDim) * 0.01
        self.Uo = np.random.randn(VocabSize, EmbedDim) * 0.01
        self.NegTable, self.LearningRate, self.LossHistory = NegTable, LearningRate, []

    def TrainOnePair(self, CentreIdx, ContextIdx, NumNeg):
        """processes a single centre context word pair computes loss and updates embeddings via gradient descent"""
        Vc, Uo = self.Vc[CentreIdx], self.Uo[ContextIdx]
        # np dot computes the dot product of two arrays
        ScorePos = Sigmoid(np.dot(Uo, Vc))
        
        NegIndices = SampleNegatives(self.NegTable, ContextIdx, NumNeg)
        NegVecs = self.Uo[NegIndices]
        ScoreNeg = Sigmoid(-(NegVecs @ Vc))
        
        # math log calculates natural logarithm
        Loss = -math.log(ScorePos + 1e-9) - np.sum(np.log(ScoreNeg + 1e-9))
        
        GradVc = (ScorePos - 1.0) * Uo + ((1.0 - ScoreNeg)[:, np.newaxis] * NegVecs).sum(axis=0)
        GradUo = (ScorePos - 1.0) * Vc
        GradUk = (1.0 - ScoreNeg)[:, np.newaxis] * Vc[np.newaxis, :]
        
        self.Vc[CentreIdx] -= self.LearningRate * GradVc
        self.Uo[ContextIdx] -= self.LearningRate * GradUo
        for Idx, NegIdx in enumerate(NegIndices):
            self.Uo[NegIdx] -= self.LearningRate * GradUk[Idx]
            
        return Loss

    def Train(self, AllSentences, Word2Idx, WindowSize, NumNeg, Epochs):
        """iterates over sentences to extract context windows and train the skip gram model recording loss per epoch"""
        for Epoch in range(Epochs):
            EpochLoss, PairCount = 0.0, 0
            # time time returns the current time in seconds since the epoch
            StartTime = time.time()
            for Sent in AllSentences:
                Indices = [Word2Idx[W] for W in Sent if W in Word2Idx]
                for CentrePos, CentreIdx in enumerate(Indices):
                    DynWindow = random.randint(1, WindowSize)
                    # max and min keep the window boundaries within the sentence length
                    Start, End = max(0, CentrePos - DynWindow), min(len(Indices), CentrePos + DynWindow + 1)
                    for ContextPos in range(Start, End):
                        if ContextPos != CentrePos:
                            EpochLoss += self.TrainOnePair(CentreIdx, Indices[ContextPos], NumNeg)
                            PairCount += 1
            self.LossHistory.append(EpochLoss / max(PairCount, 1))
            print(f"  [Skip-gram Scratch] Epoch {Epoch+1}/{Epochs} loss={self.LossHistory[-1]:.4f} time={time.time()-StartTime:.1f}s")

    def GetVector(self, WordIdx):
        """returns the final word representation by averaging the input and output embedding vectors"""
        return (self.Vc[WordIdx] + self.Uo[WordIdx]) / 2.0

    def MostSimilar(self, Word, Word2Idx, TopN=5):
        """finds the top nearest neighbours for a target word based on cosine similarity"""
        if Word not in Word2Idx: return []
        QueryVec = self.GetVector(Word2Idx[Word])
        # np linalg norm returns the matrix or vector norm representing vector magnitude
        QueryNorm = np.linalg.norm(QueryVec) + 1e-9
        Sims = [(W, float(np.dot(QueryVec, self.GetVector(Idx)) / (QueryNorm * (np.linalg.norm(self.GetVector(Idx)) + 1e-9)))) 
                for W, Idx in Word2Idx.items() if W != Word]
        # sorted sorts elements based on a key here reversing it for descending order
        return sorted(Sims, key=lambda X: X[1], reverse=True)[:TopN]

    def Analogy(self, WordA, WordB, WordC, Word2Idx, TopN=5):
        """performs word analogy calculation mathematically representing a to b as c to unknown"""
        if any(W not in Word2Idx for W in [WordA, WordB, WordC]): return []
        Query = self.GetVector(Word2Idx[WordB]) - self.GetVector(Word2Idx[WordA]) + self.GetVector(Word2Idx[WordC])
        QNorm = np.linalg.norm(Query) + 1e-9
        Sims = [(W, float(np.dot(Query, self.GetVector(Idx)) / (QNorm * (np.linalg.norm(self.GetVector(Idx)) + 1e-9)))) 
                for W, Idx in Word2Idx.items() if W not in {WordA, WordB, WordC}]
        return sorted(Sims, key=lambda X: X[1], reverse=True)[:TopN]

#  Objective (from image 2):
#    J = - log σ(u_o^T v̂) - Σ_{i=1}^{k} log σ(-u_n_i^T v̂)
#
#  where  v̂ = (1/2m) Σ_{j∈C} v_j   (mean of context vectors)
#
#  Gradients:
#    ∂J/∂v̂   = (σ(u_o^T v̂) - 1) u_o  +  Σ_i σ(u_n_i^T v̂) u_n_i
#    ∂J/∂v_j  = (1/2m) ∂J/∂v̂    for each context word j
#    ∂J/∂u_o  = (σ(u_o^T v̂) - 1) v̂
#    ∂J/∂u_n_i = σ(u_n_i^T v̂) v̂

class ScratchCbowNegSampling(ScratchSkipgramNegSampling):
    """trains word embeddings using the continuous bag of words objective predicting centre words from averaged contexts"""
    def TrainOnePair(self, ContextIndices, CentreIdx, NumNeg):
        """processes a cbow window averages context vectors computes analytical gradients and updates matrices"""
        if not ContextIndices: return 0.0
        # np mean computes the arithmetic mean along the specified axis
        VHat = self.Vc[ContextIndices].mean(axis=0)
        Uo = self.Uo[CentreIdx]
        ScorePos = Sigmoid(np.dot(Uo, VHat))
        
        NegIndices = SampleNegatives(self.NegTable, CentreIdx, NumNeg)
        NegVecs = self.Uo[NegIndices]
        ScoreNeg = Sigmoid(-(NegVecs @ VHat))
        
        Loss = -math.log(ScorePos + 1e-9) - np.sum(np.log(ScoreNeg + 1e-9))
        
        GradVHat = (ScorePos - 1.0) * Uo + ((1.0 - ScoreNeg)[:, np.newaxis] * NegVecs).sum(axis=0)
        GradVj = GradVHat / len(ContextIndices)
        GradUo = (ScorePos - 1.0) * VHat
        GradUk = (1.0 - ScoreNeg)[:, np.newaxis] * VHat[np.newaxis, :]
        
        for CtxIdx in ContextIndices: self.Vc[CtxIdx] -= self.LearningRate * GradVj
        self.Uo[CentreIdx] -= self.LearningRate * GradUo
        for Idx, NegIdx in enumerate(NegIndices): self.Uo[NegIdx] -= self.LearningRate * GradUk[Idx]
            
        return Loss

    def Train(self, AllSentences, Word2Idx, WindowSize, NumNeg, Epochs):
        """iterates over sentences to extract targets and surrounding contexts to train the cbow model"""
        for Epoch in range(Epochs):
            EpochLoss, PairCount = 0.0, 0
            StartTime = time.time()
            for Sent in AllSentences:
                Indices = [Word2Idx[W] for W in Sent if W in Word2Idx]
                for CentrePos, CentreIdx in enumerate(Indices):
                    DynWindow = random.randint(1, WindowSize)
                    Start, End = max(0, CentrePos - DynWindow), min(len(Indices), CentrePos + DynWindow + 1)
                    ContextIndices = [Indices[Pos] for Pos in range(Start, End) if Pos != CentrePos]
                    if ContextIndices:
                        EpochLoss += self.TrainOnePair(ContextIndices, CentreIdx, NumNeg)
                        PairCount += 1
            self.LossHistory.append(EpochLoss / max(PairCount, 1))
            print(f"  [CBOW Scratch] Epoch {Epoch+1}/{Epochs} loss={self.LossHistory[-1]:.4f} time={time.time()-StartTime:.1f}s")

def TrainGensimModels(AllSentences, VectorSize=100, WindowSize=5, NumNeg=5, Epochs=10):
    """trains gensim cbow and skip gram architectures to act as baseline references for the scratch implementations"""
    print("\n  Training Gensim Models...")
    GensimCbow = Word2Vec(sentences=AllSentences, vector_size=VectorSize, window=WindowSize, negative=NumNeg, min_count=2, sg=0, workers=4, epochs=Epochs)
    GensimSg = Word2Vec(sentences=AllSentences, vector_size=VectorSize, window=WindowSize, negative=NumNeg, min_count=2, sg=1, workers=4, epochs=Epochs)
    return GensimCbow, GensimSg

def TrainGensimCbow300(AllSentences):
    """trains a three hundred dimensional cbow model and prints the full embedding vector for the word artificial"""
    print("\n" + "═"*60 + "\nTRAINING 300-DIM GENSIM CBOW\n" + "═"*60)
    Model = Word2Vec(sentences=AllSentences, vector_size=300, window=5, negative=10, min_count=2, sg=0, workers=4, epochs=15)
    print("  Model trained successfully.")
    
    TargetWord = "artificial"
    if TargetWord in Model.wv.key_to_index:
        print(f"\n  [CBOW 300-Dim] Top 5 Neighbours for '{TargetWord}':")
        for W, S in Model.wv.most_similar(TargetWord, topn=5):
            print(f"    {W:<20} {S:.4f}")
            
        print(f"\n  [CBOW 300-Dim] Full Embedding Vector for '{TargetWord}':")
        # getting the exact numerical array representing the word in three hundred dimensional space
        Vector = Model.wv[TargetWord]
        # np array2string formats the numpy array nicely for console output ensuring it wraps properly
        print(np.array2string(Vector, separator=', '))
    else:
        print(f"    '{TargetWord}' not in vocabulary")
        
    return Model

ProbePairs = [("research", "publication"), ("student", "examination"), ("phd", "thesis"), ("btech", "course"), ("faculty", "professor"), ("admission", "programme")]
HyperConfigs = [{"Size": 50, "Win": 3, "Neg": 3}, {"Size": 100, "Win": 5, "Neg": 5}, {"Size": 200, "Win": 7, "Neg": 10}]

def ProbeScore(ModelObj, Pairs):
    """calculates average cosine similarity for predefined word pairs to evaluate semantic quality of embeddings"""
    Scores = [ModelObj.wv.similarity(A, B) for A, B in Pairs if A in ModelObj.wv.key_to_index and B in ModelObj.wv.key_to_index]
    return float(np.mean(Scores)) if Scores else 0.0

def RunHyperparameterExperiment(AllSentences):
    """tests multiple gensim configurations recording semantic coherence to find optimal hyperparameters"""
    print("\n" + "═"*60 + "\nHYPERPARAMETER EXPERIMENT\n" + "═"*60)
    CbowResults, SgResults = [], []
    for Cfg in HyperConfigs:
        Label = f"dim={Cfg['Size']} win={Cfg['Win']} neg={Cfg['Neg']}"
        CbowM = Word2Vec(sentences=AllSentences, vector_size=Cfg["Size"], window=Cfg["Win"], negative=Cfg["Neg"], min_count=2, sg=0, workers=4, epochs=10)
        SgM = Word2Vec(sentences=AllSentences, vector_size=Cfg["Size"], window=Cfg["Win"], negative=Cfg["Neg"], min_count=2, sg=1, workers=4, epochs=10)
        CbowResults.append((Label, ProbeScore(CbowM, ProbePairs)))
        SgResults.append((Label, ProbeScore(SgM, ProbePairs)))
        print(f"  {Label} -> CBOW: {CbowResults[-1][1]:.4f} | SG: {SgResults[-1][1]:.4f}")
    
    # plt subplots creates a figure with distinct plotting areas
    Fig, Ax = plt.subplots(figsize=(10, 5))
    # np arange creates evenly spaced positions
    X = np.arange(len(HyperConfigs))
    Ax.bar(X - 0.2, [S for _, S in CbowResults], 0.4, label='CBOW', color='#457b9d')
    Ax.bar(X + 0.2, [S for _, S in SgResults], 0.4, label='Skip-gram', color='#e63946')
    Ax.set_xticks(X)
    # ax set xticklabels applies text labels to the current x axis ticks
    Ax.set_xticklabels([L for L, _ in CbowResults])
    Ax.legend()
    plt.title("Hyperparameter Selection — CBOW vs Skip-gram")
    plt.savefig("hyperparams.png")
    plt.close()
    return CbowResults, SgResults

SemanticGroups = {"Academic": ["btech", "mtech", "phd", "ug", "pg", "course", "degree"], "Research": ["research", "publication", "project", "lab", "thesis"], "AI": ["artificial", "intelligence", "machine", "learning", "data"]}
GroupColours = {"Academic": "#e63946", "Research": "#2a9d8f", "AI": "#f4a261"}

def PlotTsne(ModelCbow, ModelSg, Word2Idx, LabelCbow, LabelSg, SaveName):
    """reduces embedding dimensionality using tsne and plots side by side colour coded semantic clusters"""
    AllWords = [W for Group in SemanticGroups.values() for W in Group]
    def GetVecs(Model):
        # hasattr checks if an object has the specified attribute mapping gensim vs scratch
        if hasattr(Model, "wv"): return [W for W in AllWords if W in Model.wv.key_to_index], [Model.wv[W] for W in AllWords if W in Model.wv.key_to_index]
        return [W for W in AllWords if W in Word2Idx], [Model.GetVector(Word2Idx[W]) for W in AllWords if W in Word2Idx]
        
    WordsCbow, VecsCbow = GetVecs(ModelCbow)
    WordsSg, VecsSg = GetVecs(ModelSg)
    if len(VecsCbow) < 5 or len(VecsSg) < 5: return
    
    # TSNE transforms high dimensional data into a 2d space for visual plotting
    Perp = min(15, len(VecsCbow) - 1, len(VecsSg) - 1)
    TsneObj = TSNE(n_components=2, perplexity=Perp, random_state=42)
    CoordsCbow, CoordsSg = TsneObj.fit_transform(np.array(VecsCbow)), TsneObj.fit_transform(np.array(VecsSg))
    
    Fig, (Ax1, Ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for Ax, Coords, Words, Title in [(Ax1, CoordsCbow, WordsCbow, LabelCbow), (Ax2, CoordsSg, WordsSg, LabelSg)]:
        for Idx, Word in enumerate(Words):
            Col = next((GroupColours[G] for G, Ws in SemanticGroups.items() if Word in Ws), "black")
            # ax scatter draws points on the graph based on x and y coordinate arrays
            Ax.scatter(Coords[Idx, 0], Coords[Idx, 1], c=Col, s=80, alpha=0.8)
            # ax annotate places text labels relative to the exact coordinate values
            Ax.annotate(Word, (Coords[Idx, 0], Coords[Idx, 1]), xytext=(5, 4), textcoords='offset points', fontsize=9)
        Ax.set_title(Title)
        
    Fig.legend(handles=[Mpatches.Patch(color=Col, label=Grp) for Grp, Col in GroupColours.items()], loc='lower center', ncol=3)
    # plt savefig exports the current figure to a local file
    plt.savefig(SaveName, bbox_inches='tight')
    plt.close()

def PrintTasks(Models, Word2Idx, TaskType):
    """executes and prints nearest neighbour and analogy tasks across multiple trained models"""
    print("\n" + "═"*60 + f"\n{TaskType} EXPERIMENTS\n" + "═"*60)
    # Added artificial and machine learning artificial analogy
    Words = ["research", "student", "artificial", "phd", "examination", "exam"] if TaskType == "NEIGHBOURS" else [("ug", "btech", "pg"), ("btech", "course", "phd"), ("machine", "learning", "artificial")]
    
    for Label, ModelObj in Models:
        print(f"\n  [{Label}]")
        for Item in Words:
            if TaskType == "NEIGHBOURS":
                Res = ModelObj.wv.most_similar(Item, topn=5) if hasattr(ModelObj, "wv") else ModelObj.MostSimilar(Item, Word2Idx, TopN=5)
                print(f"    '{Item}': {[W for W, S in Res]}")
            else:
                try: Res = ModelObj.wv.most_similar(positive=[Item[1], Item[2]], negative=[Item[0]], topn=5) if hasattr(ModelObj, "wv") else ModelObj.Analogy(Item[0], Item[1], Item[2], Word2Idx, TopN=5)
                except KeyError: Res = []
                print(f"    {Item[0]}:{Item[1]} :: {Item[2]}:? -> {[W for W, S in Res]}")

def Main():
    """entry point coordinating corpus loading training evaluating and visualising both scratch and library models"""
    random.seed(42)
    np.random.seed(42)
    FilePath = input("Enter the path to your corpus file: ")
    AllSentences = LoadCorpus(FilePath)
    Word2Idx, Idx2Word, FreqMap = BuildVocabulary(AllSentences)
    NegTable = BuildNegativeSamplingTable(FreqMap, Word2Idx)
    RunHyperparameterExperiment(AllSentences)
    
    print("\n" + "═"*60 + "\nTRAINING SCRATCH MODELS\n" + "═"*60)
    ScratchSg = ScratchSkipgramNegSampling(len(Word2Idx), 100, NegTable)
    ScratchSg.Train(AllSentences, Word2Idx, 5, 5, 5)
    ScratchCbow = ScratchCbowNegSampling(len(Word2Idx), 100, NegTable)
    ScratchCbow.Train(AllSentences, Word2Idx, 5, 5, 5)
    
    GensimCbow, GensimSg = TrainGensimModels(AllSentences)
    
    # Call new Gensim 300 Dim function
    ModelCbow300 = TrainGensimCbow300(AllSentences)
    
    ModelsList = [("CBOW Scratch", ScratchCbow), ("SG Scratch", ScratchSg), ("CBOW Gensim", GensimCbow), ("SG Gensim", GensimSg)]
    
    PrintTasks(ModelsList, Word2Idx, "NEIGHBOURS")
    PrintTasks(ModelsList, Word2Idx, "ANALOGY")
    
    PlotTsne(ScratchCbow, ScratchSg, Word2Idx, "CBOW (Scratch)", "Skip-gram (Scratch)", "tsne_scratch.png")
    PlotTsne(GensimCbow, GensimSg, Word2Idx, "CBOW (Gensim)", "Skip-gram (Gensim)", "tsne_gensim.png")

Main()