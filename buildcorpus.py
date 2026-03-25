"""installs required dependencies"""
# !pip install beautifulsoup4 requests nltk wordcloud gensim scikit-learn numpy
import requests
from bs4 import BeautifulSoup
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

"""list of target web addresses to scrape information from"""
urls = [
"https://iitj.ac.in/office-of-academics/en/academic-regulations",
"https://www.iitj.ac.in/m/Index/main-programs?lg=en",
"https://www.iitj.ac.in/office-of-academics/en/academics",
"https://www.iitj.ac.in/main/en/recruitments",
"https://www.iitj.ac.in/main/en/faculty-members",
"https://www.iitj.ac.in/m/Index/main-departments?lg=en",
"https://www.iitj.ac.in/main/en/research-highlight",
"https://home.iitj.ac.in/~palashdas/",
"https://anandmishra22.github.io/",
"https://home.iitj.ac.in/~vimalraj/",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/admissions",
"https://www.iitj.ac.in/",
"https://www.iitj.ac.in/Main/en/Help",
"https://www.iitj.ac.in/Search?lg=en",
"https://www.iitj.ac.in/AtoZ?lg=en",
"https://www.iitj.ac.in/winter-school/en/Winter-School",
"https://www.iitj.ac.in/grad-cohort-workshop/en/Grad-Cohort-Workshop",
"https://www.iitj.ac.in/admission-postgraduate-programs/en/list-of-provisionally-selected-candidates",
"https://www.iitj.ac.in/admission-postgraduate-programs/en/list-of-shortlisted-candidates",
"https://www.iitj.ac.in/admission-postgraduate-programs/en/Rolling-advertisement-for-Admission-to-Interdisciplinary-and-Ph.D.-Programmes?ep=fw",
"https://www.iitj.ac.in/main/en/important-links",
"https://www.iitj.ac.in/correspondence",
"https://www.iitj.ac.in/rti",
"https://www.iitj.ac.in/office-of-stores-purchase/en/tender-details",
"https://www.iitj.ac.in/techscape/en/Techscape",
"https://www.iitj.ac.in/main/en/contact",
"https://www.iitj.ac.in/main/en/how-to-reach-iit-jodhpur",
"https://www.iitj.ac.in/Institute-Repository/en/Institute-Repository",
"https://www.iitj.ac.in/office-of-corporate-relations/en/Donate",
"https://www.iitj.ac.in/main/en/web-policy",
"https://www.iitj.ac.in/main/en/web-information-manager",
"https://www.iitj.ac.in/feedback",
"https://www.iitj.ac.in/institute-repository/en/nirf",
"https://www.iitj.ac.in/anti-sexual-harassment-policy/en/anti-sexual-harassment-policy",
"https://www.iitj.ac.in/main/en/intranet-page",
"https://www.iitj.ac.in/dia/en/dia",
"https://iitj.ac.in/",
"https://iitj.ac.in/Main/en/Help",
"https://iitj.ac.in/Search?lg=en",
"https://iitj.ac.in/AtoZ?lg=en",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/school-of-artificial-intelligence-and-data-science",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/mission-and-vision",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/ai-and-ds-ecosystem",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/outreach",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/administration",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/campus",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/faqs",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/phd",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/ms-program",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/mtech",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/btech",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/executive-programs",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/courses",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/about-research",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/themes",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/projects",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/publications",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/facilities-labs",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/About-CoEs",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/School-of-Artificial-Intelligence-and-Data-Science",
"https://ayurtech.iitj.ac.in/",
"https://iitj.ac.in/People?dept=School-of-Artificial-Intelligence-Data-Science",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/students",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/alumni",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/directory",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/project-positions",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/postdoc-positions",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/internship-irograms",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/faculty-positions",
"https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/contact"
]

def ExtractText(Url):
    """extracts raw text from a webpage and removes navigational html tags"""
    try:
        # using requests get to fetch the webpage content
        Response = requests.get(Url, timeout=10)
        # using beautifulsoup to parse the fetched html content
        Soup = BeautifulSoup(Response.text, "html.parser")

        # iterating over specific html tags and decomposing them to clean the tree
        for Tag in Soup(["script", "style", "nav", "footer", "header"]):
            Tag.decompose()

        # using get text to extract all string content separated by a space
        Text = Soup.get_text(separator=" ")
        return Text

    except Exception as Error:
        print(f"Error fetching {Url}: {Error}")
        return ""

"""list of unwanted phrases to filter out"""
JunkPhrases = [
    "return to index", "back to top", "click here",
    "home", "main menu", "skip to content",
    "read more", "view more", "apply now"
]

def RemoveJunkPhrases(Text):
    """replaces common navigational junk phrases with spaces"""
    for Phrase in JunkPhrases:
        # using string replace to remove exact phrase matches
        Text = Text.replace(Phrase, " ")
    return Text

def NormalizeDegrees(Text):
    """standardizes variations of degree names like btech and mtech to a uniform format"""
    # using lower to make the text case insensitive for easier matching
    Text = Text.lower()

    # using re sub to find degree patterns and replace them uniformly
    Text = re.sub(r'\b(b[\.\s]*tech)\b', 'btech', Text)
    Text = re.sub(r'\b(m[\.\s]*tech)\b', 'mtech', Text)
    Text = re.sub(r'\b(ph[\.\s]*d)\b', 'phd', Text)

    return Text

def CleanTextForW2v(Text):
    """cleans the text by normalizing degrees removing junk and filtering special characters"""
    Text = NormalizeDegrees(Text)
    Text = RemoveJunkPhrases(Text)

    # using re sub to keep only alphanumeric characters hyphens periods exclamation marks question marks and spaces
    Text = re.sub(r'[^a-z0-9\-\.\!\?\s]', ' ', Text)

    # using re sub to replace multiple consecutive spaces with a single space and strip removes leading trailing spaces
    Text = re.sub(r'\s+', ' ', Text).strip()

    return Text

def TokenizeForW2v(Sentence):
    """extracts words from a sentence and filters out single letter characters"""
    # using re findall to extract all whole words containing alphanumeric characters and hyphens
    Tokens = re.findall(r'\b[a-z0-9\-]+\b', Sentence)
    # using list comprehension to filter out words shorter than two characters
    Tokens = [Word for Word in Tokens if len(Word) >= 2]
    return Tokens

"""main execution block that processes each url extracts text cleans it tokenizes it and saves to a corpus file"""
Sentences = []

# using open to create or overwrite a text file safely with a context manager
with open("corpus.txt", "w", encoding="utf-8") as CorpusFile:
    print("\n=========== SENTENCE LEVEL TOKENS (PER WEBSITE) ===========\n")

    for Index, Url in enumerate(urls):
        print(f"\n--- Website {Index+1} ---")
        
        RawText = ExtractText(Url)
        CleanText = CleanTextForW2v(RawText)

        # using sent tokenize from nltk to split the paragraph into sentences
        Sents = sent_tokenize(CleanText)

        if len(Sents) < 20:
            # using re split as a fallback to split by periods or newlines if nltk fails
            Sents = re.split(r'[.\n]', CleanText)

        WebsiteSentences = []

        for Sent in Sents:
            Tokens = TokenizeForW2v(Sent)

            if len(Tokens) > 2:
                Sentences.append(Tokens)
                WebsiteSentences.append(Tokens)
                # using join to combine tokens into a space separated string and write it to the file
                CorpusFile.write(" ".join(Tokens) + "\n")
        
        print("\nSample tokenized sentences:")
        for s in WebsiteSentences[:5]:
            print(s)

"""filters out rare words from the sentences to improve dataset quality while keeping important academic terms"""
# using counter to create a frequency map of all words across all sentences
WordCounts = Counter([Word for Sent in Sentences for Word in Sent])
ImportantWords = {"ug", "pg", "phd", "btech", "mtech"}

# using list comprehension to keep only words that appear at least twice or are in the important words set
Sentences = [
    [Word for Word in Sent if WordCounts[Word] >= 2 or Word in ImportantWords]
    for Sent in Sentences
]
Sentences = [Sent for Sent in Sentences if len(Sent) > 2]

"""calculates and prints overall statistics of the clean corpus"""
TotalSentences = len(Sentences)
TotalTokens = sum(len(Sent) for Sent in Sentences)
# using set to find unique words
Vocab = set(Word for Sent in Sentences for Word in Sent)

print("\n=========== FINAL DATASET STATS ===========")
print(f"Total Sentences: {TotalSentences}")
print(f"Total Tokens: {TotalTokens}")
print(f"Vocabulary Size: {len(Vocab)}")

"""generates and displays a word cloud from the most frequent words"""
AllTokens = [Word for Sent in Sentences for Word in Sent]
WordFreq = Counter(AllTokens)

CustomWcStopwords = {
    "the", "and", "for", "with", "from", "this", "that",
    "are", "was", "were", "has", "have", "had", "of", "in", "to"
}

FilteredFreq = {
    Word: Count for Word, Count in WordFreq.items()
    if len(Word) > 2 and Word not in CustomWcStopwords
}

# using wordcloud class to generate an image based on word frequencies
WordCloudObj = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(FilteredFreq)

# using matplotlib to plot and display the image without axes
plt.figure(figsize=(10,5))
plt.imshow(WordCloudObj, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud - Most Frequent Words")
plt.show()

"""prints the top ten most frequent words along with their occurrences"""
print("\n=========== TOP 10 MOST FREQUENT WORDS ===========")
# using most common method from the counter object to get the top ten items
TopTenWords = Counter(FilteredFreq).most_common(10)
for Word, Freq in TopTenWords:
    print(f"{Word}: {Freq}")