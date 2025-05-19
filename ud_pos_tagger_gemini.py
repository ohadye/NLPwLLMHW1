# --- Imports ---
import os, json, copy
from google import genai
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Annotated

with open("test_hard_sentences.json", "r") as f:
    test_hard_sentences = json.load(f)

th_sentences = [[tuple(pair) for pair in sentence] for sentence in test_hard_sentences]
test_sentences = copy.deepcopy(th_sentences)
#for i in range(len(th_sentences[0][0])):
#    print(f"{th_sentences[0][0][i][0]:<15} {th_sentences[0][0][i][1]:<10} {th_sentences[0][1][i][1]:<10}")
test_sentence = ' '.join(tup[0] for tup in test_sentences[0][0])
print(test_sentence)


def untag(tagged_sentence):
    return [w for w, _ in tagged_sentence]

def untag_pos(tagged_sentence):
    return [t for _, t in tagged_sentence]
# 
gemini_model = 'gemini-2.0-flash-lite'

# --- Define Pydantic Models for Structured Output ---

# --- Define the Universal Dependencies POS Tagset (17 core tags) as an enum ---
# class UDPosTag(str, Enum):
    # TODO

# TODO Define more Pydantic models for structured output
class TokenPOS(BaseModel):
    token: str = Field(description="The word/token in the sentence.")
    pos: str = Field(description="The part-of-speech tag for the token.")

class SentencePOS(BaseModel):
    tokens: List[TokenPOS] = Field(description="List of tokens with POS tags for a single sentence.")

class TaggedSentences(BaseModel):
    """Represents a list of sentences with their tagged tokens."""
    sentences: List[SentencePOS] = Field(description="A list of sentences, each containing tagged tokens.")

class ErrorAnalysis(BaseModel):
    """Represents a token , its correct and predicted tags and an explanation of the error and error category."""
    word: str #Annotated[str, Field(desription="The word/token in the sentence.")]
    correct_tag: str #Annotated[str, Field(description="The correct part-of-speech tag for the token.")]
    predicted_tag: str #Annotated[str , Field(description="The predicted part-of-speech tag for the token.")]
    explanation: str #Annotated[str , Field(description="A one paragraph explanation of why this mismatch might occur in tagging this token within this sentence")]
    category: str #Annotated[str , Field(description="Error category, to be used for further analysis of most common error categories")]

class FullAnalysis(BaseModel):
    """represents a list of Error Analysis"""
    errors: List[ErrorAnalysis]

# --- Configure the Gemini API ---
# Get a key https://aistudio.google.com/plan_information 
# Use os.environ.get for production environments.
# For Colab/AI Studio, you might use userdata.get
# Example:
# from google.colab import userdata
# GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
# genai.configure(api_key=GOOGLE_API_KEY)

# Make sure to replace "YOUR_API_KEY" with your actual key if running locally
# and not using environment variables or userdata.
try:
    # Attempt to get API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # Fallback or specific instruction for local setup
        # Replace with your actual key if needed, but environment variables are safer
        api_key = "YOUR_API_KEY"
        if api_key == "YOUR_API_KEY":
           print("⚠️ Warning: API key not found in environment variables. Using placeholder.")
           print("   Please set the GOOGLE_API_KEY environment variable or replace 'YOUR_API_KEY' in the code.")

    genai.configure(api_key=api_key)

except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    print("Please ensure you have a valid API key set.")
    # Depending on the environment, you might want to exit here
    # import sys
    # sys.exit(1)


# --- Function to Perform POS Tagging ---

def tag_sentences_ud(text_to_tag: str) -> Optional[TaggedSentences]:
    """
    Performs POS tagging on the input text using the Gemini API and
    returns the result structured according to the SentencePOS Pydantic model.

    Args:
        text_to_tag: The sentence or text to be tagged.

    Returns:
        A TaggedSentences object containing the tagged tokens, or None if an error occurs.
    """
    # Construct the prompt
    prompt = f"""Tag the following 5 sentences separated by the delimiter @@@ using the Universal POS tagset which are the following: 
    ADJ: adjective ADP: 
    adposition ADV: 
    adverb AUX: auxiliary
    CCONJ: coordinating conjunction
    DET: determiner
    INTJ: interjection
    NOUN: noun
    NUM: numeral
    PART: particle
    PRON: pronoun
    PROPN: proper noun
    PUNCT: punctuation
    SCONJ: subordinating conjunction
    SYM: symbol
    VERB: verb
    X: other. Return a list of (token, tag) pairs in JSON format. Sentences: "{text_to_tag}"""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': TaggedSentences,
        },
    )
    # Use the response as a JSON string.
    print(response.text)

    # Use instantiated objects.
    res: TaggedSentences = response.parsed
    return res

#----function to perform error analysis
def error_analysis_ud(errors_to_analyze: str) -> Optional[FullAnalysis]:
    # Construct the prompt
    prompt = f"""Here are 3 sentences, each tagged correctly with Universal POS , then tagged again with a POS tagger prediction.
    Perform an analysis of the errors in the POS tagger - where there is a mismatch between the correct tag and the predicted tag of the same token.
    Provide an explanation and categorize each error. Skip the word/token if both tags are the same.
    Return a list of object with the fields: word, correct_tag, predicted_tag, explanation and category in JSON format.
    word field should be the token with the tagging mismatch, explanation is your analysis of the error, and category the error category.
    Here is the list:
    {errors_to_analyze}"""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': FullAnalysis,
        },
    )
    # Use the response as a JSON string.
    print(response.text)

    # Use instantiated objects.
    res: FullAnalysis = response.parsed
    return res

if __name__ == "__main__":
    # example_text = "The quick brown fox jumps over the lazy dog."
    #example_text = "What if Google expanded on its search-engine (and now e-mail) wares into a full-fledged operating system?"
    # example_text = "Google Search is a web search engine developed by Google LLC."
    # example_text = "החתול המהיר קופץ מעל הכלב העצלן." # Example in Hebrew

#    print(f"\nTagging text: \"{example_text}\"")

 #   tagged_result = tag_sentences_ud(example_text)

  #  if tagged_result:
   #     print("\n--- Tagging Results ---")
   #     for s in tagged_result.sentences:
            
    #        for tokenpos in s:
     #           token = tokenpos[0]
      #          tag = tokenpos[1]
       #         # Handle potential None for pos_tag if model couldn't assign one
        #        ctag = tag if tag is not None else "UNKNOWN"
         #       print(f"Token: {token:<15} {str(ctag)}")
          #      print("----------------------")
   # else:
    #    print("\nFailed to get POS tagging results.")

    if os.path.exists("test_hard_sentences_gemini.json"):
        print("File already exists. Skipping Gemini prompt task.")
    else:
        print("File test_hard_sentences_gemini.json not found, starting pos tagging for hard sentences using gemini api..")

        for i in range(0, len(test_sentences), 5):
            prompt_sentences = []
            for j in range(5):
                if(i+j < len(test_sentences)):
                    prompt_sentences.append(' '.join(tup[0] for tup in test_sentences[i+j][0]))
            
            tagged_result = tag_sentences_ud(" @@@ ".join(prompt_sentences))

            if tagged_result:
                print("\n--- Tagging Results ---")
                for j in range(len(tagged_result.sentences)):
                    s = tagged_result.sentences[j]
                    gemini_tagged_sentence = []
                    for tokenpos in s.tokens:
                        gemini_tagged_sentence.append((tokenpos.token, tokenpos.pos))
                    test_sentences[i+j].insert(2, gemini_tagged_sentence)
                    for tokenpos in s:
                        token = tokenpos[0]
                        tag = tokenpos[1]
                        # Handle potential None for pos_tag if model couldn't assign one
                        ctag = tag if tag is not None else "UNKNOWN"
                        print(f"Token: {token:<15} {str(ctag)}")
                        print("----------------------")
            else:
                print("\nFailed to get POS tagging results.")


        with open("test_hard_sentences_gemini.json", "w") as f:
            json.dump(test_sentences, f)
    
    with open("test_hard_sentences_gemini.json", "r") as f:
        task = json.load(f)
    task_sentences = [[tuple(pair) for pair in sentence] for sentence in task]

    error_analysis = []
    for j in range(len(task_sentences)):
        errors=0 
        for i in range(len(task_sentences[j][0])):
        
            token = task_sentences[j][0][i][0]
            tag = task_sentences[j][0][i][1]
            pred_tag = task_sentences[j][1][i][1]
            gemini_tag = "ERROR"
            if len(task_sentences[j]) > 2 and i<len(task_sentences[j][2]) and task_sentences[j][2][i][0] == token:
                gemini_tag = task_sentences[j][2][i][1]
                if tag!=gemini_tag: errors+=1
        
        if errors >=3 and len(error_analysis) < 30 : 
            error_analysis.append((task_sentences[j][0], task_sentences[j][2]))
    

    if os.path.exists("error_analysis.json"):
        print("File already exists. Skipping Gemini prompt task.")
    else:
        print("File error_analysis.json not found, starting pos tagging for hard sentences using gemini api..")
        full_analysis = []
        for i in range(0, len(error_analysis), 3):
            prompt_sentences = []
            for j in range(3):
                if(i+j < len(error_analysis)):
                    prompt_sentences.append("correct tags: ")
                    prompt_sentences.append(' '.join(f"{tup[0]}/{tup[1]}" for tup in error_analysis[i+j][0]))
                    prompt_sentences.append("predicted tags: ")
                    prompt_sentences.append(' '.join(f"{tup[0]}/{tup[1]}" for tup in error_analysis[i+j][1]))
            
            analysis_result = error_analysis_ud("\n".join(prompt_sentences))

            if analysis_result:
                for ea in analysis_result.errors:
                    full_analysis.append(ea)
            else:
                print("\nFailed to get POS tagging results.")
            
        json_ready = [e.model_dump() for e in full_analysis]
        with open("error_analysis.json", "w") as f:
            json.dump(json_ready, f)
    if os.path.exists("task_3_1.json"):
        print("File already exists. Skipping Gemini prompt task.")
    else:
        task_3_1 = tag_sentences_ud("What if Google expanded on its search - engine ( and now e-mail ) wares into a full - fledged operating system ? @@@ What if Google expanded on its search-engine (and now e-mail) wares into a full-fledged operating system?")
        tokenized = []
        original = []
        tokenizedObject = task_3_1.sentences[0]
        originalObject = task_3_1.sentences[1]
        for tokenpos in tokenizedObject.tokens:
            tokenized.append((tokenpos.token, tokenpos.pos))
        for tokenpos in originalObject.tokens:
            original.append((tokenpos.token , tokenpos.pos))  
        with open("task_3_1.json", "w") as f:
            json.dump([tokenized, original], f)
    


        
    



